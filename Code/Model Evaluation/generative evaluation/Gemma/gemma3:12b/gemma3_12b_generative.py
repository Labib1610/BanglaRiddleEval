#!/usr/bin/env python3
"""
gemma3_12b_generative.py

Generative evaluation pipeline for Bengali riddles using Ollama gemma3:12b model.
- Evaluates open-ended riddle answer generation on full dataset
- Computes BERTScore and LLM-as-a-judge metrics
- Uses Google Gemini API for LLM judging
- Includes zero-shot, few-shot, and chain-of-thought (CoT) prompting modes
- Semantic similarity via multilingual BERTScore for meaning comparison
"""

import os
import json
import time
import re
import unicodedata
import difflib
import random
import socket
from pathlib import Path
from tqdm import tqdm



# BERTScore import with fallback
try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️ bert_score not found. Using fallback scoring.")
    print("Install with: pip install bert_score")
    BERTSCORE_AVAILABLE = False
    BERTScorer = None

# Ollama client import
try:
    from ollama import Client
except Exception as e:
    raise RuntimeError("ollama client not available. Install the ollama client package.") from e

# Google Gemini API imports
try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai package not found. Install it with:")
    print("pip install google-generativeai")
    exit(1)

# ==================== CONFIG ====================
# Ollama Configuration
LLM_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:12b"
LLM_NUM_CTX = 4096
LLM_SEED = 0

# Google Gemini API Configuration for LLM-as-a-Judge
KEY_LIST = [
    # Add your Google Gemini API keys here
]

JUDGE_MODEL_NAME = "gemini-2.5-flash"
NETWORK_TIMEOUT_SECONDS = 120

# Dataset and Output Configuration
RIDDLES_DATASET_PATH = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/main_dataset/riddles.json")
OUTPUT_ROOT = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/Code/Model Evaluation (Benchmark)/generative evaluation/Gemma/gemma3:12b")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ==================== PROMPT TEMPLATES ====================
PROMPT_ZERO_SHOT = (
    "You are an AI assistant that solves Bengali riddles.\n"
    "Task: Read the Bengali riddle below and provide the most accurate answer in Bengali.\n\n"
    "Riddle: {riddle}\n\n"
    "Response Requirements:\n"
    "- Provide ONLY the answer in Bengali text\n"
    "- Do NOT add any explanations, reasoning, or extra text\n"
    "- Answer must be a single Bengali word or short Bengali phrase\n"
    "- Do NOT use English words or explanations\n"
    "- Respond with ONLY the Bengali answer\n\n"
    "- Follow the exact format: Answer: <bengali_text>\n\n"
    "Answer:"
)

PROMPT_FEW_SHOT = (
    "You are an AI assistant that solves Bengali riddles.\n"
    "Task: Read the Bengali riddle below and provide the most accurate answer in Bengali.\n\n"
    "Examples:\n"
    "Riddle: বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।\n"
    "Answer: আনারস\n\n"
    "Riddle: কোন জিনিস কাটলে বাড়ে?\n"
    "Answer: পুকুর\n\n"
    "Riddle: এই ঘরে যাই, ওই ঘরে যাই দুম দুমিয়ে আছাড় খাই।\n"
    "Answer: ঘাঁটা\n\n"
    "Now solve this riddle following the same format:\n"
    "Riddle: {riddle}\n\n"
    "Response Requirements:\n"
    "- Provide ONLY the Bengali answer like the examples above\n"
    "- Do NOT add any explanations, reasoning, or extra text\n"
    "- Answer must be a single Bengali word or short Bengali phrase\n"
    "- Do NOT use English words or explanations\n"
    "- Follow the exact format: Answer: <bengali_text>\n\n"
    "Answer:"
)

PROMPT_CHAIN_OF_THOUGHTS = (
    "You are an AI assistant that solves Bengali riddles using step-by-step reasoning.\n"
    "Task: Read the Bengali riddle below and provide the most accurate answer using chain-of-thought reasoning.\n\n"
    "Riddle: {riddle}\n\n"
    "Response Format Requirements:\n"
    "- Think step by step about the riddle's metaphors and meaning\n"
    "- Write reasoning steps in English but think with Bengali cultural knowledge\n"
    "- Final answer MUST be in Bengali and ONLY Bengali text\n"
    "- Follow this EXACT format without deviation:\n\n"
    "Reasoning_En:\n"
    "Step 1: <analyze the riddle's key words and metaphors>\n"
    "Step 2: <connect observations with possible meanings>\n"
    "Step 3: <eliminate wrong possibilities with brief reasoning>\n"
    "Step 4: <explain why the final choice is correct>\n\n"
    "Final Answer: <single_bengali_word_or_phrase>\n\n"
    "IMPORTANT: The Final Answer line must contain ONLY Bengali text, no English explanations or additional text."
)

# LLM-as-a-Judge prompt template
LLM_JUDGE_PROMPT = (
    "You are an expert evaluator for Bengali riddle answers. Your task is to score how correct a predicted answer is for a given riddle.\n\n"
    "Riddle: {riddle}\n"
    "Ground Truth Answer: {ground_truth}\n"
    "Predicted Answer: {predicted}\n\n"
    "Evaluation Criteria:\n"
    "1. Exact Match: Are the answers exactly the same?\n"
    "2. Semantic Equivalence: Do they refer to the same concept/object?\n"
    "3. Cultural Context: Consider Bengali cultural and linguistic variations\n"
    "4. Acceptable Synonyms: Different Bengali words for the same thing\n"
    "5. Spelling Variations: Minor spelling differences in Bengali\n"
    "6. Partial Correctness: Consider partial matches as valid (e.g., if ground truth is 'আনারস' and predicted is 'আনার', give partial credit)\n\n"
    "Scoring Instructions:\n"
    "- Give a score between 0 and 1\n"
    "- 1.0: Perfect match (exact or semantically equivalent)\n"
    "- 0.7-0.9: Very close match (minor variations, synonyms, or slight differences)\n"
    "- 0.4-0.6: Partial match (partially correct but missing important parts)\n"
    "- 0.1-0.3: Poor match (completely different but some tiny relevance)\n"
    "- 0.0: Completely wrong or no answer\n\n"
    "Respond with ONLY the numerical score (e.g., 0.8, 0.5, 1.0, 0.0)\n\n"
    "Score:"
)

# ==================== GEMINI CLIENT WITH KEY ROTATION ====================
class RotatingGeminiClient:
    def __init__(self, key_list, model_name):
        assert key_list, "Provide at least one API key"
        self.keys = key_list
        self.model_name = model_name
        self.key_index = 0
        self.key_usage_count = {key: 0 for key in self.keys}
        self.failed_keys = set()
        self._configure_current_key()

    def _configure_current_key(self):
        key = self.keys[self.key_index]
        genai.configure(api_key=key)
        print(f"➡️ Using Judge API key index {self.key_index}")
        
        try:
            socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"⚠️ Could not configure socket timeout: {e}")

    def _advance_key(self):
        old = self.key_index
        self.key_index = (self.key_index + 1) % len(self.keys)
        print(f"🔁 Switching Judge API key: {old} -> {self.key_index}")
        self._configure_current_key()

    def judge_answer(self, riddle, ground_truth, predicted, max_attempts=5):
        """Judge predicted answer using LLM-as-a-judge with 0-1 scoring."""
        prompt = LLM_JUDGE_PROMPT.format(
            riddle=riddle,
            ground_truth=ground_truth,
            predicted=predicted
        )
        
        attempt = 0
        last_exc = None
        while attempt < max_attempts:
            attempt += 1
            try:
                current_key = self.keys[self.key_index]
                self.key_usage_count[current_key] += 1
                
                model = genai.GenerativeModel(self.model_name)
                resp = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Low temperature for consistent judging
                        top_k=10,
                        top_p=0.8,
                        max_output_tokens=20,  # Very short responses expected (just a number)
                    )
                )
                
                if hasattr(resp, "text") and resp.text:
                    result = resp.text.strip()
                elif hasattr(resp, "candidates") and resp.candidates:
                    result = resp.candidates[0].content.parts[0].text.strip()
                else:
                    result = "0.0"
                
                # Parse numerical score
                score = self._parse_score(result)
                return score
                
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "rate limit" in msg:
                    print(f"❗ Judge quota/rate-limit on key {self.key_index}: {e}")
                    self.failed_keys.add(self.keys[self.key_index])
                    self._advance_key()
                    sleep_time = 10 + random.uniform(5, 15)
                    print(f"⏳ Judge rate limit - backing off for {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    continue
                elif any(keyword in msg for keyword in ['timeout', 'connection', 'ssl', 'socket', 'network']):
                    print(f"❗ Judge network error on key {self.key_index} (attempt {attempt}/{max_attempts}): {e}")
                    self._advance_key()
                    sleep_time = 5 * (2 ** (attempt - 1))
                    print(f"⏳ Judge network issue - backing off for {sleep_time:.1f}s")
                    time.sleep(min(sleep_time, 120))
                    continue
                
                print(f"❗ Judge API call failed on key {self.key_index} (attempt {attempt}/{max_attempts}): {e}")
                self._advance_key()
                sleep_time = 2 * (2 ** (attempt - 1))
                time.sleep(min(sleep_time, 60))
                continue

        print(f"❌ Judge failed after {max_attempts} attempts. Last error: {last_exc}")
        return 0.0  # Default to 0.0 when judge fails

    def _parse_score(self, response_text):
        """Parse numerical score from judge response."""
        import re
        
        # Clean the response
        text = response_text.strip()
        
        # Look for decimal number (0.0 to 1.0)
        decimal_match = re.search(r'\b([0-1](?:\.[0-9]+)?)\b', text)
        if decimal_match:
            try:
                score = float(decimal_match.group(1))
                # Ensure score is between 0 and 1
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # Look for percentage (0% to 100%)
        percent_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)%\b', text)
        if percent_match:
            try:
                percent = float(percent_match.group(1))
                # Convert percentage to 0-1 scale
                return max(0.0, min(1.0, percent / 100.0))
            except ValueError:
                pass
        
        # Look for fraction (e.g., 3/4, 1/2)
        fraction_match = re.search(r'\b([0-9]+)/([0-9]+)\b', text)
        if fraction_match:
            try:
                numerator = float(fraction_match.group(1))
                denominator = float(fraction_match.group(2))
                if denominator > 0:
                    return max(0.0, min(1.0, numerator / denominator))
            except ValueError:
                pass
        
        # Fallback: try to extract any number and normalize
        number_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', text)
        if number_match:
            try:
                num = float(number_match.group(1))
                # If number is > 1, assume it's a percentage
                if num > 1:
                    return max(0.0, min(1.0, num / 100.0))
                else:
                    return max(0.0, min(1.0, num))
            except ValueError:
                pass
        
        print(f"⚠️ Could not parse score from: '{text}', defaulting to 0.0")
        return 0.0

# ==================== HELPER FUNCTIONS ====================
def load_riddles_data(path: Path):
    """Load riddles data from JSON file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []



def normalize_text(text):
    """Normalize Bengali text for comparison."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text.strip())

def extract_answer_from_response(response):
    """Extract answer from model response."""
    if not response:
        return ""
    
    response_text = ""
    
    # Handle different response types
    if isinstance(response, dict) and 'response' in response:
        response_text = response['response']
    elif isinstance(response, str):
        # Check if it's a string representation of the object
        if "response='" in response:
            # Extract from string representation like "response='শূন্য'"
            response_match = re.search(r"response='([^']+)'", response)
            if response_match:
                response_text = response_match.group(1)
            else:
                response_text = response
        else:
            response_text = response
    else:
        # Last resort: convert to string and try to extract
        response_str = str(response)
        if "response='" in response_str:
            response_match = re.search(r"response='([^']+)'", response_str)
            if response_match:
                response_text = response_match.group(1)
            else:
                response_text = response_str
        else:
            response_text = response_str
    
    if not response_text:
        return ""
        
    response_text = response_text.strip()
    
    # Enhanced parsing for all modes - try multiple patterns
    answer_patterns = [
        r'Final\s*Answer\s*:\s*([^\n\r]+)',  # CoT Final Answer
        r'Answer\s*:\s*([^\n\r]+)',  # Standard Answer format
        r'\u0989\u09a4\u09cd\u09a4\u09b0\s*:\s*([^\n\r]+)',  # Bengali "উত্তর:"
        r'Final\s*Answer\s*:\s*\*\*([^*]+)\*\*',  # Markdown bold
        r'Answer\s*:\s*\*\*([^*]+)\*\*',  # Markdown bold for Answer
        r'Final\s*Answer\s*:\s*(.+?)(?:\n|$)',  # Until newline
        r'Answer\s*:\s*(.+?)(?:\n|$)',  # Until newline for Answer
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # Clean up common formatting
            answer = re.sub(r'^\*\*|\*\*$', '', answer)  # Remove markdown bold
            answer = re.sub(r'^["\']|["\']$', '', answer)  # Remove quotes
            answer = re.sub(r'^\s*-\s*', '', answer)  # Remove leading dash
            answer = answer.split('\n')[0].strip()  # Take only first line
            
            # Prefer Bengali answers, but accept any non-empty clean answer
            if answer:
                if is_bengali_text(answer):
                    return answer
                # Store as fallback if no Bengali found later
                elif not hasattr(extract_answer_from_response, '_fallback_answer'):
                    extract_answer_from_response._fallback_answer = answer
    
    # Fallback 1: Return stored fallback answer if we found one
    if hasattr(extract_answer_from_response, '_fallback_answer'):
        fallback = extract_answer_from_response._fallback_answer
        delattr(extract_answer_from_response, '_fallback_answer')
        if fallback:
            return fallback
    
    # Fallback 2: Find Bengali text in the response
    bengali_pattern = r'[\u0980-\u09FF]+(?:\s+[\u0980-\u09FF]+)*'
    bengali_matches = re.findall(bengali_pattern, response_text)
    if bengali_matches:
        # Return the longest Bengali match (likely the answer)
        longest_match = max(bengali_matches, key=len).strip()
        if len(longest_match) >= 2:  # Ensure it's not just a single character
            return longest_match
    
    # Fallback 3: Extract first meaningful line
    lines = [line.strip() for line in response_text.split('\n') if line.strip() and len(line.strip()) >= 2]
    if lines:
        # Try to find a line that looks like an answer (short and meaningful)
        for line in lines:
            if len(line) <= 50 and not line.lower().startswith(('the ', 'this ', 'i ', 'let', 'so ', 'but ')):
                return line
        # If no good line found, return the first line
        return lines[0]
    
    # Last resort: return cleaned response if it's short enough
    if len(response_text) <= 100:
        return response_text
    
    return ""

def extract_reasoning_from_response(response):
    """Extract reasoning steps from CoT response."""
    if not response:
        return [], ""
    
    # Look for reasoning section
    reasoning_match = re.search(r'Reasoning_En:\s*(.*?)(?:Final\s*Answer:|$)', response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        
        # Extract individual steps
        steps = []
        for match in re.finditer(r'Step\s*\d+:\s*(.*?)(?=Step\s*\d+:|$)', reasoning_text, re.DOTALL):
            step = match.group(1).strip()
            if step:
                steps.append(step)
        
        return steps, reasoning_text
    
    return [], ""

def is_bengali_text(text):
    """Check if text contains primarily Bengali characters."""
    if not text:
        return False
    bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
    total_chars = sum(1 for char in text if char.isalpha())
    return total_chars > 0 and (bengali_chars / total_chars) > 0.5

def compute_bert_score(predicted, ground_truth):
    """Compute BERTScore between predicted and ground truth answers."""
    if not BERTSCORE_AVAILABLE:
        # Return zero scores if BERTScore not available
        print("⚠️ BERTScore not available, returning default scores")
        return 0.0, 0.0, 0.0
    
    try:
        scorer = BERTScorer(model_type="bert-base-multilingual-cased", lang="other")
        
        # Handle empty predictions
        pred_text = predicted.strip() if predicted else "<empty>"
        gt_text = ground_truth.strip() if ground_truth else "<empty>"
        
        precision, recall, f1 = scorer.score([pred_text], [gt_text])
        
        return float(precision[0]), float(recall[0]), float(f1[0])
    except Exception as e:
        print(f"⚠️ BERTScore computation failed: {e}")
        return 0.0, 0.0, 0.0

# ==================== OLLAMA WRAPPER ====================
class OllamaLLM:
    def __init__(self, host: str, model: str, num_ctx: int = 4096, seed: int = 0):
        self.host = host
        self.model = model
        self.num_ctx = num_ctx
        self.seed = seed
        self.client = Client(host=self.host)

    def generate(self, prompt: str, max_tokens: int = None):
        """Generate text using Ollama."""
        options = {
            "seed": self.seed,
            "num_ctx": self.num_ctx,
        }
        
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        gen_args = {
            "model": self.model,
            "prompt": prompt,
            "options": options
        }

        resp = self.client.generate(**gen_args)
        
        # Extract response text consistently
        if isinstance(resp, dict) and 'response' in resp:
            response_text = resp['response']
        else:
            response_text = str(resp)
        
        return resp, response_text

# ==================== CORE PROCESSING ====================
def process_generative_evaluation(ollama_llm: OllamaLLM, gemini_judge: RotatingGeminiClient, prompt_mode="zero_shot"):
    """Process generative evaluation with specified prompt mode."""
    
    out_file = OUTPUT_ROOT / f"riddles_generative_gemma3_4b_{prompt_mode}.json"
    
    # Check if results already exist
    existing_results = []
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf8") as f:
                existing_results = json.load(f)
            print(f"📂 Found {len(existing_results)} existing results")
        except Exception as e:
            print(f"⚠️ Error loading existing results: {e}")
            existing_results = []
    
    # Create set of processed IDs
    processed_ids = {result.get("riddle_id") for result in existing_results}
    
    # Load riddles data
    riddles_data = load_riddles_data(RIDDLES_DATASET_PATH)
    if not riddles_data:
        print("❌ No riddles data found!")
        return
    
    print(f"🎯 Using full dataset with {len(riddles_data)} examples")
    
    # Filter to unprocessed items
    remaining_data = [item for item in riddles_data if item.get("riddle_id") not in processed_ids]
    
    if not remaining_data:
        print("✅ All examples already processed!")
        # Compute metrics from existing results
        llm_judge_scores = [r.get("llm_judge_score", 0.0) for r in existing_results if "llm_judge_score" in r]
        bert_f1_scores = [r.get("bert_f1", 0.0) for r in existing_results if "bert_f1" in r]
        reasoning_count = sum(1 for r in existing_results if r.get("reasoning_steps_en"))
        
        total = len(existing_results)
        avg_llm_judge_score = (sum(llm_judge_scores) / len(llm_judge_scores)) if llm_judge_scores else 0.0
        avg_bert_f1 = (sum(bert_f1_scores) / len(bert_f1_scores)) if bert_f1_scores else 0.0
        llm_judge_acc = (avg_llm_judge_score * 100) if avg_llm_judge_score > 0 else 0
        
        metrics = {
            "LLM Judge Average Score": round(avg_llm_judge_score, 3),
            "LLM Judge Accuracy (%)": round(llm_judge_acc, 2),
            "Average BERTScore F1": round(avg_bert_f1, 3),
            "n_examples_total": total,
            "avg_judge_score": round(avg_llm_judge_score, 3),
            "avg_bert_f1": round(avg_bert_f1, 3),
            "n_with_reasoning": reasoning_count
        }
        
        metrics_out = OUTPUT_ROOT / f"riddles_generative_metrics_gemma3_4b_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics computed from existing results: LLM Judge = {avg_llm_judge_score:.3f}, BERTScore F1 = {avg_bert_f1:.3f}")
        return
    
    print(f"🔄 Processing {len(remaining_data)} remaining examples...")
    
    # Select prompt template
    if prompt_mode == "zero_shot":
        prompt_template = PROMPT_ZERO_SHOT
    elif prompt_mode == "few_shot":
        prompt_template = PROMPT_FEW_SHOT
    elif prompt_mode == "chain_of_thought":
        prompt_template = PROMPT_CHAIN_OF_THOUGHTS
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    
    results = existing_results.copy()
    
    # Get all predictions for batch BERTScore processing
    all_predictions = []
    all_ground_truths = []
    
    # Process remaining examples
    for item in tqdm(remaining_data, desc=f"Generative Evaluation ({prompt_mode})"):
        riddle_id = item.get("riddle_id")
        riddle_text = item.get("riddle", "")
        ground_truth = item.get("ans", "")
        
        if not riddle_text or not ground_truth:
            print(f"⚠️ Skipping item {riddle_id}: missing riddle or answer")
            continue
        
        # Create prompt
        prompt = prompt_template.format(riddle=riddle_text)
        
        # Get model response
        max_attempts = 3
        response = ""
        raw_resp = None
        for attempt in range(max_attempts):
            try:
                raw_resp, response = ollama_llm.generate(prompt)
                break
            except Exception as e:
                print(f"❗ Ollama generate failed (attempt {attempt+1}/{max_attempts}): {e}")
                time.sleep(3 * (attempt + 1))
                if attempt == max_attempts - 1:
                    response = ""
                    raw_resp = None
        
        # Extract answer and reasoning
        predicted_answer = extract_answer_from_response(response)
        
        reasoning_steps = []
        reasoning_text = ""
        
        if prompt_mode == "chain_of_thought":
            reasoning_steps, reasoning_text = extract_reasoning_from_response(response)
        

        
        # Store for batch BERTScore processing
        all_predictions.append(predicted_answer.strip() if predicted_answer else "<empty>")
        all_ground_truths.append(ground_truth.strip() if ground_truth else "<empty>")
        
        # Get LLM judge evaluation
        if predicted_answer.strip():
            llm_judge_score = gemini_judge.judge_answer(riddle_text, ground_truth, predicted_answer)
        else:
            llm_judge_score = 0.0
        
        # Create result (BERTScore will be added later)
        result = {
            "riddle_id": riddle_id,
            "riddle": riddle_text,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "llm_judge_score": llm_judge_score,
            "bert_precision": 0.0,  # Placeholder
            "bert_recall": 0.0,     # Placeholder  
            "bert_f1": 0.0,         # Placeholder
            "reasoning_text": reasoning_text,
            "reasoning_steps_en": reasoning_steps
        }
        
        results.append(result)
        
        # Save incrementally
        with open(out_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Rate limiting
        time.sleep(5 + random.uniform(3, 5))
    
    # Batch BERTScore computation
    print("🧠 Computing BERTScore for all predictions...")
    if BERTSCORE_AVAILABLE and all_predictions:
        try:
            scorer = BERTScorer(model_type="bert-base-multilingual-cased", lang="other")
            precision_scores, recall_scores, f1_scores = scorer.score(all_predictions, all_ground_truths)
            
            # Update results with BERTScore
            new_results_start = len(results) - len(all_predictions)
            for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
                if new_results_start + i < len(results):
                    results[new_results_start + i]["bert_precision"] = float(p)
                    results[new_results_start + i]["bert_recall"] = float(r) 
                    results[new_results_start + i]["bert_f1"] = float(f)
            
        except Exception as e:
            print(f"⚠️ Batch BERTScore computation failed: {e}")
            # Fill with default scores as fallback
            new_results_start = len(results) - len(all_predictions)
            for i in range(len(all_predictions)):
                if new_results_start + i < len(results):
                    results[new_results_start + i]["bert_precision"] = 0.0
                    results[new_results_start + i]["bert_recall"] = 0.0
                    results[new_results_start + i]["bert_f1"] = 0.0
    
    # Final save with BERTScore
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Compute final metrics
    llm_judge_scores = [r.get("llm_judge_score", 0.0) for r in results if "llm_judge_score" in r]
    bert_f1_scores = [r.get("bert_f1", 0.0) for r in results if "bert_f1" in r]
    reasoning_count = sum(1 for r in results if r.get("reasoning_steps_en"))
    
    total = len(results)
    avg_llm_judge_score = (sum(llm_judge_scores) / len(llm_judge_scores)) if llm_judge_scores else 0.0
    avg_bert_f1 = (sum(bert_f1_scores) / len(bert_f1_scores)) if bert_f1_scores else 0.0
    llm_judge_acc = (avg_llm_judge_score * 100) if avg_llm_judge_score > 0 else 0

    metrics = {
        "LLM Judge Average Score": round(avg_llm_judge_score, 3),
        "LLM Judge Accuracy (%)": round(llm_judge_acc, 2),
        "Average BERTScore F1": round(avg_bert_f1, 3),
        "n_examples_total": total,
        "avg_judge_score": round(avg_llm_judge_score, 3),
        "avg_bert_f1": round(avg_bert_f1, 3),
        "n_with_reasoning": reasoning_count
    }

    metrics_out = OUTPUT_ROOT / f"riddles_generative_metrics_gemma3_4b_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Finished Generative evaluation ({prompt_mode})")
    print(f"Results: {out_file}, Metrics: {metrics_out}")
    print(f"LLM Judge Average Score: {avg_llm_judge_score:.3f}/1.0 ({llm_judge_acc:.2f}%)")
    print(f"Average BERTScore F1: {avg_bert_f1:.3f} ({avg_bert_f1 * 100:.2f}%)")

def main():
    """Main function to run generative evaluation."""
    print("🎯 Bengali Riddle Generative Evaluation - Gemma3:12b")
    print("=" * 60)
    
    # Initialize Ollama client
    ollama_llm = OllamaLLM(host=LLM_URL, model=MODEL_NAME, num_ctx=LLM_NUM_CTX, seed=LLM_SEED)
    
    # Initialize Gemini judge client
    gemini_judge = RotatingGeminiClient(KEY_LIST, JUDGE_MODEL_NAME)
    
    # Run evaluations for all three modes
    modes = ["zero_shot", "few_shot", "chain_of_thought"]
    
    for mode in modes:
        print(f"\n🚀 Starting {mode} evaluation...")
        try:
            process_generative_evaluation(ollama_llm, gemini_judge, mode)
        except Exception as e:
            print(f"❌ Error in {mode} evaluation: {e}")
            continue
    
    print("\n✅ All evaluations completed!")
    
    # Print judge usage statistics
    print("\n📊 LLM Judge API Key Usage Statistics:")
    for i, key in enumerate(gemini_judge.keys):
        usage = gemini_judge.key_usage_count[key]
        status = "❌ FAILED" if key in gemini_judge.failed_keys else "✅ OK"
        print(f"Key {i}: {usage} requests - {status}")

if __name__ == "__main__":
    main()