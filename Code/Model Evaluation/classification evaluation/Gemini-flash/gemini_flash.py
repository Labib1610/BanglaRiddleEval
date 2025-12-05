#!/usr/bin/env python3
"""
gemini_flash.py

Semantic Ambiguity evaluation pipeline for Bengali riddles using Google Gemini 2.5 Flash model.
- Evaluates multiple choice questions about semantic ambiguity in riddle terms
- Analyzes riddle context, answer, and ambiguous word meanings
- Extracts reported index and answer text from model responses
- Reconciles index vs text conflicts with exact and fuzzy matching
- Saves JSON outputs and accuracy metrics
- Includes zero-shot, few-shot, and chain-of-thought (CoT) prompting modes
- Uses rotating API keys to handle rate limits
- Processes entire dataset (no sampling)
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

# Google Gemini API imports
try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai package not found. Install it with:")
    print("pip install google-generativeai")
    exit(1)

# ==================== CONFIG ====================
# Google Gemini API Configuration
# Add your API keys to this list
KEY_LIST = [
    # "your-api-key-1",
    # "your-api-key-2",
    # Add more keys as needed
]

MODEL_NAME = "gemini-2.5-flash"  # Use the latest Gemini model
NETWORK_TIMEOUT_SECONDS = 120

# Dataset and Output Configuration
DATASET_PATH = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/dataset/riddles_semantic_ambiguity.json")
OUTPUT_ROOT = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/Code/Model Evaluation (Benchmark)/classification evaluation/Gemini-flash")

# Processing entire dataset - no sampling

# ==================== PROMPT TEMPLATES ====================
PROMPT_ZERO_SHOT = (
    "You are an AI assistant that analyzes Bengali riddle semantic ambiguity questions.\n"
    "Task:\n"
    "1. Read the original riddle: {riddle}\n"
    "2. Read the semantic question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer** that best explains the semantic meaning.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Bengali text for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n\n"
    "Index: <option_index>, Answer: \"<option_text_in_Bengali>\""
)

PROMPT_FEW_SHOT = (
    "You are an AI assistant that analyzes Bengali riddle semantic ambiguity questions.\n"
    "Task:\n"
    "1. Read the original riddle: {riddle}\n"
    "2. Read the semantic question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer** that best explains the semantic meaning.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Bengali text for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n"
    "Index: <option_index>, Answer: \"<option_text_in_Bengali>\"\n\n"
    "Examples:\n"
    "Riddle: \"বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।\"\n"
    "Answer: \"আনারস\"\n"
    "Question: \"এই ধাঁধায় 'সোনার টোপর' শব্দটি কী বোঝায়?\"\n"
    "Options: [\"সোনার তৈরি টুপি\", \"আনারসের শীর্ষের হলুদ অংশ\", \"পাখির গলা\", \"সোনার তৈরি মুকুট\"]\n"
    "Answer: Index: 1, Answer: \"আনারসের শীর্ষের হলুদ অংশ\"\n\n"
    "Riddle: \"এই ঘরে যাই, ওই ঘরে যাই দুম দুমিয়ে আছাড় খাই।\"\n"
    "Answer: \"ঘাঁটা\"\n"
    "Question: \"এই ধাঁধায় 'দুম' শব্দটি কী বোঝায়?\"\n"
    "Options: [\"ছোট পাখির গুঞ্জন\", \"গাড়ির হর্নের শব্দ\", \"বাতাসের গর্জন\", \"শূকরের গর্জন\"]\n"
    "Answer: Index: 0, Answer: \"ছোট পাখির গুঞ্জন\"\n\n"
    "Riddle: \"লাল টুকটুক ছোটমামা গায়ে পরে অনেক জামা।\"\n"
    "Answer: \"পেঁয়াজ\"\n"
    "Question: \"এই ধাঁধায় 'জামা' শব্দটি কী বোঝায়?\"\n"
    "Options: [\"পেঁয়াজের স্তর\", \"পোশাক\", \"বস্ত্র\", \"চামড়ার টুকরা\"]\n"
    "Answer: Index: 0, Answer: \"পেঁয়াজের স্তর\"\n\n"
    "Now answer for the given semantic question.\n"
)

PROMPT_CHAIN_OF_THOUGHTS = (
    "You are an AI assistant that analyzes Bengali riddle semantic ambiguity questions using step-by-step reasoning.\n\n"
    "Task:\n"
    "1. Read the original riddle: {riddle}\n"
    "2. Read the semantic question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer** using chain-of-thought reasoning.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Think step by step about the semantic ambiguity and metaphorical meanings.\n"
    "- Write reasoning steps in English but think with Bengali cultural knowledge.\n"
    "- Final answer must be in Bengali.\n"
    "- In Reasoning_En, write step-by-step reasoning in English — break down the analysis logically:\n"
    "  Step 1: Analyze the riddle's context and the given answer.\n"
    "  Step 2: Identify the ambiguous term and its possible meanings.\n"
    "  Step 3: Evaluate each option against the riddle's metaphorical context.\n"
    "  Step 4: Explain why the final choice best resolves the semantic ambiguity.\n"
    "- Be clear, concise, and factual (avoid overly lengthy explanations).\n"
    "- Follow this exact answer format:\n\n"
    "Reasoning_En:\n"
    "Step 1: <your_context_analysis>\n"
    "Step 2: <your_ambiguity_identification>\n"
    "Step 3: <your_option_evaluation>\n"
    "Step 4: <your_final_choice_explanation>\n\n"
    "Final Answer: Index: <option_index>, Answer: \"<option_text_in_Bengali>\""
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
        print(f"➡️ Using API key index {self.key_index}")
        
        # Configure client settings for better timeout handling
        try:
            socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"⚠️ Could not configure socket timeout: {e}")

    def _advance_key(self):
        old = self.key_index
        self.key_index = (self.key_index + 1) % len(self.keys)
        print(f"🔁 Switching API key: {old} -> {self.key_index}")
        self._configure_current_key()

    def ask_once(self, prompt):
        """One API attempt with currently configured key. Exceptions propagate."""
        try:
            model = genai.GenerativeModel(self.model_name)
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_k=40,
                    top_p=0.95,
                    max_output_tokens=1024,
                )
            )
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['timeout', 'timed out', 'connection', 'ssl', 'socket']):
                raise Exception(f"Network/timeout error during content generation: {e}")
            else:
                raise e
        
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        elif hasattr(resp, "candidates") and resp.candidates:
            cand_text = resp.candidates[0].content.parts[0].text
            return cand_text.strip()
        else:
            return "⚠️ Empty response"

    def ask_with_rotation(self, prompt, max_attempts_per_example=10, backoff_base=2.0):
        """
        Try up to max_attempts_per_example (across keys). On quota/429 or other errors,
        rotate to the next key and retry the same example.
        Enhanced sleep times to avoid PMR (50/min) and PDR (1000/day) limits.
        """
        attempt = 0
        last_exc = None
        while attempt < max_attempts_per_example:
            attempt += 1
            try:
                current_key = self.keys[self.key_index]
                self.key_usage_count[current_key] += 1
                return self.ask_once(prompt)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "rate limit" in msg or "quota exceeded" in msg:
                    print(f"❗ Quota/rate-limit detected on key index {self.key_index}: {e}")
                    self.failed_keys.add(self.keys[self.key_index])
                    self._advance_key()
                    sleep_time = 10 + random.uniform(5, 15)  # 15-25 seconds backoff
                    print(f"⏳ Rate limit - backing off for {sleep_time:.1f}s before retrying")
                    time.sleep(sleep_time)
                    continue
                elif any(keyword in msg for keyword in ['timeout', 'timed out', 'connection', 'ssl', 'socket', 'network']):
                    print(f"❗ Network/timeout error on key index {self.key_index} (attempt {attempt}/{max_attempts_per_example}): {e}")
                    self._advance_key()
                    sleep_time = backoff_base * (3 ** (attempt - 1))
                    print(f"⏳ Network issue - backing off for {sleep_time:.1f}s before retrying")
                    time.sleep(min(sleep_time, 180))
                    continue
                
                print(f"❗ API call failed on key index {self.key_index} (attempt {attempt}/{max_attempts_per_example}): {e}")
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"⏳ backing off for {sleep_time:.1f}s before retrying this example")
                time.sleep(min(sleep_time, 120))
                continue

        print(f"❌ Failed to get response after {max_attempts_per_example} attempts. Last error: {last_exc}")
        return "❌ Failed to get response"

# ==================== HELPER FUNCTIONS ====================
def load_mcq_data(path: Path):
    """Load MCQ data from JSON file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def normalize_text(text):
    """Normalize Bengali text for comparison."""
    if not isinstance(text, str):
        return ""
    # Normalize Unicode and strip whitespace
    return unicodedata.normalize('NFC', text.strip())

def extract_answer_from_response(response, options):
    """Extract index and answer from model response."""
    if not response or response in ["❌ Failed to get response", "⚠️ Empty response"]:
        return None, None, "Error response"
    
    response = response.strip()
    
    # Pattern to match "Index: X, Answer: "text""
    pattern = r'Index:\s*(\d+),\s*Answer:\s*["\']([^"\']+)["\']'
    match = re.search(pattern, response)
    
    if match:
        try:
            index = int(match.group(1))
            answer_text = match.group(2).strip()
            return index, answer_text, "Parsed"
        except (ValueError, IndexError):
            pass
    
    # Fallback: Look for just numbers
    numbers = re.findall(r'\d+', response)
    if numbers:
        try:
            index = int(numbers[0])
            if 0 <= index < len(options):
                return index, options[index], "Index only"
        except (ValueError, IndexError):
            pass
    
    return None, None, "Parse failed"

def fuzzy_match_answer(predicted_text, options, threshold=0.8):
    """Find best matching option using fuzzy string matching."""
    if not predicted_text:
        return None
    
    predicted_norm = normalize_text(predicted_text)
    best_match_idx = None
    best_score = 0
    
    for i, option in enumerate(options):
        option_norm = normalize_text(option)
        score = difflib.SequenceMatcher(None, predicted_norm, option_norm).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match_idx = i
    
    return best_match_idx

def reconcile_prediction(predicted_idx, predicted_text, options):
    """Reconcile index and text predictions."""
    # If we have a valid index, use it
    if predicted_idx is not None and 0 <= predicted_idx < len(options):
        expected_text = options[predicted_idx]
        if predicted_text and normalize_text(predicted_text) == normalize_text(expected_text):
            return predicted_idx, "Index + text match"
        else:
            return predicted_idx, "Index valid (text ignored)"
    
    # Try fuzzy matching with predicted text
    if predicted_text:
        fuzzy_idx = fuzzy_match_answer(predicted_text, options)
        if fuzzy_idx is not None:
            return fuzzy_idx, "Fuzzy text match"
    
    return None, "No valid prediction"

def extract_reasoning_steps(response):
    """Extract reasoning steps from CoT response."""
    if not response:
        return [], ""
    
    # Look for reasoning section
    reasoning_match = re.search(r'Reasoning_En:\s*(.*?)(?:Final Answer:|$)', response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        
        # Extract individual steps
        steps = []
        for match in re.finditer(r'Step \d+:\s*(.*?)(?=Step \d+:|$)', reasoning_text, re.DOTALL):
            step = match.group(1).strip()
            if step:
                steps.append(step)
        
        return steps, reasoning_text
    
    return [], ""

def process_mcq_evaluation(mcq_data, gemini_client, prompt_mode="zero_shot"):
    """Process Semantic Ambiguity evaluation with specified prompt mode."""
    
    out_file = OUTPUT_ROOT / f"riddles_semantic_ambiguity_gemini_flash_{prompt_mode}.json"
    
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
    
    # Filter to unprocessed items
    remaining_data = [item for item in mcq_data if item.get("id") not in processed_ids]
    
    if not remaining_data:
        print("✅ All examples already processed!")
        # Compute metrics from existing results
        preds = []
        for result in existing_results:
            if result.get("predicted_index") is not None and result.get("ground_truth_index") is not None:
                preds.append({
                    "predicted": result["predicted_index"],
                    "ground_truth": result["ground_truth_index"],
                    "has_reasoning": bool(result.get("reasoning_steps_en"))
                })
        
        valid_count = len(preds)
        correct = sum(1 for p in preds if p["predicted"] == p["ground_truth"])
        reasoning_count = sum(1 for p in preds if p["has_reasoning"])
        acc = (correct / valid_count * 100) if valid_count > 0 else 0
        
        metrics = {
            "Accuracy (%)": round(acc, 2),
            "n_examples_total": len(existing_results), 
            "n_valid_evaluated": valid_count, 
            "n_correct": correct,
            "n_with_reasoning": reasoning_count
        }
        metrics_out = OUTPUT_ROOT / f"riddles_semantic_ambiguity_metrics_gemini_flash_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics computed from existing results: Accuracy = {acc}%")
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
    
    # Process remaining examples
    for item in tqdm(remaining_data, desc=f"Semantic Ambiguity Evaluation ({prompt_mode})"):
        riddle_id = item.get("id")
        riddle_text = item.get("riddle", "")
        ans = item.get("ans", "")
        semantic_question = item.get("question", "")
        options = item.get("options", [])
        correct_answer = item.get("correct_option", "")
        
        if not riddle_text or not semantic_question or not options:
            print(f"⚠️ Skipping item {riddle_id}: missing required fields")
            continue
        
        # Find ground truth index
        ground_truth_index = None
        try:
            if correct_answer in options:
                ground_truth_index = options.index(correct_answer)
            else:
                if isinstance(correct_answer, int) and 0 <= correct_answer < len(options):
                    ground_truth_index = int(correct_answer)
        except Exception:
            ground_truth_index = None
        
        if ground_truth_index is None:
            print(f"⚠️ Could not find ground truth for item {riddle_id}")
            continue
        
        # Format options for prompt
        options_str = str(options)
        
        # Create prompt
        prompt = prompt_template.format(riddle=riddle_text, question=semantic_question, options=options_str)
        
        # Get model response
        response = gemini_client.ask_with_rotation(prompt)
        
        # Parse response
        predicted_idx, predicted_text, parse_status = extract_answer_from_response(response, options)
        
        # Extract reasoning for CoT mode
        reasoning_steps = []
        reasoning_text = ""
        if prompt_mode == "chain_of_thought":
            reasoning_steps, reasoning_text = extract_reasoning_steps(response)
        
        # Reconcile prediction
        final_idx, reconcile_status = reconcile_prediction(predicted_idx, predicted_text, options)
        
        # Create result
        result = {
            "riddle_id": riddle_id,
            "riddle": normalize_text(riddle_text),
            "riddle_ans": normalize_text(ans),
            "semantic_question": normalize_text(semantic_question),
            "ambiguous_word": item.get("ambiguous_word"),
            "options": [normalize_text(x) for x in options],
            "ground_truth_index": int(ground_truth_index) if ground_truth_index is not None else None,
            "ground_truth_answer": normalize_text(correct_answer) if correct_answer is not None else None,
            "predicted_index": int(final_idx) if final_idx is not None else None,
            "predicted_answer_text": normalize_text(predicted_text) if predicted_text else None,
            "reasoning_text": reasoning_text if reasoning_text else None,
            "reasoning_steps_en": reasoning_steps if reasoning_steps else None,
            "raw_response": response
        }
        
        results.append(result)
        
        # Save incrementally
        with open(out_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Rate limiting between requests
        time.sleep(5 + random.uniform(1, 3))
    
    # Compute final metrics
    preds = []
    for result in results:
        if result.get("predicted_index") is not None and result.get("ground_truth_index") is not None:
            preds.append({
                "predicted": result["predicted_index"],
                "ground_truth": result["ground_truth_index"],
                "has_reasoning": bool(result.get("reasoning_steps_en"))
            })
    
    valid_count = len(preds)
    correct = sum(1 for p in preds if p["predicted"] == p["ground_truth"])
    reasoning_count = sum(1 for p in preds if p["has_reasoning"])
    acc = (correct / valid_count * 100) if valid_count > 0 else 0

    metrics = {
        "Accuracy (%)": round(acc, 2),
        "n_examples_total": len(results), 
        "n_valid_evaluated": valid_count, 
        "n_correct": correct,
        "n_with_reasoning": reasoning_count
    }

    metrics_out = OUTPUT_ROOT / f"riddles_semantic_ambiguity_metrics_gemini_flash_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Finished Semantic Ambiguity evaluation ({prompt_mode})")
    print(f"Results: {out_file}, Metrics: {metrics_out}")
    print(f"Final Accuracy: {acc:.2f}% ({correct}/{valid_count})")

def main():
    """Main function to run Semantic Ambiguity evaluation."""
    print("🎯 Bengali Riddle Semantic Ambiguity Evaluation - Gemini 2.5 Flash Lite")
    print("=" * 60)
    
    # Load data
    print(f"📂 Loading Semantic Ambiguity data from: {DATASET_PATH}")
    mcq_data = load_mcq_data(DATASET_PATH)
    
    if not mcq_data:
        print("❌ No Semantic Ambiguity data found!")
        return
    
    print(f"📊 Total examples: {len(mcq_data)}")
    print(f"🎯 Processing full dataset: {len(mcq_data)} examples")
    
    # Initialize Gemini client
    gemini_client = RotatingGeminiClient(KEY_LIST, MODEL_NAME)
    
    # Run evaluations for all three modes
    modes = ["zero_shot", "few_shot", "chain_of_thought"]
    
    for mode in modes:
        print(f"\n🚀 Starting {mode} evaluation...")
        try:
            process_mcq_evaluation(mcq_data, gemini_client, mode)
        except Exception as e:
            print(f"❌ Error in {mode} evaluation: {e}")
            continue
    
    print("\n✅ All evaluations completed!")

    # Print usage statistics
    print("\n📊 API Key Usage Statistics:")
    for i, key in enumerate(gemini_client.keys):
        usage = gemini_client.key_usage_count[key]
        status = "❌ FAILED" if key in gemini_client.failed_keys else "✅ OK"
        print(f"Key {i}: {usage} requests - {status}")

if __name__ == "__main__":
    main()