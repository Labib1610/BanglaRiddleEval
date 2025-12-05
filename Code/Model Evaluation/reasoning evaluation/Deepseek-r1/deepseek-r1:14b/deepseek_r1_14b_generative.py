#!/usr/bin/env python3
"""
deepseek_r1_14b_generative.py

Reasoning evaluation pipeline for Bengali riddles using Ollama deepseek-r1:14b model.
- Evaluates detailed Bengali reasoning generation for riddle solutions
- Uses a single Bengali reasoning prompt for all modes
- Computes BERTScore and LLM-as-a-judge metrics for reasoning quality
- Uses Google Gemini API for LLM judging with key rotation
- Semantic similarity via multilingual BERTScore for reasoning comparison
- Removed Levenshtein Distance as it's not suitable for reasoning evaluation
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

# Additional imports for new metrics
# try:
#     from Levenshtein import distance as levenshtein_distance
# except ImportError:
#     print("⚠️ python-Levenshtein package not found. Install it with:")
#     print("pip install python-Levenshtein")
#     def levenshtein_distance(a, b):
#         """Fallback implementation if library not available"""
#         if len(a) < len(b):
#             return levenshtein_distance(b, a)
#         if len(b) == 0:
#             return len(a)
#         previous_row = list(range(len(b) + 1))
#         for i, c1 in enumerate(a):
#             current_row = [i + 1]
#             for j, c2 in enumerate(b):
#                 insertions = previous_row[j + 1] + 1
#                 deletions = current_row[j] + 1
#                 substitutions = previous_row[j] + (c1 != c2)
#                 current_row.append(min(insertions, deletions, substitutions))
#             previous_row = current_row
#         return previous_row[-1]

# BERTScore removed - not needed for this evaluation

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
MODEL_NAME = "deepseek-r1:14b"
LLM_NUM_CTX = 4096
LLM_SEED = 0

# Google Gemini API Configuration for LLM-as-a-Judge
# Add your API keys to this list
KEY_LIST = [
    # "your-api-key-1",
    # "your-api-key-2",
    # Add more keys as needed
]

JUDGE_MODEL_NAME = "gemini-2.5-flash"
NETWORK_TIMEOUT_SECONDS = 120

# Dataset and Output Configuration
RIDDLES_DATASET_PATH = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/main_dataset/riddles_reasoning.json")
OUTPUT_ROOT = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/Code/Model Evaluation (Benchmark)/reasoning evaluation/Deepseek-r1/deepseek-r1:14b")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Processing entire dataset - no sampling

# ==================== PROMPT TEMPLATES ====================
# Bengali Reasoning Prompt Template (same for all modes)
BENGALI_REASONING_PROMPT = (
    "আপনি একটি বাংলা ধাঁধার জন্য বিস্তারিত যুক্তি তৈরি করবেন।\n\n"
    "ধাঁধা: \"{riddle}\"\n"
    "উত্তর: \"{answer}\"\n\n"
    "নিচের ৪টি ধাপে বিশ্লেষণ করুন এবং একটি সুন্দর বাংলা অনুচ্ছেদ আকারে লিখুন:\n\n"
    "1. উত্তর চিহ্নিতকরণ: ধাঁধার নির্দিষ্ট শব্দগুলো উদ্ধৃত করুন\n"
    "2. রূপকের ব্যাখ্যা: রূপকটি কী প্রতিনিধিত্ব করে তা ব্যাখ্যা করুন\n"
    "3. উত্তরের সাথে সংযোগ: \"{answer}\" এর কোন বৈশিষ্ট্য এই ধাঁধার সাথে মিলে যায় তা ব্যাখ্যা করুন\n"
    "4. সিদ্ধান্ত: কেন এটাই একমাত্র যুক্তিসংগত উত্তর তা সংক্ষেপে বলুন\n\n"
    "উদাহরণ ফরম্যাট:\n"
    "১. 'এক থালা': এখানে আকাশকে একটি বিশাল থালার সাথে তুলনা করা হয়েছে।\n"
    "২. 'সুপারি': সুপারি যেমন ছোট ছোট গোল হয়, আকাশের নক্ষত্রগুলোকেও দেখতে ছোট বিন্দুর মতো লাগে।\n"
    "৩. 'গুনতে নারি': সুপারি গোনা সম্ভব হলেও, আকাশের তারা বা নক্ষত্র অসংখ্য, যা গুনে শেষ করা যায় না।\n"
    "সিদ্ধান্ত: আকাশের বিশাল থালায় ছড়িয়ে থাকা অগণিত নক্ষত্রই হলো এই ধাঁধার উত্তর।\n\n"
    "অনুগ্রহ করে শুধুমাত্র reasoning টেক্সট দিন। JSON বা অন্য কোনো ফরম্যাট ব্যবহার করবেন না।"
)

# LLM-as-a-Judge prompt template for reasoning evaluation
LLM_JUDGE_PROMPT = (
    "You are an expert evaluator for Bengali riddle reasoning explanations. Your task is to score the quality of a reasoning explanation for a given riddle.\n\n"
    "Riddle: {riddle}\n"
    "Correct Answer: {ground_truth}\n"
    "Generated Reasoning: {predicted}\n\n"
    "Evaluation Criteria:\n"
    "1. Logical Structure: Does the reasoning follow a clear logical progression?\n"
    "2. Accuracy: Does the reasoning correctly identify the metaphors and connections?\n"
    "3. Completeness: Does it cover the key elements of the riddle?\n"
    "4. Cultural Context: Does it demonstrate understanding of Bengali cultural context?\n"
    "5. Language Quality: Is the Bengali language clear and well-structured?\n"
    "6. Conclusion: Does it reach the correct answer through valid reasoning?\n\n"
    "Scoring Instructions:\n"
    "- Give a score between 0 and 10\n"
    "- 9-10: Excellent reasoning (logical, accurate, complete, culturally aware)\n"
    "- 7-8: Good reasoning (mostly correct with minor issues)\n"
    "- 4-6: Average reasoning (some correct elements but lacks depth or accuracy)\n"
    "- 1-3: Poor reasoning (limited understanding, mostly incorrect)\n"
    "- 0: No meaningful reasoning or completely wrong\n\n"
    "Respond with ONLY the numerical score (e.g., 8, 5, 10, 0)\n\n"
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
        
        # Look for number (0 to 10)
        number_match = re.search(r'\b([0-9](?:\.[0-9]+)?|10(?:\.0+)?)\b', text)
        if number_match:
            try:
                score = float(number_match.group(1))
                # Ensure score is between 0 and 10
                return max(0.0, min(10.0, score))
            except ValueError:
                pass
        
        # Look for percentage (0% to 100%)
        percent_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)%\b', text)
        if percent_match:
            try:
                percent = float(percent_match.group(1))
                # Convert percentage to 0-10 scale
                return max(0.0, min(10.0, percent / 10.0))
            except ValueError:
                pass
        
        # Look for fraction (e.g., 3/4, 1/2)
        fraction_match = re.search(r'\b([0-9]+)/([0-9]+)\b', text)
        if fraction_match:
            try:
                numerator = float(fraction_match.group(1))
                denominator = float(fraction_match.group(2))
                if denominator > 0:
                    return max(0.0, min(10.0, (numerator / denominator) * 10.0))
            except ValueError:
                pass
        
        # Fallback: try to extract any number and normalize
        fallback_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', text)
        if fallback_match:
            try:
                num = float(fallback_match.group(1))
                # If number is > 10, assume it's a percentage
                if num > 10:
                    return max(0.0, min(10.0, num / 10.0))
                else:
                    return max(0.0, min(10.0, num))
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

# Removed sampling functions - processing entire dataset

def normalize_text(text):
    """Normalize Bengali text for comparison."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text.strip())

def extract_reasoning_from_response(response):
    """Extract reasoning text from model response."""
    if not response:
        return ""
    
    # Handle different response types
    if isinstance(response, dict) and 'response' in response:
        response_text = response['response']
    elif isinstance(response, str):
        response_text = response
    else:
        response_text = str(response)
    
    return response_text.strip() if response_text else ""

# Removed unused CoT extraction function - not needed for reasoning evaluation

def is_bengali_text(text):
    """Check if text contains primarily Bengali characters."""
    if not text:
        return False
    bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
    total_chars = sum(1 for char in text if char.isalpha())
    return total_chars > 0 and (bengali_chars / total_chars) > 0.5

# Levenshtein functions removed as they're not needed for reasoning evaluation

# BERTScore computation removed

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
        
        # Extract response text from Ollama GenerateResponse object
        response_text = ""
        if hasattr(resp, 'response'):
            # Ollama GenerateResponse object
            response_text = resp.response
        elif isinstance(resp, dict) and 'response' in resp:
            # Dictionary format
            response_text = resp['response']
        else:
            # Fallback
            response_text = str(resp)
        
        # Debug: Check what we're returning
        print(f"🔍 Raw response type: {type(resp)}")
        print(f"🔍 Response text type: {type(response_text)}")
        if isinstance(response_text, str) and len(response_text) > 50:
            print(f"🔍 Response text preview: {response_text[:50]}...")
        
        return resp, response_text

# ==================== CORE PROCESSING ====================
def process_generative_evaluation(ollama_llm: OllamaLLM, gemini_judge: RotatingGeminiClient, prompt_mode="zero_shot"):
    """Process generative evaluation with specified prompt mode."""
    
    suffix = "_full_dataset"
    out_file = OUTPUT_ROOT / f"riddles_reasoning_deepseek_r1_14b_{prompt_mode}{suffix}.json"
    
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
    
    print(f"🎯 Processing entire dataset: {len(riddles_data)} examples")
    
    # Filter to unprocessed items (use 'riddle_id' field for reasoning dataset)
    remaining_data = [item for item in riddles_data if item.get("riddle_id") not in processed_ids]
    
    if not remaining_data:
        print("✅ All examples already processed!")
        # Compute metrics from existing results
        llm_judge_scores = [r.get("llm_judge_score", 0.0) for r in existing_results if "llm_judge_score" in r]
        
        total = len(existing_results)
        avg_llm_judge_score = (sum(llm_judge_scores) / len(llm_judge_scores)) if llm_judge_scores else 0.0
        
        metrics = {
            "LLM Judge Average Score": round(avg_llm_judge_score, 3),
            "LLM Judge Reasoning Quality (%)": round(avg_llm_judge_score * 10, 2),
            "n_examples_total": total,
            "avg_judge_score": round(avg_llm_judge_score, 3)
        }
        
        metrics_out = OUTPUT_ROOT / f"riddles_reasoning_metrics_deepseek_r1_14b_{prompt_mode}{suffix}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics computed from existing results: LLM Judge Reasoning Quality = {avg_llm_judge_score:.3f}/10.0")
        return
    
    print(f"🔄 Processing {len(remaining_data)} remaining examples...")
    
    # All modes use the same Bengali reasoning prompt
    print(f"📝 Using Bengali reasoning prompt for {prompt_mode} mode")
    
    results = existing_results.copy()
    
    # Process remaining examples
    for item in tqdm(remaining_data, desc=f"Reasoning Evaluation ({prompt_mode})"):
        riddle_id = item.get("riddle_id")
        riddle_text = item.get("riddle", "")
        ground_truth_answer = item.get("ans", "")
        ground_truth_reasoning = item.get("reasoning", "")
        
        if not riddle_text or not ground_truth_answer:
            print(f"⚠️ Skipping item {riddle_id}: missing riddle or answer")
            continue
        
        # Create Bengali reasoning prompt (same for all modes)
        prompt = BENGALI_REASONING_PROMPT.format(riddle=riddle_text, answer=ground_truth_answer)
        
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
        
        # Debug: Print response type and preview
        print(f"🔍 Response type: {type(response)}")
        if isinstance(response, str) and len(response) > 100:
            print(f"🔍 Response preview: {response[:100]}...")
        elif isinstance(response, dict):
            print(f"🔍 Response keys: {response.keys() if response else 'None'}")
            if 'response' in response:
                print(f"🔍 Actual response: '{response['response']}'")
        
        # Extract generated reasoning (the entire response is the reasoning)
        generated_reasoning = extract_reasoning_from_response(response)
        print(f"📝 Generated reasoning length: {len(generated_reasoning)} characters")
        
        # Get LLM judge evaluation for reasoning quality
        if generated_reasoning.strip():
            llm_judge_score = gemini_judge.judge_answer(riddle_text, ground_truth_reasoning, generated_reasoning)
        else:
            llm_judge_score = 0.0
        
        # Create result
        result = {
            "riddle_id": riddle_id,
            "riddle": riddle_text,
            "ground_truth_answer": ground_truth_answer,
            "ground_truth_reasoning": ground_truth_reasoning,
            "generated_reasoning": generated_reasoning,
            "llm_judge_score": llm_judge_score
        }
        
        results.append(result)
        
        # Save incrementally
        with open(out_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Rate limiting
        time.sleep(5 + random.uniform(3, 5))
    
    # Save final results
    if remaining_data:
        with open(out_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Compute final metrics
    llm_judge_scores = [r.get("llm_judge_score", 0.0) for r in results if "llm_judge_score" in r]
    
    total = len(results)
    avg_llm_judge_score = (sum(llm_judge_scores) / len(llm_judge_scores)) if llm_judge_scores else 0.0

    metrics = {
        "LLM Judge Average Score": round(avg_llm_judge_score, 3),
        "LLM Judge Reasoning Quality (%)": round(avg_llm_judge_score * 10, 2),
        "n_examples_total": total,
        "avg_judge_score": round(avg_llm_judge_score, 3)
    }

    metrics_out = OUTPUT_ROOT / f"riddles_reasoning_metrics_deepseek_r1_14b_{prompt_mode}{suffix}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Finished Reasoning evaluation ({prompt_mode})")
    print(f"Results: {out_file}, Metrics: {metrics_out}")
    print(f"LLM Judge Reasoning Quality: {avg_llm_judge_score:.3f}/10.0 ({avg_llm_judge_score * 10:.1f}%)")
    print(f"Total examples processed: {total}")

def main():
    """Main function to run reasoning evaluation."""
    print("🎯 Bengali Riddle Reasoning Evaluation - DeepSeek-R1:7b")
    print("=" * 60)
    
    # Initialize Ollama client
    ollama_llm = OllamaLLM(host=LLM_URL, model=MODEL_NAME, num_ctx=LLM_NUM_CTX, seed=LLM_SEED)
    
    # Initialize Gemini judge client
    gemini_judge = RotatingGeminiClient(KEY_LIST, JUDGE_MODEL_NAME)
    
    # Run evaluations for all three modes
    modes = ["chain_of_thought"]
    
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