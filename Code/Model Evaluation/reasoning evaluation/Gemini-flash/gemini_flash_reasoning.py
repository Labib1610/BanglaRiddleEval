#!/usr/bin/env python3
"""
gemini_flash_reasoning.py

Reasoning evaluation pipeline for Bengali riddles using Google Gemini-2.5-Flash model.
- Evaluates detailed Bengali reasoning generation for riddle solutions
- 4-step analysis in Bengali paragraph format
- **Dual Metrics**: LLM Judge + LLM-as-a-Judge for reasoning quality
- Uses Google Gemini API with rotating keys for reasoning quality assessment
- Semantic similarity via multilingual LLM Judge for reasoning comparison
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

# LLM-as-a-Judge only evaluation (LLM Judge removed)

# Google Gemini API imports
try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai package not found. Install it with:")
    print("pip install google-generativeai")
    exit(1)

# ==================== CONFIG ====================
# Google Gemini API Configuration
MODEL_NAME = "gemini-2.5-flash"

# Google Gemini API Configuration for LLM-as-a-Judge
# Add your API keys to this list
KEY_LIST = [
    # "your-api-key-1",
    # "your-api-key-2",
    # Add more keys as needed
]

# Split API keys - First half for judging, second half for evaluation
TOTAL_KEYS = len(KEY_LIST)
MID_POINT = TOTAL_KEYS // 2
JUDGE_KEYS = KEY_LIST[:MID_POINT]
EVALUATION_KEYS = KEY_LIST[MID_POINT:]

JUDGE_MODEL_NAME = "gemini-2.5-flash"
EVALUATION_MODEL_NAME = "gemini-2.5-flash"
NETWORK_TIMEOUT_SECONDS = 120

# Dataset and Output Configuration
RIDDLES_DATASET_PATH = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/main_dataset/riddles_reasoning.json")
OUTPUT_ROOT = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/Code/Model Evaluation (Benchmark)/reasoning evaluation/Gemini-flash")
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
    "- Give a score between 0 and 1\n"
    "- 1.0: Excellent reasoning (logical, accurate, complete, culturally aware)\n"
    "- 0.7-0.9: Good reasoning (mostly correct with minor issues)\n"
    "- 0.4-0.6: Average reasoning (some correct elements but lacks depth or accuracy)\n"
    "- 0.1-0.3: Poor reasoning (limited understanding, mostly incorrect)\n"
    "- 0.0: No meaningful reasoning or completely wrong\n\n"
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

# ==================== GEMINI EVALUATION CLIENT ====================
class RotatingGeminiEvaluationClient:
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
        print(f"➡️ Using Evaluation API key index {self.key_index}")
        
        try:
            socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"⚠️ Could not configure socket timeout: {e}")

    def _advance_key(self):
        old = self.key_index
        self.key_index = (self.key_index + 1) % len(self.keys)
        print(f"🔁 Switching Evaluation API key: {old} -> {self.key_index}")
        self._configure_current_key()

    def generate(self, prompt: str, max_tokens: int = None):
        """Generate text using Gemini."""
        max_attempts = 3
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
                        temperature=0.3,  # Slightly higher temperature for creative reasoning
                        top_k=40,
                        top_p=0.95,
                        max_output_tokens=max_tokens or 512,
                    )
                )
                
                if hasattr(resp, "text") and resp.text:
                    response_text = resp.text.strip()
                elif hasattr(resp, "candidates") and resp.candidates:
                    response_text = resp.candidates[0].content.parts[0].text.strip()
                else:
                    response_text = ""
                
                return resp, response_text
                
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "rate limit" in msg:
                    print(f"❗ Evaluation quota/rate-limit on key {self.key_index}: {e}")
                    self.failed_keys.add(self.keys[self.key_index])
                    self._advance_key()
                    sleep_time = 10 + random.uniform(5, 15)
                    print(f"⏳ Evaluation rate limit - backing off for {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    continue
                elif any(keyword in msg for keyword in ['timeout', 'connection', 'ssl', 'socket', 'network']):
                    print(f"❗ Evaluation network error on key {self.key_index} (attempt {attempt}/{max_attempts}): {e}")
                    self._advance_key()
                    sleep_time = 5 * (2 ** (attempt - 1))
                    print(f"⏳ Evaluation network issue - backing off for {sleep_time:.1f}s")
                    time.sleep(min(sleep_time, 120))
                    continue
                
                print(f"❗ Evaluation API call failed on key {self.key_index} (attempt {attempt}/{max_attempts}): {e}")
                self._advance_key()
                sleep_time = 2 * (2 ** (attempt - 1))
                time.sleep(min(sleep_time, 60))
                continue

        print(f"❌ Evaluation failed after {max_attempts} attempts. Last error: {last_exc}")
        return None, ""

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

def is_bengali_text(text):
    """Check if text contains primarily Bengali characters."""
    if not text:
        return False
    bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
    total_chars = sum(1 for char in text if char.isalpha())
    return total_chars > 0 and (bengali_chars / total_chars) > 0.5

    # Filter out empty predictions and references
    filtered_predicted = []
    filtered_ground_truth = []
    empty_indices = []
    
    for i, (pred, ref) in enumerate(zip(predicted_list, ground_truth_list)):
        if pred.strip() and ref.strip():
            filtered_predicted.append(pred)
            filtered_ground_truth.append(ref)
        else:
            empty_indices.append(i)
    
    if not filtered_predicted:
        print("⚠️ All predictions or references are empty, returning default scores")
        return [0.0] * len(predicted_list), [0.0] * len(predicted_list), [0.0] * len(predicted_list)
    
    try:
        # Use multilingual BERT model for Bengali support with explicit language setting
        P, R, F1 = llm_judge(filtered_predicted, filtered_ground_truth, 
                            model_type="bert-base-multilingual-cased",
                            lang="bn",  # Specify Bengali language
                            verbose=False)
        
        # Create full results list with 0.0 for empty entries
        full_P = [0.0] * len(predicted_list)
        full_R = [0.0] * len(predicted_list)
        full_F1 = [0.0] * len(predicted_list)
        
        # Fill in computed scores for non-empty entries
        filtered_idx = 0
        for i in range(len(predicted_list)):
            if i not in empty_indices:
                full_P[i] = P[filtered_idx].item()
                full_R[i] = R[filtered_idx].item()
                full_F1[i] = F1[filtered_idx].item()
                filtered_idx += 1
        
        return full_P, full_R, full_F1
        
    except Exception as e:
        print(f"⚠️ LLM Judge computation failed: {e}")
        print("Falling back to multilingual model without language specification...")
        try:
            # Fallback without lang parameter
            P, R, F1 = llm_judge(filtered_predicted, filtered_ground_truth, 
                                model_type="bert-base-multilingual-cased", 
                                verbose=False)
            
            # Create full results list with 0.0 for empty entries
            full_P = [0.0] * len(predicted_list)
            full_R = [0.0] * len(predicted_list)
            full_F1 = [0.0] * len(predicted_list)
            
            # Fill in computed scores for non-empty entries
            filtered_idx = 0
            for i in range(len(predicted_list)):
                if i not in empty_indices:
                    full_P[i] = P[filtered_idx].item()
                    full_R[i] = R[filtered_idx].item()
                    full_F1[i] = F1[filtered_idx].item()
                    filtered_idx += 1
            
            return full_P, full_R, full_F1
            
        except Exception as e2:
            print(f"⚠️ LLM Judge fallback also failed: {e2}")
            return [0.0] * len(predicted_list), [0.0] * len(predicted_list), [0.0] * len(predicted_list)

# ==================== CORE PROCESSING ====================
def process_generative_evaluation(gemini_llm: RotatingGeminiEvaluationClient, gemini_judge: RotatingGeminiClient, prompt_mode="zero_shot"):
    """Process generative evaluation with specified prompt mode."""
    
    out_file = OUTPUT_ROOT / f"riddles_reasoning_gemini_flash_{prompt_mode}.json"
    
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
    
    print(f"🎯 Processing {len(riddles_data)} examples from dataset")
    
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
            "LLM Judge Reasoning Quality (%)": round(avg_llm_judge_score * 100, 2),
            "n_examples_total": total,
            "avg_judge_score": round(avg_llm_judge_score, 3)
    }
        
        metrics_out = OUTPUT_ROOT / f"riddles_reasoning_metrics_gemini_flash_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
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
        raw_resp, response = gemini_llm.generate(prompt)
        
        # Extract generated reasoning (the entire response is the reasoning)
        generated_reasoning = extract_reasoning_from_response(response)
        print(f"📝 Generated reasoning length: {len(generated_reasoning)} characters")
        
        # Get LLM judge evaluation for reasoning quality
        if generated_reasoning.strip():
            llm_judge_score = gemini_judge.judge_answer(riddle_text, ground_truth_reasoning, generated_reasoning)
        else:
            llm_judge_score = 0.0
        
        # Create result (LLM Judge will be computed in batch later)
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

    # Compute final metrics
    llm_judge_scores = [r.get("llm_judge_score", 0.0) for r in results if "llm_judge_score" in r]
    total = len(results)
    avg_llm_judge_score = (sum(llm_judge_scores) / len(llm_judge_scores)) if llm_judge_scores else 0.0
    metrics = {
        "LLM Judge Average Score": round(avg_llm_judge_score, 3),
        "LLM Judge Reasoning Quality (%)": round(avg_llm_judge_score * 100, 2),
        "n_examples_total": total,
        "avg_judge_score": round(avg_llm_judge_score, 3)
    }

    metrics_out = OUTPUT_ROOT / f"riddles_reasoning_metrics_gemini_flash_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Finished Reasoning evaluation ({prompt_mode})")
    print(f"Results: {out_file}, Metrics: {metrics_out}")
    print(f"LLM Judge Reasoning Quality: {avg_llm_judge_score:.3f}/1.0 ({avg_llm_judge_score * 100:.1f}%)")
    
    print(f"Total examples processed: {total}")

def main():
    """Main function to run reasoning evaluation."""
    print("🎯 Bengali Riddle Reasoning Evaluation - Gemini-2.5-Flash")
    print("=" * 60)
    print(f"🔑 Using {len(JUDGE_KEYS)} keys for judging, {len(EVALUATION_KEYS)} keys for evaluation")
    print(f"🤖 Both evaluation and judging using {MODEL_NAME}")
    print("=" * 60)
    
    # Initialize Gemini evaluation client with second half of keys
    gemini_llm = RotatingGeminiEvaluationClient(EVALUATION_KEYS, EVALUATION_MODEL_NAME)
    
    # Initialize Gemini judge client with first half of keys
    gemini_judge = RotatingGeminiClient(JUDGE_KEYS, JUDGE_MODEL_NAME)
    
    # Run evaluations for all three modes
    modes = ["chain_of_thought"]
    
    for mode in modes:
        print(f"\n🚀 Starting {mode} evaluation...")
        try:
            process_generative_evaluation(gemini_llm, gemini_judge, mode)
        except Exception as e:
            print(f"❌ Error in {mode} evaluation: {e}")
            continue
    
    print("\n✅ All evaluations completed!")
    
    # Print usage statistics for both evaluation and judge
    print("\n📊 Gemini Evaluation API Key Usage Statistics:")
    for i, key in enumerate(gemini_llm.keys):
        usage = gemini_llm.key_usage_count[key]
        status = "❌ FAILED" if key in gemini_llm.failed_keys else "✅ OK"
        print(f"Evaluation Key {i}: {usage} requests - {status}")
    
    print("\n📊 Gemini Judge API Key Usage Statistics:")
    for i, key in enumerate(gemini_judge.keys):
        usage = gemini_judge.key_usage_count[key]
        status = "❌ FAILED" if key in gemini_judge.failed_keys else "✅ OK"
        print(f"Judge Key {i}: {usage} requests - {status}")

if __name__ == "__main__":
    main()