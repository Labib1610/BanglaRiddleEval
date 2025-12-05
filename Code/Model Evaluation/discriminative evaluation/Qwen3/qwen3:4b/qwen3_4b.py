#!/usr/bin/env python3
"""
qwen3_4b.py

MCQ evaluation pipeline for Bengali riddles using Ollama qwen3:4b model.
- Evaluates multiple choice questions with semantic ambiguity
- Extracts reported index and answer text from model responses
- Reconciles index vs text conflicts with exact and fuzzy matching
- Saves JSON outputs and accuracy metrics
- Includes zero-shot, few-shot, and chain-of-thought (CoT) prompting modes
"""

import os
import json
import time
import re
import unicodedata
import difflib
from pathlib import Path
from tqdm import tqdm

# Ollama client import
try:
    from ollama import Client
except Exception as e:
    raise RuntimeError("ollama client not available. Install the ollama client package.") from e

# ---------------- CONFIG ----------------
LLM_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:4b"
LLM_NUM_CTX = 4096
LLM_SEED = 0

N_EXAMPLES = None

OUTPUT_ROOT = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/Code/Model Evaluation (Benchmark)/discriminative evaluation/Qwen3/qwen3:4b")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MCQ_DATASET_PATH = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/dataset/riddles_mcq.json")

# ---------------- PROMPTS ----------------
PROMPT_ZERO_SHOT = (
    "You are an AI assistant that answers Bengali riddle multiple-choice questions.\n"
    "Task:\n"
    "1. Read the riddle in Bengali: {riddle}\n"
    "2. Review the provided answer choices: {options}\n"
    "3. Select the **single most accurate answer** and respond in Bengali.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Bengali text for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n\n"
    "Index: <option_index>, Answer: \"<option_text_in_Bengali>\""
)

PROMPT_FEW_SHOT = (
    "You are an AI assistant that answers Bengali riddle multiple-choice questions.\n"
    "Task:\n"
    "1. Read the riddle in Bengali: {riddle}\n"
    "2. Review the provided answer choices: {options}\n"
    "3. Select the **single most accurate answer** and respond in Bengali.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Bengali text for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n"
    "Index: <option_index>, Answer: \"<option_text_in_Bengali>\"\n\n"
    "Examples:\n"
    "Riddle: \"বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।\"\n"
    "Options: [\"আনারস\", \"কলা\", \"আম\", \"পেঁপে\"]\n"
    "Answer: Index: 0, Answer: \"আনারস\"\n\n"
    "Riddle: \"কোন জিনিস কাটলে বাড়ে?\"\n"
    "Options: [\"চুল\", \"পুকুর\", \"নখ\", \"গাছ\"]\n"
    "Answer: Index: 1, Answer: \"পুকুর\"\n\n"
    "Riddle: \"হাত আছে, পা নেই, বুক তার ফাটা, মানুষকে গিলে খায়, নাই তার মাথা।\"\n"
    "Options: [\"জ্যাকেট\", \"শার্ট\", \"বই\", \"ব্যাগ\"]\n"
    "Answer: Index: 1, Answer: \"শার্ট\"\n\n"
    "Now answer for the given riddle.\n"
)

PROMPT_CHAIN_OF_THOUGHTS = (
    "You are an AI assistant that answers Bengali riddle multiple-choice questions using step-by-step reasoning.\n\n"
    "Task:\n"
    "1. Read the riddle in Bengali: {riddle}\n"
    "2. Review the provided answer choices: {options}\n"
    "3. Select the **single most accurate answer** using chain-of-thought reasoning.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Think step by step about the riddle's metaphors and meaning.\n"
    "- Write reasoning steps in English but think with Bengali cultural knowledge.\n"
    "- Final answer must be in Bengali.\n"
    "- In Reasoning_En, write step-by-step reasoning in English — break down the solution logically:\n"
    "  Step 1: Analyze the riddle's key words and metaphors.\n"
    "  Step 2: Connect observations with relevant answer options.\n"
    "  Step 3: Eliminate wrong options with brief reasoning.\n"
    "  Step 4: Explain why the final choice is correct.\n"
    "- Be clear, concise, and factual (avoid overly lengthy explanations).\n"
    "- Follow this exact answer format:\n\n"
    "Reasoning_En:\n"
    "Step 1: <your_riddle_analysis>\n"
    "Step 2: <your_matching_logic>\n"
    "Step 3: <your_elimination_reasoning>\n"
    "Step 4: <your_final_choice_explanation>\n\n"
    "Final Answer: Index: <option_index>, Answer: \"<option_text_in_Bengali>\""
)

# ---------------- Helpers ----------------
def load_mcq_data(path: Path):
    """Load MCQ data from JSON file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def normalize_text(text):
    """Unicode NFKC, collapse whitespace, strip"""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def safe_parse_response(resp):
    """
    Extract a textual preview from various possible ollama response shapes.
    """
    if resp is None:
        return ""
    if isinstance(resp, dict):
        for key in ("response", "output", "text", "result"):
            if key in resp:
                val = resp[key]
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, dict):
                    for k2 in ("content", "message", "text"):
                        if k2 in val and isinstance(val[k2], str):
                            return val[k2].strip()
        # fallback: join string values or stringify
        string_parts = [v for v in resp.values() if isinstance(v, str)]
        if string_parts:
            return "\n".join(string_parts).strip()
        return str(resp)
    elif isinstance(resp, str):
        return resp.strip()
    else:
        return str(resp)

# ---------------- Extraction & Matching Helpers ----------------
def extract_index_from_answer(answer_text):
    """Return integer index if found (Index: N or leading 'N'), else None."""
    if not isinstance(answer_text, str):
        return None
    t = normalize_text(answer_text)
    match = re.search(r'Index\s*:\s*(\d+)', t, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match2 = re.search(r'^\s*(\d+)[\.\)\s-]*', t)
    if match2:
        return int(match2.group(1))
    return None

def extract_answer_text_from_response(answer_text):
    """
    Extract the quoted Answer text like Answer: "আনারস" or the substring after 'Answer:'.
    Returns normalized extracted string (may be empty).
    """
    if not isinstance(answer_text, str):
        return ""
    t = answer_text
    # quoted variant
    m = re.search(r'Answer\s*:\s*[""'"'"'`]{1}\s*(.+?)\s*[""'"'"'`]{1}', t, flags=re.IGNORECASE)
    if m:
        return normalize_text(m.group(1))
    # after Answer:
    m2 = re.search(r'Answer\s*:\s*(.+)', t, flags=re.IGNORECASE)
    if m2:
        return normalize_text(m2.group(1))
    return normalize_text(t)

def extract_reasoning_and_final_answer(answer_text):
    """
    Extract reasoning steps and final answer from CoT response.
    Returns tuple: (reasoning_steps, reasoning_text, predicted_index, predicted_answer_text)
    """
    if not answer_text:
        return None, None, None, None

    text = str(answer_text).strip()

    # Helper: clean a step label like "Step 1:" => returns remainder
    def _clean_step(s):
        return re.sub(r'^\s*Step\s*\d+\s*[:\-]?\s*', '', s, flags=re.IGNORECASE).strip()

    # 1) Try to find Reasoning_En block then Final Answer (CoT)
    reasoning_block = None
    m_reason = re.search(r'Reasoning[_ ]?En\s*[:\-]?\s*(.*?)(?:Final\s*Answer|Index\s*:|\Z)', text, flags=re.IGNORECASE | re.DOTALL)
    if m_reason:
        reasoning_block = m_reason.group(1).strip()

    reasoning_steps = None
    if reasoning_block:
        # attempt to extract up to 8 Step N items in order
        steps = []
        for i in range(1, 9):
            m_step = re.search(r'(?:^|\n)\s*Step\s*' + str(i) + r'\s*[:\-]?\s*(.*?)(?=(?:\n\s*Step\s*' + str(i+1) + r'\b)|\Z)', reasoning_block, flags=re.IGNORECASE | re.DOTALL)
            if m_step:
                step_text = _clean_step(m_step.group(1))
                if step_text:
                    steps.append(step_text)
        if steps:
            reasoning_steps = steps
        else:
            lines = [ln.strip() for ln in reasoning_block.splitlines() if ln.strip()]
            if lines:
                reasoning_steps = lines

    # 2) Extract index via "Final Answer: Index: X" or "Final Answer - Index X"
    m_final_idx = re.search(r'Final\s*Answer\s*[:\-]?\s*(?:Index\s*[:\-]?\s*(\d+))', text, flags=re.IGNORECASE)
    if m_final_idx:
        idx = int(m_final_idx.group(1))
    else:
        # 3) Try "Index: X" anywhere else
        m_idx_any = re.search(r'\bIndex\s*[:\-]?\s*(\d+)\b', text, flags=re.IGNORECASE)
        idx = int(m_idx_any.group(1)) if m_idx_any else None

    # 4) Extract predicted answer text if present: look for Answer: "..." after Index or Final Answer
    predicted_answer_text = None
    m_ans = re.search(r'Answer\s*[:\-]?\s*[""'"'"']?([^"'"'"'\n]+)[""'"'"']?', text, flags=re.IGNORECASE)
    if m_ans:
        predicted_answer_text = m_ans.group(1).strip()

    # 5) Also attempt to extract reasoning_text more generically if not captured above
    reasoning_text = None
    if reasoning_block:
        reasoning_text = reasoning_block
    else:
        m_reason2 = re.search(r'Reasoning\s*[:\-]\s*(.*?)(?:Final\s*Answer|Index\s*:|\Z)', text, flags=re.IGNORECASE | re.DOTALL)
        reasoning_text = m_reason2.group(1).strip() if m_reason2 else None
        if reasoning_text:
            lines = [ln.strip() for ln in reasoning_text.splitlines() if ln.strip()]
            reasoning_steps = lines if lines else reasoning_steps

    return reasoning_steps, reasoning_text, idx, predicted_answer_text

def match_text_to_options(pred_text, options):
    """
    Try exact normalized match first, else try difflib close match.
    Returns (matched_index_or_None, matched_text_or_empty, method)
    """
    if not options:
        return None, "", "none"
    norm_opts = [normalize_text(o) for o in options]
    # direct exact (raw)
    for i, o in enumerate(options):
        if pred_text == o:
            return i, o, "exact"
    # normalized exact
    for i, no in enumerate(norm_opts):
        if pred_text == no:
            return i, options[i], "norm_exact"
    # difflib on normalized strings
    close = difflib.get_close_matches(pred_text, norm_opts, n=1, cutoff=0.6)
    if close:
        idx = norm_opts.index(close[0])
        return idx, options[idx], "difflib"
    return None, "", "none"

# ---------------- Ollama wrapper ----------------
class OllamaLLM:
    def __init__(self, host: str, model: str, num_ctx: int = 4096, seed: int = 0):
        self.host = host
        self.model = model
        self.num_ctx = num_ctx
        self.seed = seed
        self.client = Client(host=self.host)

    def generate(self, prompt: str, max_tokens: int = None):
        """
        Calls Ollama for text generation.
        Returns (raw_response_obj, cleaned_text_preview).
        """
        options = {
            "seed": self.seed,
            "num_ctx": self.num_ctx,
        }
        
        # Only add num_predict if max_tokens is specified
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        gen_args = {
            "model": self.model,
            "prompt": prompt,
            "options": options
        }

        resp = self.client.generate(**gen_args)  # may raise; caller handles retries
        raw_preview = safe_parse_response(resp)

        # regex: extract response='...' or output='...' if present
        m = re.search(r"(?:response|output)='(.*?)'", raw_preview)
        if m:
            extracted = m.group(1)
        else:
            m2 = re.search(r'(?:response|output)="(.*?)"', raw_preview)
            extracted = m2.group(1) if m2 else raw_preview

        cleaned = normalize_text(extracted)
        return resp, cleaned

# ---------------- Core processing ----------------
def compute_accuracy(preds, gts):
    correct = 0
    valid = 0
    for p, g in zip(preds, gts):
        if p is not None and g is not None:
            valid += 1
            if p == g:
                correct += 1
    return round(100.0 * correct / valid, 2) if valid > 0 else 0.0, valid, correct

def load_existing_results(out_file: Path):
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf8") as f:
                data = json.load(f)
            processed_ids = {item["riddle_id"] for item in data if "riddle_id" in item}
            return data, processed_ids
        except Exception as e:
            print(f"⚠️ Could not load existing results {out_file}: {e}")
            return [], set()
    return [], set()

def save_results_atomic(out_file: Path, results_list):
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    tmp.replace(out_file)

def process_mcq_evaluation(ollama_llm: OllamaLLM, prompt_mode="few", n_examples=None):
    print(f"\n==== Processing MCQ evaluation (prompt_mode={prompt_mode}) ====")

    mcq_data = load_mcq_data(MCQ_DATASET_PATH)
    
    if not mcq_data:
        print(f"⚠️ No MCQ data found in {MCQ_DATASET_PATH}")
        return
    
    # Apply n_examples limit if specified
    if n_examples is not None:
        mcq_data = mcq_data[:n_examples]
        print(f"📊 Using first {len(mcq_data)} items (n_examples={n_examples})")
    else:
        print(f"📊 Using all {len(mcq_data)} items")

    if not mcq_data:
        print("⚠️ No MCQ data available")
        return

    out_file = OUTPUT_ROOT / f"riddles_mcq_qwen3_4b_{prompt_mode}.json"
    results_list, processed_ids = load_existing_results(out_file)

    # Choose prompt template
    if prompt_mode == "zero":
        prompt_template = PROMPT_ZERO_SHOT
    elif prompt_mode == "cot":
        prompt_template = PROMPT_CHAIN_OF_THOUGHTS
    else:
        prompt_template = PROMPT_FEW_SHOT

    examples_to_process = [item for item in mcq_data if item.get("id") not in processed_ids]
    print(f"Total examples: {len(mcq_data)}; already done: {len(processed_ids)}; to process: {len(examples_to_process)}")

    if not examples_to_process:
        print("✅ Nothing to process; computing metrics from existing results.")
        preds = [r.get("predicted_index") for r in results_list]
        gts = [r.get("ground_truth_index") for r in results_list]
        acc, valid_count, correct = compute_accuracy(preds, gts)
        reasoning_count = sum(1 for r in results_list if r.get("reasoning_text"))
        metrics = {
            "Accuracy (%)": acc, 
            "n_examples_total": len(preds), 
            "n_valid_evaluated": valid_count, 
            "n_correct": correct,
            "n_with_reasoning": reasoning_count
        }
        metrics_out = OUTPUT_ROOT / f"riddles_mcq_metrics_qwen3_4b_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics computed from existing results: Accuracy = {acc}%")
        return

    pred_indices, gt_indices = [], []

    for item in tqdm(examples_to_process, desc=f"MCQ Evaluation"):
        riddle_id = item.get("id")
        riddle = item.get("question")  # Fixed: dataset uses 'question' not 'riddle'
        options = item.get("options", [])
        correct_answer = item.get("correct_answer")  # ground truth answer string

        # ground-truth index (programming index)
        gt_index = None
        try:
            if correct_answer in options:
                gt_index = options.index(correct_answer)
            else:
                if isinstance(correct_answer, int) and 0 <= correct_answer < len(options):
                    gt_index = int(correct_answer)
        except Exception:
            gt_index = None

        full_prompt = prompt_template.format(riddle=riddle, options=json.dumps(options, ensure_ascii=False))

        # call LLM with retries
        resp_preview = ""
        raw_resp = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                raw_resp, resp_preview = ollama_llm.generate(full_prompt)
                break
            except Exception as e:
                print(f"❗ Ollama generate failed (attempt {attempt}/{max_attempts}): {e}")
                time.sleep(0.5 * attempt)
                if attempt == max_attempts:
                    print("❌ Failed after retries, skipping this example.")
                    raw_resp, resp_preview = None, ""

        answer_text_raw = resp_preview or ""
        
        # Different extraction logic for CoT vs other modes
        if prompt_mode == "cot":
            reasoning_steps, reasoning_text, predicted_index, predicted_answer_text = extract_reasoning_and_final_answer(answer_text_raw)
            
            # If CoT extraction failed, fall back to standard extraction
            if predicted_index is None:
                predicted_index = extract_index_from_answer(answer_text_raw)
                predicted_answer_text = extract_answer_text_from_response(answer_text_raw)
                reasoning_steps = None
                reasoning_text = None
        else:
            predicted_index = extract_index_from_answer(answer_text_raw)
            predicted_answer_text = extract_answer_text_from_response(answer_text_raw)
            reasoning_steps = None
            reasoning_text = None

        # Text-to-option matching for cases where index is missing but text is present
        if predicted_index is None and predicted_answer_text:
            try:
                norm_pred = normalize_text(predicted_answer_text).lower()
                mapped = None
                for i, opt in enumerate(options):
                    if normalize_text(opt).lower() == norm_pred:
                        mapped = i
                        break
                predicted_index = mapped
            except Exception:
                predicted_index = None

        # Create result item
        result_item = {
            "riddle_id": riddle_id,
            "riddle": normalize_text(riddle),
            "options": [normalize_text(x) for x in options],
            "ground_truth_index": int(gt_index) if gt_index is not None else None,
            "ground_truth_answer": normalize_text(correct_answer) if correct_answer is not None else None,
            "predicted_index": int(predicted_index) if predicted_index is not None else None,
            "predicted_answer_text": normalize_text(predicted_answer_text) if predicted_answer_text else None,
            "reasoning_text": reasoning_text if reasoning_text else None,
            "reasoning_steps_en": reasoning_steps if reasoning_steps else None,
            "raw_response": answer_text_raw
        }

        results_list.append(result_item)
        save_results_atomic(out_file, results_list)

        pred_indices.append(predicted_index)
        gt_indices.append(gt_index)

        time.sleep(0.3)

    # Compute final metrics
    acc, valid_count, correct = compute_accuracy(pred_indices, gt_indices)
    reasoning_count = sum(1 for r in results_list if r.get("reasoning_text"))
    metrics = {
        "Accuracy (%)": acc, 
        "n_examples_total": len(pred_indices), 
        "n_valid_evaluated": valid_count, 
        "n_correct": correct,
        "n_with_reasoning": reasoning_count
    }

    metrics_out = OUTPUT_ROOT / f"riddles_mcq_metrics_qwen3_4b_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Finished MCQ evaluation ({prompt_mode})")
    print(f"Results: {out_file}, Metrics: {metrics_out}")
    print(f"Accuracy: {acc}% over {valid_count} valid examples (total: {len(pred_indices)})")
    if reasoning_count > 0:
        print(f"Examples with reasoning: {reasoning_count}")

def main():
    ollama_llm = OllamaLLM(host=LLM_URL, model=MODEL_NAME, num_ctx=LLM_NUM_CTX, seed=LLM_SEED)
    
    # Run three prompt modes: zero-shot, few-shot, and chain-of-thought
    for prompt_mode in ["zero", "few", "cot"]:
        print(f"\n########### Starting run for prompt_mode={prompt_mode} ###########")
        try:
            process_mcq_evaluation(ollama_llm, prompt_mode=prompt_mode, n_examples=N_EXAMPLES)
        except Exception as e:
            print(f"❗ Error processing MCQ evaluation ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()