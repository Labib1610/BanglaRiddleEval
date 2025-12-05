# Gemini 2.5 Flash Lite Generative Evaluation

This directory contains the evaluation pipeline for testing the Google Gemini 2.5 Flash Lite model on Bengali riddle generative answer generation using the Google Gemini API. Both the evaluation model and the LLM-as-a-Judge use Gemini 2.5 Flash Lite with API key rotation.

## Features

- **Three Prompting Modes**: Zero-shot, Few-shot, and Chain-of-Thought (CoT)
- **Dual Gemini Usage**: Uses Gemini 2.5 Flash Lite for both answer generation and LLM-as-a-Judge
- **Split API Key Management**: First half of keys for judging, second half for evaluation
- **Multiple Metrics**: Levenshtein Distance, BERTScore, and LLM Judge scoring (0-1 scale)
- **Random Sampling**: Uses 150 random samples from the full dataset for consistent evaluation
- **Progress Tracking**: Resume capability with incremental saves
- **Advanced Error Handling**: Retry mechanism with exponential backoff for network and quota errors

## Files

- `gemini_flash_lite.py` - Main generative evaluation script with dual Gemini clients
- `run_evaluation.py` - Simple runner script
- `README.md` - This documentation

## Dataset Format

The script expects generative riddle data in this format:
```json
[
  {
    "riddle_id": 1,
    "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
    "ans": "আনারস"
  }
]
```

## Output Format

### Results JSON
```json
{
  "riddle_id": 1,
  "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
  "ground_truth": "আনারস",
  "predicted_answer": "আনারস",
  "llm_judge_score": 1.0,
  "levenshtein_distance": 0,
  "levenshtein_similarity": 1.0,
  "bert_precision": 1.0,
  "bert_recall": 1.0,
  "bert_f1": 1.0,
  "reasoning_text": "Step-by-step reasoning in English...",
  "reasoning_steps_en": ["Step 1: ...", "Step 2: ..."]
}
```

### Metrics JSON
```json
{
  "LLM Judge Average Score": 0.756,
  "LLM Judge Accuracy (%)": 75.6,
  "Levenshtein Similarity Average": 0.678,
  "Levenshtein Similarity (%)": 67.8,
  "BERTScore F1 Average": 0.734,
  "BERTScore F1 (%)": 73.4,
  "n_examples_total": 150,
  "n_with_reasoning": 50
}
```

## Usage

### Prerequisites
```bash
# Install required packages
pip install google-generativeai tqdm

# API keys are already configured in the script
# 25 rotating API keys for handling rate limits
```

### Run Evaluation
```bash
# Navigate to the directory
cd "/path/to/Gemini-flash-lite"

# Run the evaluation
python run_evaluation.py

# Or run directly
python gemini_flash_lite.py
```

### Configuration

Edit the CONFIG section in `gemini_flash_lite.py`:

```python
MODEL_NAME = "gemini-2.5-flash-lite"         # Google Gemini model
JUDGE_MODEL_NAME = "gemini-2.5-flash-lite"   # Same model for judging
EVALUATION_MODEL_NAME = "gemini-2.5-flash-lite" # Same model for evaluation
NETWORK_TIMEOUT_SECONDS = 120                # API timeout
RANDOM_SAMPLE_SIZE = 150                     # Sample size for evaluation
RANDOM_SEED = 42                             # Seed for reproducibility
# First 12 keys used for judging, last 13 keys for evaluation
```

## Prompting Modes

### 1. Zero-shot
- Direct question without examples
- Simple format: "Answer: <bengali_text>"

### 2. Few-shot 
- Includes 3 example riddle-answer pairs
- Same format as zero-shot with examples

### 3. Chain-of-Thought (CoT)
- Step-by-step reasoning in English
- Final answer in Bengali
- Format: Reasoning_En: Steps + Final Answer: <bengali_text>

## Output Files

The script generates these files in the output directory:

- `riddles_generative_gemini_flash_lite_zero_shot_sample150.json` - Zero-shot results (150 samples)
- `riddles_generative_gemini_flash_lite_few_shot_sample150.json` - Few-shot results (150 samples)
- `riddles_generative_gemini_flash_lite_chain_of_thought_sample150.json` - CoT results (150 samples)
- `riddles_generative_metrics_gemini_flash_lite_zero_shot_sample150.json` - Zero-shot metrics
- `riddles_generative_metrics_gemini_flash_lite_few_shot_sample150.json` - Few-shot metrics
- `riddles_generative_metrics_gemini_flash_lite_chain_of_thought_sample150.json` - CoT metrics
- `sampled_ids.json` - List of sampled riddle IDs for consistency

## Error Handling

- **API Key Rotation**: Automatic rotation across 25 API keys when encountering rate limits
- **Rate Limit Management**: Smart backoff strategies for quota (50/min) and daily (1000/day) limits
- **Network Timeouts**: Enhanced timeout handling with 120-second limits and retry logic
- **Parse Errors**: Graceful fallback to text matching with fuzzy string matching
- **Resume**: Automatically skips processed examples for interrupted runs
- **Logging**: Detailed progress and error reporting with API key usage statistics

## Evaluation Metrics

The system calculates multiple metrics:

### 1. LLM-as-a-Judge Score (0-1)
- Uses Gemini 2.5 Flash Lite to evaluate answer quality
- Considers exact match, semantic equivalence, cultural context, synonyms, and partial correctness
- Provides nuanced scoring beyond simple exact matching

### 2. Levenshtein Similarity (0-1)
- Character-level edit distance between predicted and ground truth answers
- Normalized to 0-1 scale for consistent comparison

### 3. BERTScore (Precision, Recall, F1)
- Semantic similarity using multilingual BERT embeddings
- Captures meaning beyond character-level differences
- Falls back to Levenshtein if BERTScore unavailable

## API Key Management

The system includes a sophisticated dual API key rotation mechanism:

### Features:
- **Split Key Management**: 25 keys split into two pools
  - First 12 keys: LLM-as-a-Judge evaluation
  - Last 13 keys: Answer generation
- **Independent Rotation**: Each pool rotates independently for optimal load balancing
- **Smart Rate Limiting**: Handles both per-minute (50 requests) and daily (1000 requests) limits
- **Failure Detection**: Automatically marks and skips failed/exhausted keys in each pool
- **Usage Tracking**: Monitors request counts per key for both evaluation and judging
- **Backoff Strategies**: 
  - Rate limits: 10-25 second delays with randomization
  - Network errors: Progressive backoff up to 120 seconds
  - General errors: Exponential backoff up to 60 seconds

### Dual Client Architecture:
1. **RotatingGeminiEvaluationClient**: Handles answer generation with second half of keys
2. **RotatingGeminiClient**: Handles LLM judging with first half of keys
3. Both clients operate independently with separate error handling
4. Provides detailed statistics on key usage and failures for both pools

This architecture ensures robust, uninterrupted evaluation even with API limitations while maximizing throughput through intelligent load distribution.