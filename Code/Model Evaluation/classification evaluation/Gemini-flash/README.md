# Gemini 2.5 Flash Semantic Ambiguity Evaluation

This directory contains the evaluation pipeline for testing the Google Gemini 2.5 Flash model on Bengali riddle semantic ambiguity classification questions using the Google Gemini API.

## Features

- **Three Prompting Modes**: Zero-shot, Few-shot, and Chain-of-Thought (CoT)
- **API Key Rotation**: Supports multiple API keys for rate limit handling (add your own keys)
- **Robust Parsing**: Extracts both index and answer text from model responses
- **Full Dataset Processing**: Evaluates the entire dataset (no sampling)
- **Accuracy Calculation**: Direct matching of predicted vs ground truth indices
- **Progress Tracking**: Resume capability with incremental saves
- **Error Handling**: Advanced retry mechanism with exponential backoff for network and quota errors

## Files

- `gemini_flash.py` - Main evaluation script with API key rotation
- `run_evaluation.py` - Simple runner script
- `README.md` - This documentation

## Dataset Format

The script expects MCQ data in this format:
```json
[
  {
    "id": 1,
    "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
    "options": ["আনারস", "কলা", "আম", "পেঁপে"],
    "correct_answer": "আনারস"
  }
]
```

## Output Format

### Results JSON
```json
{
  "riddle_id": 1,
  "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
  "options": ["আনারস", "কলা", "আম", "পেঁপে"],
  "ground_truth_index": 0,
  "ground_truth_answer": "আনারস",
  "predicted_index": 0,
  "predicted_answer_text": "আনারস",
  "reasoning_text": "...",
  "reasoning_steps_en": ["Step 1: ...", "Step 2: ..."],
  "raw_response": "Index: 0, Answer: \"আনারস\""
}
```

### Metrics JSON
```json
{
  "Accuracy (%)": 75.5,
  "n_examples_total": 1244,
  "n_valid_evaluated": 1200,
  "n_correct": 906,
  "n_with_reasoning": 400
}
```

## Usage

### Prerequisites
```bash
# Install required packages
pip install google-generativeai tqdm

# Add your API keys to the KEY_LIST in gemini_flash.py:
# KEY_LIST = [
#     "your-api-key-1",
#     "your-api-key-2",
#     # Add more keys as needed
# ]
```

### Run Evaluation
```bash
# Navigate to the directory
cd "/path/to/Gemini-flash"

# Run the evaluation
python run_evaluation.py

# Or run directly
python gemini_flash.py
```

### Configuration

Edit the CONFIG section in `gemini_flash.py`:

```python
MODEL_NAME = "gemini-2.5-flash"      # Google Gemini model
NETWORK_TIMEOUT_SECONDS = 120        # API timeout
# Processing entire dataset - no sampling
```

## Prompting Modes

### 1. Zero-shot
- Direct question without examples
- Simple format: "Index: X, Answer: \"text\""

### 2. Few-shot 
- Includes 3 example riddle-answer pairs
- Same format as zero-shot

### 3. Chain-of-Thought (CoT)
- Step-by-step reasoning in English
- Final answer in Bengali
- Format: Reasoning steps + Final Answer

## Output Files

The script generates these files in the output directory:

- `riddles_semantic_ambiguity_gemini_flash_zero_shot.json` - Zero-shot results (full dataset)
- `riddles_semantic_ambiguity_gemini_flash_few_shot.json` - Few-shot results (full dataset)
- `riddles_semantic_ambiguity_gemini_flash_chain_of_thought.json` - CoT results (full dataset)
- `riddles_semantic_ambiguity_metrics_gemini_flash_zero_shot.json` - Zero-shot metrics
- `riddles_semantic_ambiguity_metrics_gemini_flash_few_shot.json` - Few-shot metrics
- `riddles_semantic_ambiguity_metrics_gemini_flash_chain_of_thought.json` - CoT metrics

## Error Handling

- **API Key Rotation**: Automatic rotation across multiple API keys when encountering rate limits
- **Rate Limit Management**: Smart backoff strategies for quota (50/min) and daily (1000/day) limits
- **Network Timeouts**: Enhanced timeout handling with 120-second limits and retry logic
- **Parse Errors**: Graceful fallback to text matching with fuzzy string matching
- **Resume**: Automatically skips processed examples for interrupted runs
- **Logging**: Detailed progress and error reporting with API key usage statistics

## Accuracy Calculation

The system calculates accuracy by:
1. Extracting predicted index (0-3) from model response
2. Comparing with ground truth index
3. Computing percentage: `(correct_predictions / valid_predictions) * 100`

Both index-based and text-based matching are supported with automatic reconciliation.

## API Key Management

The system includes a sophisticated API key rotation mechanism:

### Features:
- **25 Pre-configured Keys**: Automatic cycling through multiple API keys
- **Smart Rate Limiting**: Handles both per-minute (50 requests) and daily (1000 requests) limits
- **Failure Detection**: Automatically marks and skips failed/exhausted keys
- **Usage Tracking**: Monitors request counts per key for optimization
- **Backoff Strategies**: 
  - Rate limits: 15-25 second delays
  - Network errors: Progressive backoff up to 180 seconds
  - General errors: Exponential backoff up to 120 seconds

### Rate Limit Handling:
1. When a 429/quota error occurs, the system immediately switches to the next key
2. Implements intelligent delays based on error type
3. Continues processing without manual intervention
4. Provides detailed statistics on key usage and failures

This ensures robust, uninterrupted evaluation even with API limitations.