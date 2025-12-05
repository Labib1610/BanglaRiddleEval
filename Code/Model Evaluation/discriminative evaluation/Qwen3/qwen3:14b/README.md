# Qwen3:14b MCQ Evaluation

This directory contains the evaluation pipeline for testing the Qwen3:14b model on Bengali riddle multiple-choice questions.

## Features

- **Three Prompting Modes**: Zero-shot, Few-shot, and Chain-of-Thought (CoT)
- **Robust Parsing**: Extracts both index and answer text from model responses
- **Accuracy Calculation**: Direct matching of predicted vs ground truth indices
- **Progress Tracking**: Resume capability with incremental saves
- **Error Handling**: Retry mechanism for failed API calls

## Files

- `qwen3_14b.py` - Main evaluation script
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
pip install ollama-python tqdm

# Make sure Ollama is running with qwen3:14b model
ollama serve
ollama pull qwen3:14b
```

### Run Evaluation
```bash
# Navigate to the directory
cd "/path/to/qwen3:14b"

# Run the evaluation
python run_evaluation.py

# Or run directly
python qwen3_14b.py
```

### Configuration

Edit the CONFIG section in `qwen3_14b.py`:

```python
MODEL_NAME = "qwen3:14b"          # Ollama model name
LLM_NUM_CTX = 4096                 # Context window size
N_EXAMPLES = None                  # Limit examples (None = all)
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

- `riddles_mcq_qwen3_14b_zero.json` - Zero-shot results
- `riddles_mcq_qwen3_14b_few.json` - Few-shot results  
- `riddles_mcq_qwen3_14b_cot.json` - CoT results
- `riddles_mcq_metrics_qwen3_14b_zero.json` - Zero-shot metrics
- `riddles_mcq_metrics_qwen3_14b_few.json` - Few-shot metrics
- `riddles_mcq_metrics_qwen3_14b_cot.json` - CoT metrics

## Error Handling

- **API Failures**: 3-attempt retry with exponential backoff
- **Parse Errors**: Graceful fallback to text matching
- **Resume**: Automatically skips processed examples
- **Logging**: Detailed progress and error reporting

## Accuracy Calculation

The system calculates accuracy by:
1. Extracting predicted index (0-3) from model response
2. Comparing with ground truth index
3. Computing percentage: `(correct_predictions / valid_predictions) * 100`

Both index-based and text-based matching are supported with automatic reconciliation.