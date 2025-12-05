# DeepSeek-R1:14b Generative Evaluation

This directory contains the generative evaluation pipeline for testing the DeepSeek-R1:14b model on Bengali riddle answer generation.

## Features

- **Generative Evaluation**: Open-ended riddle answer generation
- **Dual Metrics**: BERTScore + LLM-as-a-Judge evaluation
- **Full Dataset Processing**: Evaluates the complete dataset without sampling
- **Progress Tracking**: Resume capability with incremental saves
- **Cultural Context**: Bengali-aware evaluation
- **Clean API Integration**: Requires user-provided Google Gemini API keys
- **Three Prompting Modes**: Zero-shot, Few-shot, and Chain-of-Thought (CoT)

## Files

### Generative Evaluation
- `deepseek_r1_14b_generative.py` - Main generative evaluation script with LLM judge
- `run_generative_evaluation.py` - Simple runner script
- `README.md` - This documentation

## Dataset Format

The script expects riddles data in this format (`riddles.json`):
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
  "bert_precision": 0.95,
  "bert_recall": 0.92,
  "bert_f1": 0.935,
  "reasoning_text": "...",
  "reasoning_steps_en": ["Step 1: ...", "Step 2: ..."]
}
```

### Metrics JSON
```json
{
  "LLM Judge Average Score": 0.782,
  "LLM Judge Accuracy (%)": 78.2,
  "Average BERTScore F1": 0.654,
  "n_examples_total": 1500,
  "avg_judge_score": 0.782,
  "avg_bert_f1": 0.654,
  "n_with_reasoning": 500
}
```

## Usage

### Prerequisites
```bash
# Install required packages
pip install ollama-python google-generativeai tqdm

# Make sure Ollama is running with deepseek-r1:14b model
ollama serve
ollama pull deepseek-r1:14b
```

### Run Evaluation

```bash
# Navigate to the directory
cd "/path/to/deepseek-r1:14b"

# Run generative evaluation (with LLM judge)
python run_generative_evaluation.py

# Or run directly
python deepseek_r1_14b_generative.py
```

### Configuration

Edit the CONFIG section in `deepseek_r1_14b_generative.py`:

```python
MODEL_NAME = "deepseek-r1:14b"       # Ollama model name
JUDGE_MODEL_NAME = "gemini-2.5-flash"  # Gemini model for judging
KEY_LIST = ["your_api_key_here"]     # Add your Gemini API keys
```

## Prompting Modes

### 1. Zero-shot
- Direct riddle-to-answer generation
- No examples provided
- Simple format: "Answer: <Bengali_answer>"

### 2. Few-shot 
- Includes 3 Bengali example riddle-answer pairs
- Same response format as zero-shot
- Helps with format consistency

### 3. Chain-of-Thought (CoT)
- Step-by-step reasoning in English
- Final answer in Bengali
- Format: Reasoning steps + Final Answer

## Output Files

The script generates these files in the output directory:

- `riddles_generative_deepseek_r1_14b_zero_shot.json` - Zero-shot results (full dataset)
- `riddles_generative_deepseek_r1_14b_few_shot.json` - Few-shot results (full dataset)
- `riddles_generative_deepseek_r1_14b_chain_of_thought.json` - CoT results (full dataset)
- `riddles_generative_metrics_deepseek_r1_14b_zero_shot.json` - Zero-shot metrics
- `riddles_generative_metrics_deepseek_r1_14b_few_shot.json` - Few-shot metrics
- `riddles_generative_metrics_deepseek_r1_14b_chain_of_thought.json` - CoT metrics

## Error Handling

- **Ollama Failures**: 3-attempt retry with exponential backoff
- **Judge API Management**: Automatic key rotation on rate limits (429 errors)
- **Network Timeouts**: Progressive backoff up to 120 seconds for network issues
- **Parse Errors**: Graceful answer extraction with fallback methods
- **Resume**: Automatically skips processed examples for interrupted runs
- **Logging**: Detailed progress and error reporting with API usage statistics

## Evaluation Metrics

The system evaluates answers using:
1. **LLM-as-a-Judge**: Semantic scoring (0.0 to 1.0) using Gemini 2.5 Flash
2. **BERTScore**: Semantic similarity using multilingual BERT
3. **Full Dataset**: No sampling - processes complete dataset

### LLM-as-a-Judge
- **Judge Model**: Google Gemini 2.5 Flash
- **Evaluation Criteria**:
  - Exact match detection
  - Semantic equivalence assessment  
  - Bengali cultural context consideration
  - Acceptable synonym recognition
  - Spelling variation tolerance
- **Scoring**: 0.0 to 1.0 scale for precise evaluation
- **Pros**: Captures semantic similarity, cultural awareness
- **Cons**: Requires API calls

### BERTScore
- **Model**: Multilingual BERT for Bengali language support
- **Metrics**: Precision, Recall, and F1 scores
- **Pros**: Semantic similarity without API dependency
- **Cons**: May not capture cultural nuances as well as human/LLM judgment

Metrics are computed automatically and saved with results.