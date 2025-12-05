# Gemma 3:12b Generative Evaluation

This directory contains the generative evaluation pipeline for testing the Gemma 3:12b model on Bengali riddle answer generation.

## Features

- **Generative Evaluation**: Open-ended riddle answer generation on full dataset
- **Dual Metrics**: BERTScore + LLM-as-a-Judge evaluation
- **Google Gemini Judge**: Uses Gemini 2.5 Flash for semantic evaluation
- **Bengali Cultural Context**: Considers linguistic variations and synonyms
- **Three Prompting Modes**: Zero-shot, Few-shot, and Chain-of-Thought (CoT)
- **Full Dataset Processing**: Evaluates entire riddle dataset
- **Progress Tracking**: Resume capability with incremental saves
- **Cultural Context**: Bengali-aware evaluation

## Files

### Generative Evaluation
- `gemma3_12b_generative.py` - Main generative evaluation script with LLM judge
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
  "exact_match": true,
  "llm_judge_score": 1.0,
  "reasoning_text": "...",
  "reasoning_steps_en": ["Step 1: ...", "Step 2: ..."]
}
```

### Metrics JSON
```json
{
  "Exact Match Accuracy (%)": 65.5,
  "LLM Judge Average Score": 0.782,
  "LLM Judge Accuracy (%)": 78.2,
  "n_examples_total": 300,
  "n_exact_matches": 197,
  "avg_judge_score": 0.782,
  "n_with_reasoning": 100
}
```

## Usage

### Prerequisites
```bash
# Install required packages
pip install ollama-python google-generativeai tqdm

# Make sure Ollama is running with gemma3:12b model
ollama serve
ollama pull gemma3:12b
```

### Run Evaluation

```bash
# Navigate to the directory
cd "/path/to/gemma3:12b"

# Run generative evaluation (with LLM judge)
python run_generative_evaluation.py

# Or run directly
python gemma3_12b_generative.py
```

### Configuration

Edit the CONFIG section in `gemma3_12b_generative.py`:

```python
MODEL_NAME = "gemma3:12b"       # Ollama model name
JUDGE_MODEL_NAME = "gemini-2.5-flash"  # Gemini judge model
# Add your Google Gemini API key to KEY_LIST
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

- `riddles_generative_gemma3_12b_zero_shot.json` - Zero-shot results (full dataset)
- `riddles_generative_gemma3_12b_few_shot.json` - Few-shot results (full dataset)
- `riddles_generative_gemma3_12b_chain_of_thought.json` - CoT results (full dataset)
- `riddles_generative_metrics_gemma3_12b_zero_shot.json` - Zero-shot metrics
- `riddles_generative_metrics_gemma3_12b_few_shot.json` - Few-shot metrics
- `riddles_generative_metrics_gemma3_12b_chain_of_thought.json` - CoT metrics

## Error Handling

- **Ollama Failures**: 3-attempt retry with exponential backoff
- **Judge API Management**: Retry logic for API failures
- **Parse Errors**: Graceful answer extraction with fallback methods
- **Resume**: Automatically skips processed examples for interrupted runs
- **Logging**: Detailed progress and error reporting

## Evaluation Metrics

### BERTScore
- **Model**: bert-base-multilingual-cased
- **Metrics**: Precision, Recall, F1-score
- **Language**: Multilingual support for Bengali
- **Pros**: Semantic similarity, contextual understanding
- **Cons**: Requires computational resources

### LLM-as-a-Judge
- **Judge Model**: Google Gemini 2.5 Flash
- **Evaluation Criteria**:
  - Exact match detection
  - Semantic equivalence assessment  
  - Bengali cultural context consideration
  - Acceptable synonym recognition
  - Spelling variation tolerance
  - Partial correctness scoring (0-1 scale)
- **Scoring**: 0-1 numerical scores with partial credit
- **Pros**: Captures semantic similarity, cultural awareness
- **Cons**: Requires API calls, potential inconsistency