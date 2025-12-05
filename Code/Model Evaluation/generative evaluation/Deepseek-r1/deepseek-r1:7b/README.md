# DeepSeek-R1:7b Evaluation Suite

This directory contains both discriminative (MCQ) and generative evaluation pipelines for testing the DeepSeek-R1:7b model on Bengali riddles.

## Evaluation Types

### 1. Discriminative (MCQ) Evaluation
- Multiple-choice question answering
- Index-based prediction with option selection
- Accuracy based on correct option selection

### 2. Generative Evaluation  
- Open-ended riddle answer generation
- **Dual Metrics**: BERTScore + LLM-as-a-Judge
- Uses Google Gemini API for semantic evaluation

## Features

- **Three Prompting Modes**: Zero-shot, Few-shot, and Chain-of-Thought (CoT)
- **Full Dataset Processing**: Evaluates the complete dataset without sampling
- **Progress Tracking**: Resume capability with incremental saves
- **Cultural Context**: Bengali-aware evaluation
- **Clean API Integration**: Requires user-provided Google Gemini API keys

## Files

### Discriminative Evaluation
- `deepseek_r1_7b.py` - MCQ evaluation script
- `run_evaluation.py` - MCQ runner script

### Generative Evaluation
- `deepseek_r1_7b_generative.py` - Generative evaluation script with LLM judge
- `run_generative_evaluation.py` - Generative runner script

### Documentation
- `README.md` - This documentation

## Dataset Formats

### MCQ Dataset (`dataset_mcq_1244.json`)
```json
[
  {
    "id": 1,
    "question": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
    "options": ["আনারস", "কলা", "আম", "পেঁপে"],
    "correct_answer": "আনারস"
  }
]
```

### Generative Dataset (`riddles.json`)
```json
[
  {
    "riddle_id": 1,
    "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
    "ans": "আনারস"
  }
]
```

## Output Formats

### MCQ Results JSON
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

### Generative Results JSON
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

### MCQ Metrics JSON
```json
{
  "Accuracy (%)": 75.5,
  "n_examples_total": 300,
  "n_valid_evaluated": 295,
  "n_correct": 223,
  "n_with_reasoning": 100
}
```

### Generative Metrics JSON
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

# Make sure Ollama is running with deepseek-r1:7b model
ollama serve
ollama pull deepseek-r1:7b
```

### Run Evaluation

#### MCQ Evaluation
```bash
# Navigate to the directory
cd "/path/to/deepseek-r1:7b"

# Run MCQ evaluation
python run_evaluation.py

# Or run directly
python deepseek_r1_7b.py
```

#### Generative Evaluation
```bash
# Run generative evaluation (with LLM judge)
python run_generative_evaluation.py

# Or run directly
python deepseek_r1_7b_generative.py
```

### Configuration

Edit the CONFIG section in `deepseek_r1_7b_generative.py`:

```python
MODEL_NAME = "deepseek-r1:7b"      # Ollama model name
LLM_NUM_CTX = 4096                 # Context window size
JUDGE_MODEL_NAME = "gemini-2.5-flash"  # Gemini model for judging
KEY_LIST = ["your_api_key_here"]   # Add your Gemini API keys
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

### Generative Evaluation Files
- `riddles_generative_deepseek_r1_7b_zero_shot.json` - Zero-shot results
- `riddles_generative_deepseek_r1_7b_few_shot.json` - Few-shot results  
- `riddles_generative_deepseek_r1_7b_chain_of_thought.json` - CoT results
- `riddles_generative_metrics_deepseek_r1_7b_zero_shot.json` - Zero-shot metrics
- `riddles_generative_metrics_deepseek_r1_7b_few_shot.json` - Few-shot metrics
- `riddles_generative_metrics_deepseek_r1_7b_chain_of_thought.json` - CoT metrics

## Error Handling

- **API Failures**: 3-attempt retry with exponential backoff
- **Parse Errors**: Graceful fallback to text matching
- **Resume**: Automatically skips processed examples
- **Logging**: Detailed progress and error reporting

## Evaluation Metrics

The system evaluates answers using:
1. **LLM-as-a-Judge**: Semantic scoring (0.0 to 1.0) using Gemini
2. **BERTScore**: Semantic similarity using multilingual BERT
3. **Full Dataset**: No sampling - processes complete dataset

Metrics are computed automatically and saved with results.