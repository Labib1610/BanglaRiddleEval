# Qwen3:8b Reasoning Evaluation Suite

This directory contains reasoning evaluation pipeline for testing the Qwen3:8b model on Bengali riddle reasoning generation.

## Evaluation Type

### Bengali Reasoning Evaluation
- Detailed Bengali reasoning generation for riddle solutions
- 4-step analysis in Bengali paragraph format
- **Dual Metrics**: LLM Judge + LLM-as-a-Judge for reasoning quality
- Uses Google Gemini API with rotating keys for reasoning quality assessment

## Features

- **Single Bengali Prompt**: All modes use the same Bengali reasoning prompt
- **Random Sampling**: 150 samples for reasoning evaluation
- **Robust API Management**: Key rotation and rate limit handling for LLM judge
- **Progress Tracking**: Resume capability with incremental saves
- **Cultural Context**: Bengali-aware reasoning evaluation with 4-step analysis format

## Files

### Reasoning Evaluation
- `qwen3_8b_generative.py` - Reasoning evaluation script with LLM judge
- `run_generative_evaluation.py` - Reasoning runner script

### Documentation
- `README.md` - This documentation

## Dataset Format

### Reasoning Dataset (`riddles_reasoning.json`)
```json
[
  {
    "id": 1,
    "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
    "ans": "আনারস",
    "reasoning": "১. 'বন থেকে বেরুল টিয়ে': এখানে টিয়ে মানে পাখি নয়, বরং আনারস ফলকে বোঝানো হয়েছে...।"
  }
]
```

## Output Format

### Reasoning Results JSON
```json
{
  "riddle_id": 1,
  "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
  "ground_truth_answer": "আনারস",
  "ground_truth_reasoning": "১. 'বন থেকে বেরুল টিয়ে': এখানে টিয়ে মানে পাখি নয়...",
  "generated_reasoning": "১. 'টিয়ে' শব্দটি: এই ধাঁধায় টিয়ে মানে পাখি নয়...",
  "llm_judge_score": 0.85,
  "bert_precision": 0.78,
  "bert_recall": 0.82,
  "bert_f1": 0.80
}
```

### Reasoning Metrics JSON
```json
{
  "LLM Judge Average Score": 0.756,
  "LLM Judge Reasoning Quality (%)": 75.6,
  "Average LLM Judge F1": 0.723,
  "LLM Judge Reasoning Similarity (%)": 72.3,
  "n_examples_total": 150,
  "avg_judge_score": 0.756,
  "avg_bert_f1": 0.723
}
```

## Usage

### Prerequisites
```bash
# Install required packages
pip install ollama-python google-generativeai tqdm bert-score

# Make sure Ollama is running with qwen3:8b model
ollama serve
ollama pull qwen3:8b
```

### Run Evaluation

#### Reasoning Evaluation
```bash
# Navigate to the directory
cd "/path/to/qwen3:8b"

# Run reasoning evaluation (with LLM judge and LLM Judge)
python run_generative_evaluation.py

# Or run directly
python qwen3_8b_generative.py
```

### Configuration

Edit the CONFIG section in `qwen3_8b_generative.py`:

```python
MODEL_NAME = "qwen3:8b"     # Ollama model name
LLM_NUM_CTX = 4096                # Context window size
RANDOM_SAMPLE_SIZE = 150          # Number of samples to evaluate
```

## Bengali Reasoning Prompt

### Single Reasoning Format
All three modes (zero_shot, few_shot, chain_of_thought) use the same Bengali reasoning prompt that asks for:

1. **উত্তর চিহ্নিতকরণ**: Identify specific words from the riddle
2. **রূপকের ব্যাখ্যা**: Explain what the metaphor represents
3. **উত্তরের সাথে সংযোগ**: Connect answer characteristics with the riddle
4. **সিদ্ধান্ত**: Brief conclusion about why this is the logical answer

### Output Format
- Complete Bengali paragraph with 4-step analysis
- Cultural context consideration
- No JSON or structured formatting required

## Output Files

The script generates these files in the output directory:

- `riddles_reasoning_qwen3_8b_zero_shot_sample150.json` - Zero-shot results
- `riddles_reasoning_qwen3_8b_few_shot_sample150.json` - Few-shot results  
- `riddles_reasoning_qwen3_8b_chain_of_thought_sample150.json` - CoT results
- `riddles_reasoning_metrics_qwen3_8b_zero_shot_sample150.json` - Zero-shot metrics
- `riddles_reasoning_metrics_qwen3_8b_few_shot_sample150.json` - Few-shot metrics
- `riddles_reasoning_metrics_qwen3_8b_chain_of_thought_sample150.json` - CoT metrics
- `sampled_ids.json` - List of sampled riddle IDs for consistency

## Error Handling

- **API Failures**: 3-attempt retry with exponential backoff for Ollama
- **LLM Judge Failures**: Key rotation and rate limit handling for Gemini API
- **Resume**: Automatically skips processed examples
- **Logging**: Detailed progress and error reporting

## Evaluation Metrics

The system evaluates reasoning quality using:

### 1. LLM-as-a-Judge (Gemini 2.5 Flash Lite)
- **Score Range**: 0.0 - 1.0
- **Criteria**: Logical structure, accuracy, completeness, cultural context, language quality
- **Output**: Average score and percentage

### 2. LLM Judge F1
- **Model**: bert-base-multilingual-cased
- **Comparison**: Generated reasoning vs ground truth reasoning
- **Output**: Semantic similarity score (0.0 - 1.0)

### 3. Sample Consistency
- Uses fixed random seed (42) for reproducible sampling
- Same 150 samples across all three prompting modes