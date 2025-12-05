# DeepSeek R1 7B Semantic Ambiguity Evaluation

This directory contains the evaluation pipeline for testing the DeepSeek R1 7B model on Bengali riddle semantic ambiguity questions.

## Features

- **Complete Dataset Processing**: Processes all riddles in the semantic ambiguity dataset
- **Three Prompting Modes**: Zero-shot, Few-shot (3-shot), and Chain-of-Thought (CoT)
- **Semantic Ambiguity Analysis**: Analyzes metaphorical meanings in Bengali riddle terms
- **Contextual Understanding**: Uses riddle + answer + semantic question for comprehensive analysis
- **Robust Parsing**: Extracts both index and answer text from model responses
- **Accuracy Calculation**: Direct matching of predicted vs ground truth indices
- **Progress Tracking**: Resume capability with incremental saves
- **Error Handling**: Retry mechanism for failed Ollama calls

## Files

- `deepseek_r1_7b.py` - Main evaluation script
- `README.md` - This documentation

## Dataset Format

The script loads semantic ambiguity data from `/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/dataset/riddles_semantic_ambiguity.json` in this format:
```json
[
  {
    "id": 1,
    "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
    "ans": "আনারস",
    "ambiguous_word": "সোনার টোপর",
    "question": "এই ধাঁধায় 'সোনার টোপর' শব্দটি কী বোঝায়?",
    "options": [
      "সোনার তৈরি টুপি",
      "আনারসের শীর্ষের হলুদ অংশ",
      "পাখির গলা",
      "সোনার তৈরি মুকুট"
    ],
    "correct_option": "আনারসের শীর্ষের হলুদ অংশ"
  }
]
```

## Output Format

### Results JSON
```json
{
  "riddle_id": 1,
  "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
  "riddle_ans": "আনারস",
  "semantic_question": "এই ধাঁধায় 'সোনার টোপর' শব্দটি কী বোঝায়?",
  "ambiguous_word": "সোনার টোপর",
  "options": ["সোনার তৈরি টুপি", "আনারসের শীর্ষের হলুদ অংশ", "পাখির গলা", "সোনার তৈরি মুকুট"],
  "ground_truth_index": 1,
  "ground_truth_answer": "আনারসের শীর্ষের হলুদ অংশ",
  "predicted_index": 1,
  "predicted_answer_text": "আনারসের শীর্ষের হলুদ অংশ",
  "reasoning_text": "...",
  "reasoning_steps_en": ["Step 1: ...", "Step 2: ..."],
  "raw_response": "Index: 1, Answer: \"আনারসের শীর্ষের হলুদ অংশ\""
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
pip install ollama tqdm

# Make sure Ollama is running with DeepSeek R1 7B model
ollama serve
ollama pull deepseek-r1:7b
```

### Run Evaluation
```bash
# Navigate to the directory
cd "/path/to/deepseek-r1:7b"

# Run the evaluation directly
python deepseek_r1_7b.py
```

### Configuration

Edit the CONFIG section in `deepseek_r1_7b.py`:

```python
MODEL_NAME = "deepseek-r1:7b"      # Ollama model name
LLM_NUM_CTX = 4096                # Context window size
N_EXAMPLES = None                  # Limit examples (None = all)
```

## Prompting Modes

### 1. Zero-shot
- Direct question without examples
- Simple format: "Index: X, Answer: \"text\""

### 2. Few-shot (3-shot)
- Includes 3 semantic ambiguity examples
- Shows riddle + answer + question + correct semantic interpretation
- Same format as zero-shot

### 3. Chain-of-Thought (CoT)
- Step-by-step semantic ambiguity analysis in English
- Analyzes riddle context, identifies ambiguous terms, evaluates options
- Final answer in Bengali
- Format: Reasoning steps + Final Answer

## Output Files

The script generates these files in the output directory:

- `riddles_semantic_ambiguity_deepseek_r1_7b_zero.json` - Zero-shot results
- `riddles_semantic_ambiguity_deepseek_r1_7b_few.json` - Few-shot results  
- `riddles_semantic_ambiguity_deepseek_r1_7b_cot.json` - CoT results
- `riddles_semantic_ambiguity_metrics_deepseek_r1_7b_zero.json` - Zero-shot metrics
- `riddles_semantic_ambiguity_metrics_deepseek_r1_7b_few.json` - Few-shot metrics
- `riddles_semantic_ambiguity_metrics_deepseek_r1_7b_cot.json` - CoT metrics

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