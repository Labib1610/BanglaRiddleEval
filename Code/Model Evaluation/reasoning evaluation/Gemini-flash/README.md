# Gemini-2.5-Flash Bengali Riddle Evaluation

This directory contains the evaluation setup for **Gemini-2.5-Flash** model on Bengali riddle tasks including both generative and reasoning evaluations.

## Files

### Generative Evaluation (Answer Generation)
- `gemini_flash_lite.py` - Original generative evaluation script
- `run_evaluation.py` - Runner script for generative evaluation

### Reasoning Evaluation (Reasoning Generation)  
- `gemini_flash_lite_reasoning.py` - Main reasoning evaluation script with Google Gemini API integration
- `run_reasoning_evaluation.py` - Runner script to execute reasoning evaluation
- `README.md` - This documentation file

## Model Information

- **Model**: gemini-2.5-flash
- **API**: Google Generative AI with key rotation
- **Tasks**: 
  - Bengali riddle answer generation (generative)
  - Bengali riddle reasoning generation (reasoning)
- **Dataset**: Bengali riddles from the main dataset
- **Processing**: Full dataset evaluation (no sampling)

## Features

### Google Gemini API Integration
- Uses Google Generative AI SDK (`google-generativeai`)
- Implements API key rotation for high availability
- Separate client pools for evaluation and judging
- Built-in rate limiting and error handling
- Network timeout management

### Dual Evaluation System
- **Primary Generation**: Gemini-2.5-Flash for answer/reasoning generation
- **LLM-as-a-Judge**: Same model for quality assessment
- **LLM Judge**: Multilingual BERT for semantic similarity

### Multi-Mode Evaluation
Supports three different prompting strategies:
1. **Zero-shot**: Direct riddle-to-answer/reasoning generation
2. **Few-shot**: With example demonstrations
3. **Chain-of-thought**: Step-by-step reasoning prompts

### Key Rotation Strategy
- **API keys** split into two pools:
  - **First half**: LLM-as-a-Judge evaluation
  - **Second half**: Primary generation
- Automatic failover and rate limit handling
- Usage statistics tracking per key
- Add your own API keys in the KEY_LIST configuration

## Usage

### Prerequisites
```bash
pip install google-generativeai tqdm
```

**Important**: Add your Google Gemini API keys to the `KEY_LIST` in both scripts before running.

### Run Generative Evaluation (Answer Generation)
```bash
# Execute all three evaluation modes for answer generation
python run_evaluation.py

# Or run the main script directly
python gemini_flash_lite.py
```

### Run Reasoning Evaluation (Reasoning Generation)
```bash
# Execute all three evaluation modes for reasoning generation
python run_reasoning_evaluation.py

# Or run the main script directly
python gemini_flash_lite_reasoning.py
```

### Output Files

**Generative Evaluation Results:**
- `riddles_generative_gemini_flash_zero_shot.json`
- `riddles_generative_gemini_flash_few_shot.json` 
- `riddles_generative_gemini_flash_chain_of_thought.json`

**Generative Evaluation Metrics:**
- `riddles_generative_metrics_gemini_flash_zero_shot.json`
- `riddles_generative_metrics_gemini_flash_few_shot.json`
- `riddles_generative_metrics_gemini_flash_chain_of_thought.json`

**Reasoning Evaluation Results:**
- `riddles_reasoning_gemini_flash_zero_shot.json`
- `riddles_reasoning_gemini_flash_few_shot.json`
- `riddles_reasoning_gemini_flash_chain_of_thought.json`

**Reasoning Evaluation Metrics:**
- `riddles_reasoning_metrics_gemini_flash_zero_shot.json`
- `riddles_reasoning_metrics_gemini_flash_few_shot.json`
- `riddles_reasoning_metrics_gemini_flash_chain_of_thought.json`

## Configuration

### API Keys
**Required**: Edit the `KEY_LIST` in both scripts to add your Google Gemini API keys:

```python
KEY_LIST = [
    "your-api-key-1",
    "your-api-key-2",
    # Add more keys as needed
]
```

### Model Settings
- **Temperature**: 0.7 for generation, 0.1 for judging
- **Max Tokens**: 150 for answers, 512 for reasoning, 20 for judging
- **Top-k/Top-p**: Optimized for Bengali text generation

### Dataset Paths
Update these paths in the scripts:
```python
# For generative evaluation
RIDDLES_DATASET_PATH = "path/to/riddles_generative.json"

# For reasoning evaluation  
RIDDLES_DATASET_PATH = "path/to/riddles_reasoning.json"

OUTPUT_ROOT = "path/to/output/directory"
```

## Evaluation Metrics

### Generative Evaluation Metrics
- **Exact Match**: Binary exact string matching
- **Fuzzy Match**: Character similarity with 90% threshold
- **LLM Judge Score**: AI-based answer quality assessment (0-1 scale)
- **LLM Judge**: Semantic similarity (Precision/Recall/F1)

### Reasoning Evaluation Metrics
- **LLM Judge Score**: AI-based reasoning quality assessment (0-1 scale)
- **LLM Judge F1**: Semantic similarity between generated and ground truth reasoning
- **Reasoning Quality**: Evaluates logical structure, accuracy, completeness, and cultural context

### Performance Indicators
- **Average Accuracy**: Combined performance across all modes
- **Average LLM Judge**: Semantic similarity performance
- **Processing Statistics**: Response times and success rates

## Bengali Reasoning Format

The reasoning evaluation follows a structured 4-step Bengali analysis format:
1. **উত্তর চিহ্নিতকরণ** (Answer Identification): Quote specific words from the riddle
2. **রূপকের ব্যাখ্যা** (Metaphor Explanation): Explain what the metaphor represents
3. **উত্তরের সাথে সংযোগ** (Connection to Answer): Explain how the answer matches the riddle
4. **সিদ্ধান্ত** (Conclusion): Why this is the only logical answer

## Error Handling

The system includes robust error handling:
- **Rate Limiting**: Automatic backoff with exponential delays
- **Network Issues**: Retry logic with timeout management
- **API Failures**: Key rotation and fallback mechanisms
- **Invalid Responses**: Graceful handling of malformed outputs

## Notes

- Processes the entire dataset (no sampling)
- Implements incremental saving to prevent data loss
- Supports resume capability for interrupted evaluations
- Includes comprehensive logging and progress tracking
- Both evaluations use the same API infrastructure but different datasets and prompts
- **Important**: You must add your own Google Gemini API keys before running

