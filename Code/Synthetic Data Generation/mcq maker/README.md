# MCQ Generation from Bengali Riddles

This system converts Bengali riddles into Multiple Choice Questions (MCQ) format using the Ollama `gpt-oss:20b` model.

## Features

- **Automated MCQ Generation**: Creates 4-option MCQs with 1 correct and 3 plausible wrong answers
- **Progress Tracking**: Resumes from where it left off if interrupted
- **Failed Riddle Handling**: Tracks and allows retry of failed conversions
- **Ollama Integration**: Uses local Ollama model for generation

## Prerequisites

1. **Install Ollama**: Follow instructions at https://ollama.ai/
2. **Install the model**:
   ```bash
   ollama pull gpt-oss:20b
   ```
3. **Start Ollama server**:
   ```bash
   ollama serve
   ```

## Input Format

The system reads from: `/dataset/riddles.json`

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

Generates: `/dataset/riddles_mcq.json`

```json
[
  {
    "id": 1,
    "question": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
    "options": [
      "ভুট্টা",
      "আনারস",
      "কাঁঠাল", 
      "তাল"
    ],
    "correct_answer": "আনারস"
  }
]
```

## Usage

### Generate All MCQs
```bash
python run_mcq_generation.py
```

### Test Mode (first 5 riddles)
```bash
python run_mcq_generation.py --test
```

### Process Specific Range
```bash
python run_mcq_generation.py --start 100 --end 200
```

### Process Single Riddle
```bash
python run_mcq_generation.py --single 42
```

### Retry Failed Riddles
```bash
python run_mcq_generation.py --retry-failed
```

### Check Status
```bash
python show_mcq_status.py
```

## How It Works

1. **Loads riddles** from the input JSON file
2. **Connects to Ollama** using the gpt-oss:20b model
3. **Generates prompts** in Bengali asking for confusing but plausible options
4. **Parses responses** to extract 4 options including the correct answer
5. **Saves MCQs** in the specified format
6. **Tracks progress** and handles failures gracefully

## Error Handling

- **Connection Issues**: Checks Ollama availability before starting
- **Failed Generations**: Tracks failed riddles for manual review/retry
- **Progress Persistence**: Saves progress regularly, can resume anytime
- **Validation**: Ensures correct answer is included in generated options

## Files Generated

- `riddles_mcq.json` - Generated MCQ dataset
- `mcq_progress.json` - Progress and failed riddles tracking

## Monitoring

The system provides detailed progress information:

```
Processing riddle 1: বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।...
✓ Generated MCQ for riddle 1
  Options: ['ভুট্টা', 'আনারস', 'কাঁঠাল', 'তাল']
Progress: 10 processed - Success: 9, Failed: 1
```

## Customization

You can modify the prompt in `mcq.py` to:
- Change the number of options (default: 4)
- Adjust difficulty level of wrong answers
- Modify the instruction language
- Change response format requirements

## Troubleshooting

1. **"Cannot connect to Ollama"**: Ensure Ollama is running (`ollama serve`)
2. **"Model not available"**: Install the model (`ollama pull gpt-oss:20b`)
3. **Slow generation**: The model runs locally, speed depends on your hardware
4. **Invalid responses**: The system validates and retries failed generations

## Performance

- **Speed**: ~2-3 seconds per riddle (depends on hardware)
- **Accuracy**: Validates that correct answer is included in options
- **Reliability**: Automatic retry and error handling
- **Scalability**: Processes 1000+ riddles with progress tracking