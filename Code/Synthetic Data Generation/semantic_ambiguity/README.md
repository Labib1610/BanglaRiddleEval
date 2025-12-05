# Semantic Ambiguity Task Generator for Bengali Riddles

This module generates semantic ambiguity tasks from Bengali riddles, identifying metaphorical or ambiguous words/phrases and creating multiple-choice questions about their meaning in context.

## Features

- **Robust Generation**: Uses Ollama gpt-oss:20b model for high-quality semantic analysis
- **Error Handling**: 3-attempt retry mechanism with exponential backoff
- **Progress Tracking**: Resume capability and comprehensive progress monitoring
- **Quality Validation**: Automatic validation of generated tasks
- **Batch Processing**: Process individual riddles or entire dataset

## File Structure

```
semantic_ambiguity/
├── semantic_ambiguity.py      # Main generator class
├── run_semantic_generation.py # CLI interface for batch processing
├── debug_semantic.py          # Debug and testing utilities
├── monitor_semantic_progress.py # Progress monitoring
└── README.md                  # This file
```

## Output Format

Each semantic ambiguity task follows this structure:

```json
{
  "id": 1,
  "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
  "ambiguous_word": "টিয়ে",
  "question": "এই ধাঁধায় 'টিয়ে' শব্দটি কী বোঝায়?",
  "options": [
    "ফলের সবুজ পাতা",
    "সত্যিকারের পাখি",
    "টিয়ে নামের কোন ব্যক্তি", 
    "সোনার মুকুট"
  ],
  "correct_option": "ফলের সবুজ পাতা"
}
```

## Key Components

- **id**: Original riddle identifier
- **riddle**: Original Bengali riddle text
- **ambiguous_word**: The metaphorical/ambiguous word or phrase
- **question**: Bengali question asking what the word refers to
- **options**: 4 possible Bengali interpretations (1 correct, 3 plausible distractors)
- **correct_option**: The semantically correct interpretation in Bengali

## Usage

### Quick Start
```bash
cd semantic_ambiguity/
python run_semantic_generation.py
```

### Debug Single Riddle
```bash
python debug_semantic.py
```

### Monitor Progress
```bash
python monitor_semantic_progress.py
```

### Direct Python Usage
```python
from semantic_ambiguity import SemanticAmbiguityGenerator

generator = SemanticAmbiguityGenerator()

# Process sample riddles
generator.process_all_riddles(start_idx=0, end_idx=5)

# Process all riddles
generator.process_all_riddles()
```

## Output Files

- `riddles_semantic_ambiguity.json`: Main output with all generated tasks
- `semantic_progress.json`: Progress tracking and statistics
- `semantic_failed.json`: Failed riddles for manual review

## Quality Criteria

Tasks must meet these validation criteria:
- Exactly 4 unique options
- Correct option exists in options list
- Question asks about semantic meaning ("refer to", "mean", "represent")
- No duplicate or identical options
- Ambiguous word/phrase identified from riddle

## Generation Strategy

The system identifies semantic ambiguity through:

1. **Metaphor Detection**: Finding words used metaphorically (টিয়ে = parrot → green leaves)
2. **Context Analysis**: Understanding riddle's symbolic meaning
3. **Distractor Creation**: Generating plausible but incorrect interpretations
4. **Quality Assurance**: Validating logical consistency and clarity

## Requirements

- Ollama running with gpt-oss:20b model
- Python 3.7+
- requests library
- Bengali riddles dataset (riddles.json)

## Error Handling

The system handles various error scenarios:
- Connection timeouts (model loading)
- JSON parsing errors
- Incomplete task generation
- API failures with automatic retry

Progress is saved incrementally to prevent data loss during long batch operations.