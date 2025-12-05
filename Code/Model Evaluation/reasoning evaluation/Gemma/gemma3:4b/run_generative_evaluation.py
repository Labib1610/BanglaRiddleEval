#!/usr/bin/env python3
"""
Runner script for Gemma3:4b Bengali Reasoning evaluation
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    print("🎯 Bengali Riddle Reasoning Evaluation - Gemma3:4b")
    print("=" * 60)
    
    try:
        # Import and run the evaluation
        from gemma3_4b_generative import main as eval_main
        eval_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure ollama-python, google-generativeai, tqdm and bert-score are installed:")
        print("pip install ollama-python google-generativeai tqdm bert-score")
        return 1
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return 1
    
    print("\n✅ Evaluation completed!")
    return 0

if __name__ == "__main__":
    exit(main())