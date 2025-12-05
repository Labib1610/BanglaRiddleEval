#!/usr/bin/env python3
"""
Runner script for gemma3:12b MCQ evaluation
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    print("🎯 Bengali Riddle MCQ Evaluation - Gemma3:12b")
    print("=" * 50)
    
    try:
        # Import and run the evaluation
        from gemma3_12b import main as eval_main
        eval_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure ollama-python and tqdm are installed:")
        print("pip install ollama-python tqdm")
        return 1
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return 1
    
    print("\n✅ Evaluation completed!")
    return 0

if __name__ == "__main__":
    exit(main())