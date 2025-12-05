#!/usr/bin/env python3
"""
Runner script for Gemini Flash Lite Generative Evaluation
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    print("🎯 Bengali Riddle Generative Evaluation - Gemini 2.5 Flash Lite")
    print("=" * 55)
    print("📊 Dual Gemini Model Architecture with Split API Keys")
    print("🔄 LLM-as-a-Judge + Levenshtein + BERTScore Metrics")
    print("=" * 55)
    
    try:
        # Import and run the evaluation
        from gemini_flash import main as eval_main
        eval_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure required packages are installed:")
        print("pip install google-generativeai tqdm bert-score")
        return 1
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return 1
    
    print("\n✅ Generative evaluation completed!")
    print("📁 Check output files for detailed results and metrics")
    return 0

if __name__ == "__main__":
    exit(main())