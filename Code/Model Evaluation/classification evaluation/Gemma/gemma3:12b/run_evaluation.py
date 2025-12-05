#!/usr/bin/env python3
"""
Runner script for Gemma3:12b Semantic Ambiguity evaluation
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    print("🎯 Bengali Riddle Semantic Ambiguity Evaluation - Gemma3:12b")
    print("=" * 65)
    print("📊 Analyzing metaphorical meanings in riddle terms")
    print("🔄 Three prompting modes: Zero-shot, Few-shot (3-shot), Chain-of-Thought")
    print("=" * 65)
    
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
    
    print("\n✅ Semantic Ambiguity evaluation completed!")
    print("📁 Check output files for detailed results and accuracy metrics")
    return 0

if __name__ == "__main__":
    exit(main())