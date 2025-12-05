#!/usr/bin/env python3
"""
run_reasoning_evaluation.py

Runner script for Bengali Riddle Reasoning evaluation using Gemini-2.5-Flash.
This script executes the reasoning evaluation pipeline across all prompt modes.
"""

import subprocess
import sys
from pathlib import Path

def run_reasoning_evaluation():
    """Run the Gemini-2.5-Flash reasoning evaluation."""
    
    print("🚀 Starting Bengali Riddle Reasoning Evaluation - Gemini-2.5-Flash")
    print("=" * 70)
    
    # Get the current directory (should contain the evaluation script)
    current_dir = Path(__file__).parent
    script_path = current_dir / "gemini_flash_reasoning.py"
    
    if not script_path.exists():
        print(f"❌ Error: {script_path} not found!")
        sys.exit(1)
    
    try:
        # Run the reasoning evaluation script
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False,
                              text=True,
                              cwd=current_dir)
        
        if result.returncode == 0:
            print("\n✅ Reasoning evaluation completed successfully!")
        else:
            print(f"\n❌ Reasoning evaluation failed with return code: {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\n⚠️ Reasoning evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running reasoning evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_reasoning_evaluation()