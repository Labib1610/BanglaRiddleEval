#!/usr/bin/env python3
"""
Script to show MCQ generation progress and status
"""

import json
import os

BASE_PATH = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada"
RIDDLES_FILE = f"{BASE_PATH}/dataset/riddles.json"
MCQ_FILE = f"{BASE_PATH}/dataset/riddles_mcq.json"
PROGRESS_FILE = f"{BASE_PATH}/dataset/mcq_progress.json"

def show_mcq_status():
    """Show current MCQ generation status."""
    
    print("=== MCQ Generation Status ===")
    
    # Load original riddles count
    total_riddles = 0
    if os.path.exists(RIDDLES_FILE):
        try:
            with open(RIDDLES_FILE, 'r', encoding='utf-8') as f:
                riddles = json.load(f)
            total_riddles = len(riddles)
            print(f"Total riddles in dataset: {total_riddles}")
        except Exception as e:
            print(f"Error reading riddles: {e}")
    
    # Show MCQ generation progress
    generated_mcqs = 0
    if os.path.exists(MCQ_FILE):
        try:
            with open(MCQ_FILE, 'r', encoding='utf-8') as f:
                mcqs = json.load(f)
            generated_mcqs = len(mcqs)
            print(f"Generated MCQs: {generated_mcqs}")
            
            if total_riddles > 0:
                percentage = (generated_mcqs / total_riddles) * 100
                print(f"Progress: {percentage:.1f}%")
            
            # Show sample MCQ
            if mcqs:
                sample = mcqs[0]
                print(f"\nSample MCQ:")
                print(f"  ID: {sample['id']}")
                print(f"  Question: {sample['question'][:50]}...")
                print(f"  Options: {sample['options']}")
                print(f"  Correct: {sample['correct_answer']}")
                
        except Exception as e:
            print(f"Error reading MCQs: {e}")
    else:
        print("No MCQs generated yet.")
    
    # Show detailed progress
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            
            print(f"\n=== Detailed Progress ===")
            print(f"Processed: {progress.get('processed_count', 0)}")
            print(f"Failed: {progress.get('failed_count', 0)}")
            print(f"Last updated: {progress.get('last_updated', 'Unknown')}")
            
            failed_riddles = progress.get('failed_riddles', [])
            if failed_riddles:
                print(f"Failed riddle IDs: {failed_riddles[:10]}" + ("..." if len(failed_riddles) > 10 else ""))
            
        except Exception as e:
            print(f"Error reading progress: {e}")
    
    # Show remaining work
    remaining = total_riddles - generated_mcqs
    if remaining > 0:
        print(f"\nRemaining riddles to process: {remaining}")
    
    print(f"\nFiles:")
    print(f"  Riddles: {RIDDLES_FILE}")
    print(f"  MCQs: {MCQ_FILE}")
    print(f"  Progress: {PROGRESS_FILE}")
    
    print(f"\nCommands:")
    print(f"  Generate all: python run_mcq_generation.py")
    print(f"  Test mode: python run_mcq_generation.py --test")
    print(f"  Retry failed: python run_mcq_generation.py --retry-failed")

if __name__ == "__main__":
    show_mcq_status()