#!/usr/bin/env python3
"""
Monitor Semantic Ambiguity Task Generation Progress
==================================================

This script monitors and displays the progress of semantic ambiguity task generation,
showing statistics and current status.
"""

import json
import os
from datetime import datetime

# File paths
BASE_PATH = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada"
RIDDLES_FILE = f"{BASE_PATH}/dataset/riddles.json"
OUTPUT_FILE = f"{BASE_PATH}/dataset/riddles_semantic_ambiguity.json"
PROGRESS_FILE = f"{BASE_PATH}/dataset/semantic_progress.json"
FAILED_FILE = f"{BASE_PATH}/dataset/semantic_failed.json"

def load_json_file(filepath):
    """Safely load a JSON file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def show_progress():
    """Show comprehensive progress information."""
    
    print("📊 Semantic Ambiguity Task Generation Progress")
    print("=" * 50)
    
    # Load all relevant files
    riddles = load_json_file(RIDDLES_FILE)
    semantic_tasks = load_json_file(OUTPUT_FILE)
    progress = load_json_file(PROGRESS_FILE)
    failed = load_json_file(FAILED_FILE)
    
    # Basic statistics
    total_riddles = len(riddles) if riddles else 0
    completed_tasks = len(semantic_tasks) if semantic_tasks else 0
    failed_count = len(failed) if failed else 0
    
    print(f"\n📚 Dataset Overview:")
    print(f"   Total riddles in dataset: {total_riddles:,}")
    print(f"   Completed semantic tasks: {completed_tasks:,}")
    print(f"   Failed attempts: {failed_count:,}")
    
    if total_riddles > 0:
        completion_rate = (completed_tasks / total_riddles) * 100
        print(f"   Completion rate: {completion_rate:.1f}%")
    
    # Progress details
    if progress:
        print(f"\n⏱️ Progress Details:")
        print(f"   Last updated: {progress.get('last_updated', 'Unknown')}")
        print(f"   Processed count: {progress.get('processed_count', 0):,}")
        print(f"   Failed count: {progress.get('failed_count', 0):,}")
    
    # Recent semantic tasks
    if semantic_tasks and len(semantic_tasks) > 0:
        print(f"\n✅ Recent Semantic Tasks:")
        for task in semantic_tasks[-3:]:  # Show last 3 tasks
            print(f"   ID {task['id']}: '{task['ambiguous_word']}' -> {len(task['options'])} options")
    
    # Failed riddles analysis
    if failed and len(failed) > 0:
        print(f"\n❌ Failed Riddles Analysis:")
        error_types = {}
        for fail in failed:
            error = fail.get('error', 'Unknown error')
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error}: {count} cases")
    
    # File sizes and timestamps
    print(f"\n📁 File Information:")
    
    files_info = [
        ("Riddles", RIDDLES_FILE),
        ("Semantic Tasks", OUTPUT_FILE),
        ("Progress", PROGRESS_FILE),
        ("Failed", FAILED_FILE)
    ]
    
    for name, filepath in files_info:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"   {name}: {size:,} bytes, modified {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   {name}: File not found")

def show_sample_tasks():
    """Show sample semantic ambiguity tasks."""
    
    semantic_tasks = load_json_file(OUTPUT_FILE)
    
    if not semantic_tasks:
        print("❌ No semantic tasks found.")
        return
    
    print("\n📋 Sample Semantic Ambiguity Tasks")
    print("=" * 40)
    
    # Show first few tasks
    for i, task in enumerate(semantic_tasks[:3], 1):
        print(f"\n{i}. Task ID: {task['id']}")
        print(f"   Riddle: {task['riddle'][:50]}...")
        print(f"   Ambiguous Word: '{task['ambiguous_word']}'")
        print(f"   Question: {task['question']}")
        print(f"   Options:")
        for j, option in enumerate(task['options'], 1):
            marker = "✓" if option == task['correct_option'] else " "
            print(f"     {marker} {j}. {option}")

def show_quality_metrics():
    """Analyze quality metrics of generated semantic tasks."""
    
    semantic_tasks = load_json_file(OUTPUT_FILE)
    
    if not semantic_tasks:
        print("❌ No semantic tasks found for quality analysis.")
        return
    
    print("\n📈 Quality Metrics")
    print("=" * 20)
    
    # Analyze question patterns
    question_patterns = {}
    ambiguous_word_lengths = []
    option_lengths = []
    
    for task in semantic_tasks:
        # Question patterns
        question = task.get('question', '')
        if 'refer to' in question.lower():
            question_patterns['refer to'] = question_patterns.get('refer to', 0) + 1
        elif 'mean' in question.lower():
            question_patterns['mean'] = question_patterns.get('mean', 0) + 1
        elif 'represent' in question.lower():
            question_patterns['represent'] = question_patterns.get('represent', 0) + 1
        else:
            question_patterns['other'] = question_patterns.get('other', 0) + 1
        
        # Word and option analysis
        ambiguous_word_lengths.append(len(task.get('ambiguous_word', '')))
        option_lengths.extend([len(opt) for opt in task.get('options', [])])
    
    print(f"Total tasks analyzed: {len(semantic_tasks)}")
    
    print(f"\nQuestion patterns:")
    for pattern, count in sorted(question_patterns.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(semantic_tasks)) * 100
        print(f"   '{pattern}': {count} ({percentage:.1f}%)")
    
    if ambiguous_word_lengths:
        avg_word_length = sum(ambiguous_word_lengths) / len(ambiguous_word_lengths)
        print(f"\nAmbiguous word length: {avg_word_length:.1f} chars average")
    
    if option_lengths:
        avg_option_length = sum(option_lengths) / len(option_lengths)
        print(f"Option length: {avg_option_length:.1f} chars average")

def main():
    """Main monitoring interface."""
    
    while True:
        print("\n📊 Semantic Ambiguity Progress Monitor")
        print("=" * 40)
        print("1. Show overall progress")
        print("2. Show sample tasks")
        print("3. Show quality metrics")
        print("4. Refresh (re-read files)")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            show_progress()
            
        elif choice == '2':
            show_sample_tasks()
            
        elif choice == '3':
            show_quality_metrics()
            
        elif choice == '4':
            print("🔄 Refreshing data...")
            # No caching, so this just continues
            
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()