#!/usr/bin/env python3
"""
CoT Progress Monitor
==================

Monitor the progress of Chain of Thought reasoning generation.
"""

import json
import os
import time
from datetime import datetime

# File paths
BASE_PATH = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada"
OUTPUT_FILE = f"{BASE_PATH}/dataset/riddles_reasoning.json"
PROGRESS_FILE = f"{BASE_PATH}/dataset/cot_progress.json"
FAILED_FILE = f"{BASE_PATH}/dataset/cot_failed.json"
RIDDLES_FILE = f"{BASE_PATH}/dataset/riddles.json"

def get_progress_status():
    """Get current progress status."""
    
    print("🧠 === CoT Reasoning Generation Progress ===")
    
    # Load total riddles count
    total_riddles = 0
    if os.path.exists(RIDDLES_FILE):
        with open(RIDDLES_FILE, 'r', encoding='utf-8') as f:
            riddles = json.load(f)
            total_riddles = len(riddles)
    
    print(f"📊 Total riddles in dataset: {total_riddles}")
    
    # Load completed reasoning
    completed_count = 0
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                completed_cots = json.load(f)
                completed_count = len(completed_cots)
                processed_ids = {cot['riddle_id'] for cot in completed_cots}
            print(f"✅ Completed CoT reasoning: {completed_count}")
        except Exception as e:
            print(f"❌ Error reading completed reasoning: {e}")
    else:
        print("📝 No completed reasoning file found")
    
    # Load failed riddles
    failed_count = 0
    failed_ids = set()
    if os.path.exists(FAILED_FILE):
        try:
            with open(FAILED_FILE, 'r', encoding='utf-8') as f:
                failed_data = json.load(f)
                failed_ids = set(failed_data.get('failed_riddles', []))
                failed_count = len(failed_ids)
            print(f"❌ Failed reasoning: {failed_count}")
        except Exception as e:
            print(f"⚠️  Error reading failed riddles: {e}")
    else:
        print("📝 No failed riddles file found")
    
    # Load progress file
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            
            print(f"⏰ Last updated: {progress.get('last_updated', 'Unknown')}")
            print(f"📈 Progress file shows:")
            print(f"   - Processed: {progress.get('processed_count', 0)}")
            print(f"   - Failed: {progress.get('failed_count', 0)}")
        except Exception as e:
            print(f"⚠️  Error reading progress file: {e}")
    else:
        print("📝 No progress file found")
    
    # Calculate statistics
    if total_riddles > 0:
        completion_rate = (completed_count / total_riddles) * 100
        failure_rate = (failed_count / total_riddles) * 100
        remaining = total_riddles - completed_count - failed_count
        
        print(f"\\n📊 Statistics:")
        print(f"   ✅ Completion rate: {completion_rate:.2f}%")
        print(f"   ❌ Failure rate: {failure_rate:.2f}%")
        print(f"   ⏳ Remaining to process: {remaining}")
        
        if completed_count > 0:
            print(f"\\n🎯 Next steps:")
            if remaining > 0:
                print(f"   - Continue processing from where you left off")
                print(f"   - {remaining} riddles still need CoT reasoning")
            if failed_count > 0:
                print(f"   - Review {failed_count} failed riddles for manual processing")
                print(f"   - Failed riddles are saved in: {FAILED_FILE}")
        
        # Show some sample processed IDs
        if processed_ids:
            sample_processed = sorted(list(processed_ids))[:10]
            print(f"\\n📋 Sample processed riddle IDs: {sample_processed}...")
        
        # Show some sample failed IDs
        if failed_ids:
            sample_failed = sorted(list(failed_ids))[:10]
            print(f"💥 Sample failed riddle IDs: {sample_failed}...")

def show_latest_reasoning():
    """Show the latest CoT reasoning entry."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                cot_data = json.load(f)
            
            if cot_data:
                latest = cot_data[-1]
                print(f"\\n🔍 Latest CoT Reasoning (Riddle {latest['riddle_id']}):")
                print(f"📝 Riddle: {latest['riddle']}")
                print(f"✅ Answer: {latest['ans']}")
                print(f"🧠 Reasoning steps: {len(latest['reasoning'])} steps completed")
                
                # Show brief analysis from each step
                reasoning = latest['reasoning']
                for step_name, step_data in reasoning.items():
                    if isinstance(step_data, dict) and 'analysis' in step_data:
                        analysis = step_data['analysis'][:100] + "..." if len(step_data['analysis']) > 100 else step_data['analysis']
                        print(f"   {step_name}: {analysis}")
            
        except Exception as e:
            print(f"❌ Error reading latest reasoning: {e}")

if __name__ == "__main__":
    get_progress_status()
    
    if input("\\nShow latest CoT reasoning entry? (y/n): ").lower() == 'y':
        show_latest_reasoning()