#!/usr/bin/env python3
"""
Semantic Ambiguity Task Generation Runner
========================================

This script provides a command-line interface to run semantic ambiguity task generation
with various options for processing ranges and modes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_ambiguity import SemanticAmbiguityGenerator

def main():
    generator = SemanticAmbiguityGenerator()
    
    print("🎯 Bengali Riddle Semantic Ambiguity Task Generator")
    print("=" * 55)
    print()
    
    while True:
        print("Choose an option:")
        print("1. Process sample riddles (first 5)")
        print("2. Process all riddles")
        print("3. Process specific range")
        print("4. Show current status")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\n🔄 Processing sample riddles (first 5)...")
            generator.process_all_riddles(start_idx=0, end_idx=5)
            
        elif choice == '2':
            confirm = input("\n⚠️  This will process ALL riddles. Continue? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                print("\n🔄 Processing all riddles...")
                generator.process_all_riddles()
            else:
                print("❌ Cancelled.")
                
        elif choice == '3':
            try:
                start = int(input("Enter start index (0-based): "))
                end = int(input("Enter end index (exclusive): "))
                print(f"\n🔄 Processing riddles {start} to {end-1}...")
                generator.process_all_riddles(start_idx=start, end_idx=end)
            except ValueError:
                print("❌ Invalid input. Please enter valid numbers.")
                
        elif choice == '4':
            print(f"\n📊 Current Status:")
            print(f"✅ Processed riddles: {len(generator.processed_riddles)}")
            print(f"❌ Failed riddles: {len(generator.failed_riddles)}")
            
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-5.")
        
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()