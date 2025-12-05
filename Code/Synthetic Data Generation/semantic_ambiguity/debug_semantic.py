#!/usr/bin/env python3
"""
Debug and Test Semantic Ambiguity Task Generation
================================================

This script helps debug and test the semantic ambiguity task generation process
by processing individual riddles and showing detailed output.
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from semantic_ambiguity import SemanticAmbiguityGenerator

def test_single_riddle():
    """Test semantic ambiguity generation on a single example riddle."""
    
    # Example riddle from the dataset
    test_riddle = {
        "riddle_id": 1,
        "riddle": "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।",
        "ans": "আনারস"
    }
    
    print("🧪 Testing Semantic Ambiguity Task Generation")
    print("=" * 50)
    print(f"Test Riddle: {test_riddle['riddle']}")
    print(f"Answer: {test_riddle['ans']}")
    print()
    
    generator = SemanticAmbiguityGenerator()
    
    # Check connection first
    if not generator.check_ollama_connection():
        print("❌ Cannot test without Ollama connection.")
        return
    
    # Generate semantic task
    print("🔄 Generating semantic ambiguity task...")
    task_data = generator.generate_semantic_task(test_riddle['riddle'], test_riddle['ans'])
    
    if task_data:
        print("\n✅ Semantic Ambiguity Task Generated Successfully!")
        print("-" * 40)
        print(f"Ambiguous Word: {task_data['ambiguous_word']}")
        print(f"Question: {task_data['question']}")
        print("\nOptions:")
        for i, option in enumerate(task_data['options'], 1):
            marker = "✓" if option == task_data['correct_option'] else " "
            print(f"  {marker} {i}. {option}")
        print(f"\nCorrect Answer: {task_data['correct_option']}")
        
        # Create full task structure
        full_task = {
            'id': test_riddle['riddle_id'],
            'riddle': test_riddle['riddle'],
            'ambiguous_word': task_data['ambiguous_word'],
            'question': task_data['question'],
            'options': task_data['options'],
            'correct_option': task_data['correct_option']
        }
        
        print("\n📋 Full Task JSON:")
        print(json.dumps(full_task, indent=2, ensure_ascii=False))
    else:
        print("\n❌ Failed to generate semantic ambiguity task")

def test_quality_validation():
    """Test the quality validation function."""
    
    print("\n🧪 Testing Quality Validation")
    print("-" * 30)
    
    generator = SemanticAmbiguityGenerator()
    
    # Test case 1: Valid task
    print("Test 1: Valid task")
    valid = generator.validate_task_quality(
        question="এই ধাঁধায় 'টিয়ে' শব্দটি কী বোঝায়?",
        options=["ফলের সবুজ পাতা", "সত্যিকারের পাখি", "টিয়ে নামের কোন ব্যক্তি", "সোনার মুকুট"],
        correct_option="ফলের সবুজ পাতা",
        riddle="বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।"
    )
    print(f"Result: {'✅ PASS' if valid else '❌ FAIL'}\n")
    
    # Test case 2: Duplicate options
    print("Test 2: Duplicate options")
    invalid = generator.validate_task_quality(
        question="What does the word 'test' refer to in this riddle?",
        options=["Option A", "Option B", "Option A", "Option C"],
        correct_option="Option A",
        riddle="Test riddle"
    )
    print(f"Result: {'❌ FAIL (Expected)' if not invalid else '✅ Unexpected PASS'}\n")
    
    # Test case 3: Wrong number of options
    print("Test 3: Wrong number of options")
    invalid2 = generator.validate_task_quality(
        question="What does the word 'test' refer to in this riddle?",
        options=["Option A", "Option B", "Option C"],
        correct_option="Option A", 
        riddle="Test riddle"
    )
    print(f"Result: {'❌ FAIL (Expected)' if not invalid2 else '✅ Unexpected PASS'}\n")

def show_example_format():
    """Show the expected output format with an example."""
    
    print("📋 Expected Semantic Ambiguity Task Format")
    print("=" * 45)
    
    example = {
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
    
    print(json.dumps(example, indent=2, ensure_ascii=False))
    print()
    print("Key Components:")
    print("• id: Original riddle ID")
    print("• riddle: Original Bengali riddle text")
    print("• ambiguous_word: The word/phrase that has semantic ambiguity")
    print("• question: Bengali question asking what the ambiguous word refers to")
    print("• options: 4 possible Bengali answers (1 correct, 3 plausible distractors)")
    print("• correct_option: The semantically correct interpretation in Bengali")

def main():
    """Main debug interface."""
    
    while True:
        print("\n🐛 Semantic Ambiguity Debug Menu")
        print("=" * 35)
        print("1. Test single riddle generation")
        print("2. Test quality validation") 
        print("3. Show expected format")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            test_single_riddle()
            
        elif choice == '2':
            test_quality_validation()
            
        elif choice == '3':
            show_example_format()
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()