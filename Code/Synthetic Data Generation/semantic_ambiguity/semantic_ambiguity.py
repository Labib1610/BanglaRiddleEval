#!/usr/bin/env python3
"""
Semantic Ambiguity Task Generator for Bengali Riddles
===================================================

This script creates semantic ambiguity tasks from riddles.json, identifying
ambiguous words/metaphors in riddles and generating multiple choice questions
about what those words refer to in the context of the riddle.

Features:
- Robust error handling with retry mechanism
- Progress tracking and resume capability
- Failed riddles tracking for manual review
- Time-stamped saves and comprehensive logging

Author: AI Assistant
Date: November 2025
"""

import json
import requests
import time
import os
import random
from typing import Dict, List, Optional, Any

# Configuration
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# File paths
BASE_PATH = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada"
RIDDLES_FILE = f"{BASE_PATH}/dataset/riddles.json"
OUTPUT_FILE = f"{BASE_PATH}/dataset/riddles_semantic_ambiguity.json"
PROGRESS_FILE = f"{BASE_PATH}/dataset/semantic_progress.json"
FAILED_FILE = f"{BASE_PATH}/dataset/semantic_failed.json"

class SemanticAmbiguityGenerator:
    """
    Generates semantic ambiguity tasks for Bengali riddles using Ollama gpt-oss:20b model.
    """
    
    def __init__(self):
        self.processed_riddles = set()
        self.failed_riddles = []
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing semantic ambiguity data and progress."""
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_tasks = json.load(f)
                    self.processed_riddles = {task['id'] for task in existing_tasks}
                print(f"📊 Loaded {len(self.processed_riddles)} existing semantic ambiguity tasks.")
            except Exception as e:
                print(f"❌ Error loading existing tasks: {e}")
        
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    if 'failed_riddles' in progress:
                        self.failed_riddles = progress['failed_riddles']
                print(f"📊 Loaded progress with {len(self.failed_riddles)} previously failed riddles.")
            except Exception as e:
                print(f"❌ Error loading progress: {e}")
    
    def save_progress(self):
        """Save current progress."""
        progress = {
            'processed_count': len(self.processed_riddles),
            'failed_riddles': self.failed_riddles,
            'failed_count': len(self.failed_riddles),
            'last_updated': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def save_failed_riddle(self, riddle_data: Dict[str, Any], error_message: str):
        """Save failed riddle for later analysis."""
        failed_entry = {
            'riddle_id': riddle_data.get('riddle_id'),
            'riddle': riddle_data.get('riddle'),
            'ans': riddle_data.get('ans'),
            'error': error_message,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Load existing failed riddles
        failed_riddles = []
        if os.path.exists(FAILED_FILE):
            try:
                with open(FAILED_FILE, 'r', encoding='utf-8') as f:
                    failed_riddles = json.load(f)
            except Exception:
                pass
        
        failed_riddles.append(failed_entry)
        
        with open(FAILED_FILE, 'w', encoding='utf-8') as f:
            json.dump(failed_riddles, f, indent=2, ensure_ascii=False)
    
    def validate_task_quality(self, question: str, options: List[str], correct_option: str, riddle: str) -> bool:
        """Validate if the generated semantic ambiguity task meets quality criteria."""
        
        # Check for duplicate options
        if len(set(options)) != len(options):
            print(f"    ❌ Duplicate options found")
            return False
        
        # Check if correct option is in the options list
        if correct_option not in options:
            print(f"    ❌ Correct option '{correct_option}' not found in options")
            return False
        
        # Check if we have exactly 4 options
        if len(options) != 4:
            print(f"    ❌ Expected 4 options, got {len(options)}")
            return False
        
        # Check if question is asking about semantic meaning (in Bengali)
        if "বোঝায়" not in question and "অর্থ" not in question and "নির্দেশ করে" not in question:
            print(f"    ⚠️  Question might not be about semantic meaning")
        
        # Check if options are too similar (simple string similarity)
        for i, opt1 in enumerate(options):
            for j, opt2 in enumerate(options[i+1:], i+1):
                if opt1.lower().strip() == opt2.lower().strip():
                    print(f"    ❌ Identical options: '{opt1}' and '{opt2}'")
                    return False
        
        print(f"    ✅ Task validation passed")
        return True
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        print("🔍 Checking Ollama connection...")
        
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": "Test",
                "stream": False,
                "options": {"max_tokens": 1}
            }, timeout=120)
            
            if response.status_code == 200:
                print("✅ Ollama connection successful!")
                return True
            else:
                print(f"❌ Ollama connection failed with status: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"❌ Connection timeout. The model might be loading - try again in a moment.")
            return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama. Make sure 'ollama serve' is running.")
            return False
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
            return False
    
    def generate_semantic_task(self, riddle: str, correct_answer: str) -> Optional[Dict[str, Any]]:
        """Generate semantic ambiguity task using Ollama model."""
        
        prompt = f"""আপনি একজন বিশেষজ্ঞ বাংলা ভাষাবিদ এবং ধাঁধা বিশ্লেষক। আপনার কাজ হল বাংলা ধাঁধায় থাকা রূপক বা দ্ব্যর্থক শব্দ খুঁজে বের করে সেগুলো নিয়ে semantic ambiguity (অর্থগত অস্পষ্টতা) প্রশ্ন তৈরি করা।

ধাঁধা: "{riddle}"
সঠিক উত্তর: "{correct_answer}"

**কাজের ধাপ:**

১. ধাঁধায় রূপক বা দ্ব্যর্থক শব্দ চিহ্নিত করুন (যেমন: টিয়ে, সোনার টোপর, বেরুল ইত্যাদি)
২. **গুরুত্বপূর্ণ**: দ্ব্যর্থক শব্দটি অবশ্যই ধাঁধার মধ্যে থেকে নিতে হবে, উত্তর "{correct_answer}" থেকে নয়
৩. সবচেয়ে আকর্ষণীয় এবং দ্ব্যর্থক একটি শব্দ/বাক্যাংশ বেছে নিন যা ধাঁধার টেক্সটে আছে
৪. সেই শব্দটি ধাঁধার প্রেক্ষিতে আসলে কী বোঝায় তা নিয়ে প্রশ্ন তৈরি করুন
৫. ৪টি বিকল্প উত্তর দিন যার মধ্যে ১টি সঠিক এবং ৩টি যুক্তিসংগত কিন্তু ভুল

**উদাহরণ বিশ্লেষণ:**
ধাঁধা: "বন থেকে বেরুল টিয়ে সোনার টোপর মাথায় দিয়ে।"
উত্তর: "আনারস"

দ্ব্যর্থক শব্দ বিশ্লেষণ:
- "টিয়ে" = আক্ষরিক অর্থে পাখি, কিন্তু আনারসের সবুজ পাতাগুলো বোঝাচ্ছে
- "সোনার টোপর" = আনারসের হলুদ রঙের ফলের অংশ
- "বন থেকে বেরুল" = বাগান থেকে পাকা হয়ে বেরোনো

প্রশ্ন: "এই ধাঁধায় 'টিয়ে' শব্দটি কী বোঝায়?"
সঠিক উত্তর: "ফলের সবুজ পাতা"
বিভ্রান্তিকর বিকল্প:
- "সত্যিকারের পাখি" (আক্ষরিক অর্থ)
- "টিয়ে নামের কোন ব্যক্তি" (নামের সাদৃশ্য)
- "সোনার মুকুট" (অন্য রূপকের সাথে মিশ্রণ)

**গুরুত্বপূর্ণ নির্দেশনা:**
- দ্ব্যর্থক শব্দটি অবশ্যই ধাঁধার মধ্যে থাকা কোন শব্দ/বাক্যাংশ হতে হবে
- উত্তর "{correct_answer}" কে দ্ব্যর্থক শব্দ হিসেবে ব্যবহার করবেন না
- প্রশ্ন এবং সব বিকল্প উত্তর বাংলায় হবে
- প্রশ্নটি "এই ধাঁধায় '___' শব্দটি কী বোঝায়?" বা "এই ধাঁধায় '___' বলতে কী বোঝানো হয়েছে?" ফরম্যাটে
- দ্ব্যর্থক শব্দটি বাংলায় রাখুন এবং ধাঁধার টেক্সট থেকে হুবহু নিন
- বিকল্প উত্তরগুলো সহজ বাংলায় হবে
- সঠিক উত্তরটি রূপকের আসল অর্থ হতে হবে
- ভুল বিকল্পগুলো যুক্তিসংগত কিন্তু ভ্রান্ত হতে হবে

**প্রত্যাশিত JSON আউটপুট:**
{{
  "ambiguous_word": "ধাঁধার মধ্যে থাকা দ্ব্যর্থক শব্দ/বাক্যাংশ (উত্তর '{correct_answer}' নয়)",
  "question": "এই ধাঁধায় '____' শব্দটি কী বোঝায়?",
  "options": [
    "সঠিক বাংলা উত্তর",
    "যুক্তিসংগত ভুল বাংলা উত্তর ১",
    "যুক্তিসংগত ভুল বাংলা উত্তর ২", 
    "যুক্তিসংগত ভুল বাংলা উত্তর ৩"
  ],
  "correct_option": "সঠিক বাংলা উত্তর"
}}

**মনে রাখবেন:** ambiguous_word ফিল্ডে শুধুমাত্র ধাঁধার টেক্সট থেকে কোন শব্দ/বাক্যাংশ ব্যবহার করুন, উত্তর "{correct_answer}" ব্যবহার করবেন না।

এখন উপরের ধাঁধার জন্য semantic ambiguity task তৈরি করুন।"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"    🤖 Generating semantic task (attempt {attempt + 1}/{max_retries})...")
                
                response = requests.post(OLLAMA_URL, json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                }, timeout=300)  # 5 minutes timeout for semantic analysis
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '')
                    
                    # Extract JSON from response
                    try:
                        # Find JSON in the response
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = response_text[start_idx:end_idx]
                            parsed_json = json.loads(json_str)
                            
                            ambiguous_word = parsed_json.get('ambiguous_word', '')
                            question = parsed_json.get('question', '')
                            options = parsed_json.get('options', [])
                            correct_option = parsed_json.get('correct_option', '')
                            
                            # Validate the generated task
                            if (ambiguous_word and question and len(options) == 4 and 
                                correct_option and correct_option in options):
                                
                                # Check that ambiguous_word is from riddle, not the answer
                                if ambiguous_word.strip().lower() == correct_answer.strip().lower():
                                    print(f"    ❌ Ambiguous word '{ambiguous_word}' is the same as answer '{correct_answer}' (attempt {attempt + 1})")
                                    continue
                                
                                # Check that ambiguous_word appears in the riddle
                                if ambiguous_word not in riddle:
                                    print(f"    ❌ Ambiguous word '{ambiguous_word}' not found in riddle text (attempt {attempt + 1})")
                                    continue
                                
                                # Randomize the order of options
                                random.shuffle(options)
                                
                                # Additional quality validation
                                if self.validate_task_quality(question, options, correct_option, riddle):
                                    correct_index = options.index(correct_option)
                                    print(f"    ✅ Generated semantic task for ambiguous word: '{ambiguous_word}' (correct at index {correct_index})")
                                    return {
                                        'ambiguous_word': ambiguous_word,
                                        'question': question,
                                        'options': options,
                                        'correct_option': correct_option
                                    }
                                else:
                                    print(f"    ❌ Task failed quality validation (attempt {attempt + 1})")
                            else:
                                print(f"    ❌ Incomplete semantic task generated (attempt {attempt + 1})")
                                print(f"         Word: {bool(ambiguous_word)}, Question: {bool(question)}")
                                print(f"         Options: {len(options)}/4, Correct: {bool(correct_option)}")
                        else:
                            print(f"    ❌ No JSON found in response (attempt {attempt + 1})")
                    
                    except json.JSONDecodeError as e:
                        print(f"    ❌ JSON parsing error (attempt {attempt + 1}): {e}")
                else:
                    print(f"    ❌ API error {response.status_code} (attempt {attempt + 1})")
                
                # Wait before retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"    ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            
            except Exception as e:
                print(f"    ❌ Exception during generation (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None
    
    def process_riddle(self, riddle_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single riddle to generate semantic ambiguity task."""
        riddle_id = riddle_data['riddle_id']
        riddle = riddle_data['riddle']
        answer = riddle_data['ans']
        
        print(f"\n🔍 Processing riddle {riddle_id}: {riddle[:50]}...")
        
        # Generate semantic ambiguity task
        task_data = self.generate_semantic_task(riddle, answer)
        
        if task_data:
            # Create final semantic ambiguity task
            semantic_task = {
                'id': riddle_id,
                'riddle': riddle,
                'ans': answer,
                'ambiguous_word': task_data['ambiguous_word'],
                'question': task_data['question'],
                'options': task_data['options'],
                'correct_option': task_data['correct_option']
            }
            
            print(f"✅ Successfully created semantic task for riddle {riddle_id}")
            return semantic_task
        else:
            error_msg = "Failed to generate semantic task after multiple attempts"
            print(f"❌ {error_msg}")
            self.failed_riddles.append(riddle_id)
            self.save_failed_riddle(riddle_data, error_msg)
            return None
    
    def save_semantic_tasks(self, semantic_tasks: List[Dict[str, Any]]):
        """Save semantic ambiguity tasks to output file."""
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(semantic_tasks, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(semantic_tasks)} semantic tasks to {OUTPUT_FILE}")
    
    def process_all_riddles(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Process all riddles to generate semantic ambiguity tasks."""
        
        # Check Ollama connection first
        if not self.check_ollama_connection():
            print("❌ Cannot proceed without Ollama connection. Please start Ollama and try again.")
            return
        
        # Load riddles
        print(f"📚 Loading riddles from {RIDDLES_FILE}...")
        try:
            with open(RIDDLES_FILE, 'r', encoding='utf-8') as f:
                all_riddles = json.load(f)
        except Exception as e:
            print(f"❌ Error loading riddles: {e}")
            return
        
        # Determine range
        total_riddles = len(all_riddles)
        end_idx = end_idx if end_idx is not None else total_riddles
        riddles_to_process = all_riddles[start_idx:end_idx]
        
        print(f"📊 Total riddles in dataset: {total_riddles}")
        print(f"📊 Previously processed: {len(self.processed_riddles)}")
        print(f"📊 Range to process: {start_idx} to {end_idx-1} ({len(riddles_to_process)} riddles)")
        
        # Load existing semantic tasks
        existing_tasks = []
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_tasks = json.load(f)
            except Exception:
                existing_tasks = []
        
        # Process riddles
        processed_count = 0
        failed_count = 0
        
        for i, riddle_data in enumerate(riddles_to_process):
            riddle_id = riddle_data['riddle_id']
            
            # Skip if already processed
            if riddle_id in self.processed_riddles:
                print(f"⏭️  Skipping already processed riddle {riddle_id}")
                continue
            
            try:
                semantic_task = self.process_riddle(riddle_data)
                
                if semantic_task:
                    existing_tasks.append(semantic_task)
                    self.processed_riddles.add(riddle_id)
                    processed_count += 1
                    
                    # Save progress every 10 riddles
                    if processed_count % 10 == 0:
                        self.save_semantic_tasks(existing_tasks)
                        self.save_progress()
                        print(f"💾 Progress saved: {processed_count} processed, {failed_count} failed")
                else:
                    failed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing riddle {riddle_id}: {e}")
                failed_count += 1
                self.failed_riddles.append(riddle_id)
                self.save_failed_riddle(riddle_data, str(e))
        
        # Final save
        if existing_tasks:
            self.save_semantic_tasks(existing_tasks)
        self.save_progress()
        
        print(f"\n🎉 Semantic ambiguity task generation completed!")
        print(f"✅ Successfully processed: {processed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📁 Output saved to: {OUTPUT_FILE}")
        if failed_count > 0:
            print(f"📁 Failed riddles logged to: {FAILED_FILE}")

def main():
    """Main function to run semantic ambiguity task generation."""
    generator = SemanticAmbiguityGenerator()
    
    print("🎯 Bengali Riddle Semantic Ambiguity Task Generator")
    print("=" * 50)
    
    # For testing, process first 5 riddles
    generator.process_all_riddles(start_idx=0, end_idx=5)

if __name__ == "__main__":
    main()
