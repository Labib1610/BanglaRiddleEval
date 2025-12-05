#!/usr/bin/env python3
"""
Chain of Thought (CoT) Reasoning Generator for Bengali Riddles
=============================================================

This script converts riddles from riddles.json into detailed reasoning steps
using the Ollama gpt-oss:20b model for Bengali riddle analysis.

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
from typing import Dict, List, Optional, Any

# Configuration
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# File paths
BASE_PATH = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada"
RIDDLES_FILE = f"{BASE_PATH}/dataset/riddles.json"
OUTPUT_FILE = f"{BASE_PATH}/dataset/riddles_reasoning.json"
PROGRESS_FILE = f"{BASE_PATH}/dataset/cot_progress.json"
FAILED_FILE = f"{BASE_PATH}/dataset/cot_failed.json"

class CoTReasoningGenerator:
    """
    Generates Chain of Thought reasoning for Bengali riddles using Ollama gpt-oss:20b model.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "gpt-oss:20b"):
        """
        Initialize the CoT reasoning generator.
        
        Args:
            ollama_url: Base URL for Ollama API
            model: Model name to use for generation
        """
        self.ollama_url = ollama_url
        self.model = model
        self.session = requests.Session()
        self.timeout = 300  # 5 minutes timeout for complex reasoning
        self.processed_riddles = set()
        self.failed_riddles = set()
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing CoT data and progress."""
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_cots = json.load(f)
                    self.processed_riddles = {cot['riddle_id'] for cot in existing_cots}
                print(f"Loaded {len(self.processed_riddles)} existing CoT reasoning entries.")
            except Exception as e:
                print(f"Error loading existing CoT data: {e}")
        
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.failed_riddles = set(progress.get('failed_riddles', []))
                print(f"Loaded {len(self.failed_riddles)} previously failed riddles.")
            except Exception as e:
                print(f"Error loading progress: {e}")
    
    def save_progress(self):
        """Save current progress."""
        progress = {
            'processed_count': len(self.processed_riddles),
            'failed_riddles': list(self.failed_riddles),
            'failed_count': len(self.failed_riddles),
            'last_updated': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def save_failed_riddles(self):
        """Save failed riddles with details for manual review."""
        if self.failed_riddles:
            failed_data = {
                'failed_riddles': list(self.failed_riddles),
                'total_failed': len(self.failed_riddles),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'note': 'These riddles failed CoT generation and may need manual processing'
            }
            with open(FAILED_FILE, 'w', encoding='utf-8') as f:
                json.dump(failed_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.failed_riddles)} failed riddles to {FAILED_FILE}")
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            print(f"Testing connection to Ollama with model {self.model}...")
            print("This may take a moment if the model needs to load...")
            
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "Test",
                    "stream": False
                },
                timeout=120  # 2 minutes for model loading
            )
            
            if response.status_code == 200:
                print("✅ Connection successful!")
                return True
            else:
                print(f"❌ Connection failed with status: {response.status_code}")
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
    
    def create_cot_prompt(self, riddle: str, answer: str) -> str:
        """
        Create a detailed prompt for generating Chain of Thought reasoning.
        
        Args:
            riddle: The Bengali riddle text
            answer: The correct answer to the riddle
        
        Returns:
            Formatted prompt for CoT generation
        """
        prompt = f"""
আপনি একটি বাংলা ধাঁধার জন্য বিস্তারিত যুক্তি তৈরি করবেন।

ধাঁধা: "{riddle}"
উত্তর: "{answer}"

নিচের ৪টি ধাপে বিশ্লেষণ করুন এবং একটি সুন্দর বাংলা অনুচ্ছেদ আকারে লিখুন:

1. উত্তর চিহ্নিতকরণ: ধাঁধার নির্দিষ্ট শব্দগুলো উদ্ধৃত করুন
2. রূপকের ব্যাখ্যা: রূপকটি কী প্রতিনিধিত্ব করে তা ব্যাখ্যা করুন  
3. উত্তরের সাথে সংযোগ: "{answer}" এর কোন বৈশিষ্ট্য এই ধাঁধার সাথে মিলে যায় তা ব্যাখ্যা করুন
4. সিদ্ধান্ত: কেন এটাই একমাত্র যুক্তিসংগত উত্তর তা সংক্ষেপে বলুন

উদাহরণ ফরম্যাট:
১. 'এক থালা': এখানে আকাশকে একটি বিশাল থালার সাথে তুলনা করা হয়েছে।
২. 'সুপারি': সুপারি যেমন ছোট ছোট গোল হয়, আকাশের নক্ষত্রগুলোকেও দেখতে ছোট বিন্দুর মতো লাগে।
৩. 'গুনতে নারি': সুপারি গোনা সম্ভব হলেও, আকাশের তারা বা নক্ষত্র অসংখ্য, যা গুনে শেষ করা যায় না।
সিদ্ধান্ত: আকাশের বিশাল থালায় ছড়িয়ে থাকা অগণিত নক্ষত্রই হলো এই ধাঁধার উত্তর।

অনুগ্রহ করে শুধুমাত্র reasoning টেক্সট দিন। JSON বা অন্য কোনো ফরম্যাট ব্যবহার করবেন না।
"""
        return prompt

    def validate_reasoning_quality(self, reasoning: str) -> bool:
        """Validate if the generated CoT reasoning meets quality criteria."""
        
        # Check if reasoning is not empty
        if not reasoning or len(reasoning.strip()) < 50:
            print(f"    ❌ Reasoning too short")
            return False
        
        # Check for required Bengali numerals (১, ২, ৩, সিদ্ধান্ত)
        required_elements = ['১.', '২.', '৩.', 'সিদ্ধান্ত']
        missing_elements = []
        
        for element in required_elements:
            if element not in reasoning:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"    ❌ Missing required elements: {missing_elements}")
            return False
        
        # Check for failure indicators
        if "Failed to generate" in reasoning or "ব্যর্থ" in reasoning or "sorry" in reasoning.lower():
            print(f"    ❌ Found failure indicator in reasoning")
            return False
        
        # Check minimum length per step (rough estimate)
        if len(reasoning) < 200:
            print(f"    ❌ Reasoning seems too brief")
            return False
        
        print(f"    ✅ Reasoning quality validated")
        return True

    def generate_cot_reasoning(self, riddle: str, answer: str, attempt: int = 1) -> Optional[str]:
        """
        Generate Chain of Thought reasoning for a riddle using Ollama.
        
        Args:
            riddle: The riddle text
            answer: The correct answer
            attempt: Current attempt number for logging
        
        Returns:
            String containing Bengali reasoning or None if failed
        """
        prompt = self.create_cot_prompt(riddle, answer)
        
        try:
            # Send request to Ollama
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 2000
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Clean up the response text
                reasoning_text = response_text
                
                # Remove any JSON markers or extra formatting
                if reasoning_text.startswith('{') or reasoning_text.startswith('```'):
                    # Try to extract just the reasoning content
                    lines = reasoning_text.split('\n')
                    clean_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('{') and not line.startswith('}') and not line.startswith('```') and not line.startswith('"'):
                            clean_lines.append(line)
                    reasoning_text = '\n'.join(clean_lines)
                
                # Validate reasoning structure
                if self.validate_reasoning_quality(reasoning_text):
                    return reasoning_text
                else:
                    print(f"  ⚠️  Reasoning quality check failed (attempt {attempt})")
                    return None
                    
            else:
                print(f"  ⚠️  Ollama API error (attempt {attempt}): {response.status_code} - {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"  ⚠️  Timeout occurred (attempt {attempt}) for riddle: {riddle[:50]}...")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Request error (attempt {attempt}): {e}")
            return None

    def process_single_riddle(self, riddle_data: Dict) -> Optional[Dict]:
        """Process a single riddle into CoT reasoning format."""
        riddle_id = riddle_data['riddle_id']
        riddle = riddle_data['riddle']
        correct_answer = riddle_data['ans']
        
        if riddle_id in self.processed_riddles:
            print(f"Riddle {riddle_id} already processed. Skipping.")
            return None
        
        if riddle_id in self.failed_riddles:
            print(f"Riddle {riddle_id} previously failed. Retrying...")
        
        print(f"Processing riddle {riddle_id}: {riddle[:50]}...")
        
        # Generate CoT reasoning with retry mechanism
        reasoning_text = None
        max_retries = 3
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  🔄 Retry attempt {attempt + 1}/{max_retries}")
                time.sleep(5)  # Wait between retries
            
            reasoning_text = self.generate_cot_reasoning(riddle, correct_answer, attempt + 1)
            if reasoning_text:
                break
        
        if reasoning_text:
            cot_entry = {
                "riddle_id": riddle_id,
                "riddle": riddle,
                "ans": correct_answer,
                "reasoning": reasoning_text
            }
            
            print(f"✓ Generated CoT reasoning for riddle {riddle_id}")
            return cot_entry
        else:
            print(f"✗ Failed to generate CoT reasoning for riddle {riddle_id}")
            self.failed_riddles.add(riddle_id)
            return None

    def save_cot_reasoning(self, cot_data: Dict):
        """Save a single CoT reasoning to the JSON file."""
        # Load existing data
        existing_cots = []
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_cots = json.load(f)
            except Exception as e:
                print(f"Error loading existing CoT data: {e}")
        
        # Add new CoT reasoning
        existing_cots.append(cot_data)
        
        # Save updated data
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_cots, f, indent=2, ensure_ascii=False)
        
        # Update processed set
        self.processed_riddles.add(cot_data['riddle_id'])

    def process_all_riddles(self, start_id: int = 1, end_id: int = None):
        """Process all riddles in the given range with robust error handling."""
        
        # Check Ollama connection first
        if not self.check_ollama_connection():
            print("❌ Error: Cannot connect to Ollama or model not available.")
            print("Please make sure:")
            print("1. Ollama is running: ollama serve")
            print(f"2. Model is installed: ollama pull {self.model}")
            return
        
        print(f"✅ Connected to Ollama with model: {self.model}")
        
        # Load riddles
        try:
            with open(RIDDLES_FILE, 'r', encoding='utf-8') as f:
                riddles = json.load(f)
            print(f"Loaded {len(riddles)} riddles from dataset.")
        except Exception as e:
            print(f"Error loading riddles: {e}")
            return
        
        if end_id is None:
            end_id = len(riddles)
        
        print(f"Processing riddles from ID {start_id} to {end_id}")
        print(f"Output will be saved to: {OUTPUT_FILE}")
        
        successful = 0
        failed = 0
        skipped = 0
        
        try:
            for riddle_data in riddles:
                riddle_id = riddle_data['riddle_id']
                
                # Skip if outside range
                if riddle_id < start_id or riddle_id > end_id:
                    continue
                
                # Skip if already processed
                if riddle_id in self.processed_riddles:
                    skipped += 1
                    continue
                
                try:
                    cot_data = self.process_single_riddle(riddle_data)
                    
                    if cot_data:
                        self.save_cot_reasoning(cot_data)
                        successful += 1
                        print(f"📊 Progress: {successful} successful, {failed} failed, {skipped} skipped")
                    else:
                        failed += 1
                    
                    # Save progress every 10 riddles
                    if (successful + failed) % 10 == 0:
                        self.save_progress()
                        print(f"💾 Progress saved after {successful + failed} attempts")
                    
                    # Add delay between riddles to avoid overwhelming the model
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"❌ Unexpected error processing riddle {riddle_id}: {e}")
                    self.failed_riddles.add(riddle_id)
                    failed += 1
        
        except KeyboardInterrupt:
            print("\\n⏹️  Process interrupted by user")
        
        finally:
            # Final save
            self.save_progress()
            self.save_failed_riddles()
            
            print(f"\\n📈 Final Summary:")
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            print(f"   ⏭️  Skipped (already processed): {skipped}")
            print(f"   📊 Total processed: {len(self.processed_riddles)}")
            
            if self.failed_riddles:
                print(f"   ⚠️  Failed riddles saved to: {FAILED_FILE}")

    def generate_sample_reasoning(self, riddle: str = "এক থালা সুপারি, গুনতে নারি।", 
                                 answer: str = "তারা") -> Dict[str, Any]:
        """
        Generate a sample CoT reasoning for testing.
        
        Args:
            riddle: Sample riddle
            answer: Sample answer
            
        Returns:
            Sample reasoning structure
        """
        print("Generating sample CoT reasoning...")
        
        # Check connection first
        if not self.check_ollama_connection():
            print("❌ Cannot connect to Ollama for sample generation")
            return {}
        
        reasoning_text = self.generate_cot_reasoning(riddle, answer)
        
        if reasoning_text:
            sample_entry = {
                "riddle_id": 1,
                "riddle": riddle,
                "ans": answer,
                "reasoning": reasoning_text
            }
            print("Sample reasoning generated successfully!")
            return sample_entry
        else:
            print("Failed to generate sample reasoning")
            return {}


def main():
    """Main function for CoT reasoning generation."""
    
    # Initialize the generator
    generator = CoTReasoningGenerator()
    
    print("=== CoT Reasoning Generator for Bengali Riddles ===")
    print("Choose an option:")
    print("1. Generate sample reasoning (test)")
    print("2. Process all riddles (with robust error handling)")
    print("3. Process riddles in specific range (resume capability)")
    print("4. Check progress and statistics")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        # Generate sample reasoning
        sample = generator.generate_sample_reasoning()
        if sample:
            print("\\n=== Sample CoT Reasoning ===")
            print(json.dumps(sample, ensure_ascii=False, indent=2))
            
            # Save sample
            sample_file = f"{BASE_PATH}/Code/cot maker/sample_reasoning.json"
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump([sample], f, ensure_ascii=False, indent=2)
            print(f"\\nSample saved to: {sample_file}")
    
    elif choice == "2":
        # Process all riddles
        print(f"Processing all riddles with robust error handling...")
        confirmation = input("This will take a very long time. Continue? (y/n): ")
        if confirmation.lower() != 'y':
            return
            
        generator.process_all_riddles()
    
    elif choice == "3":
        # Process in specific range
        start_id = int(input("Enter starting riddle ID (default 1): ") or "1")
        end_id = int(input("Enter ending riddle ID (default 1244): ") or "1244")
        
        if start_id < 1 or end_id > 1244 or start_id > end_id:
            print("Invalid range! Must be between 1 and 1244.")
            return
            
        print(f"Processing riddles from ID {start_id} to {end_id} with resume capability...")
        generator.process_all_riddles(start_id, end_id)
    
    elif choice == "4":
        # Check progress and statistics
        print("\\n=== Current Progress ===")
        print(f"Processed riddles: {len(generator.processed_riddles)}")
        print(f"Failed riddles: {len(generator.failed_riddles)}")
        
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"Last updated: {progress.get('last_updated', 'Unknown')}")
        
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"Total CoT entries saved: {len(existing_data)}")
        
        if generator.failed_riddles:
            print(f"\\nFailed riddle IDs: {sorted(list(generator.failed_riddles))[:10]}..." if len(generator.failed_riddles) > 10 else f"\\nFailed riddle IDs: {sorted(list(generator.failed_riddles))}")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
