import json
import requests
import time
import os
from typing import List, Dict, Optional

# Configuration
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# File paths
BASE_PATH = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada"
RIDDLES_FILE = f"{BASE_PATH}/dataset/riddles.json"
OUTPUT_FILE = f"{BASE_PATH}/dataset/riddles_mcq.json"
PROGRESS_FILE = f"{BASE_PATH}/dataset/mcq_progress.json"

class MCQGenerator:
    def __init__(self):
        self.processed_riddles = set()
        self.failed_riddles = set()
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing MCQ data and progress."""
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_mcqs = json.load(f)
                    self.processed_riddles = {mcq['id'] for mcq in existing_mcqs}
                print(f"Loaded {len(self.processed_riddles)} existing MCQs.")
            except Exception as e:
                print(f"Error loading existing MCQs: {e}")
        
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
    
    def validate_mcq_quality(self, options: List[str], correct_answer: str, riddle: str) -> bool:
        """Validate if the generated MCQ options meet quality criteria."""
        
        # Check for duplicate options
        if len(set(options)) != len(options):
            print(f"    ❌ Duplicate options found")
            return False
        
        # Check if options are too similar (simple string similarity)
        for i, opt1 in enumerate(options):
            for j, opt2 in enumerate(options[i+1:], i+1):
                # Calculate simple similarity (common words ratio)
                words1 = set(opt1.split())
                words2 = set(opt2.split())
                if len(words1) > 0 and len(words2) > 0:
                    common_words = len(words1.intersection(words2))
                    total_words = max(len(words1), len(words2))
                    similarity = common_words / total_words
                    
                    if similarity > 0.7:  # Too similar
                        print(f"    ❌ Options too similar: '{opt1}' and '{opt2}'")
                        return False
        
        # Check if all options are single words (for better quality)
        word_counts = [len(option.split()) for option in options]
        if max(word_counts) > 3:  # Allow up to 3 words
            print(f"    ⚠️  Some options too long")
        
        # Check if options are reasonably different lengths
        lengths = [len(option) for option in options]
        if max(lengths) - min(lengths) > 20:  # Too much variation in length
            print(f"    ⚠️  Options have very different lengths")
        
        print(f"    ✅ Options quality validated")
        return True
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            print(f"Testing connection to Ollama with model {OLLAMA_MODEL}...")
            print("This may take a moment if the model needs to load...")
            
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": "Test",
                "stream": False
            }, timeout=120)  # Increased timeout to 2 minutes for model loading
            
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
    
    def generate_mcq_options(self, riddle: str, correct_answer: str) -> Optional[List[str]]:
        """Generate MCQ options using Ollama model with advanced confusing strategies."""
        
        prompt = f"""আপনি একজন বিশেষজ্ঞ বাংলা ধাঁধা প্রশ্ন প্রস্তুতকারী। আপনাকে ৩টি ভুল উত্তর তৈরি করতে হবে এবং ১টি সঠিক উত্তর সহ মোট ৪টি বিকল্প দিতে হবে।

ধাঁধা: {riddle}
সঠিক উত্তর: {correct_answer}

**অত্যন্ত গুরুত্বপূর্ণ**: আপনার তৈরি ৪টি বিকল্পের মধ্যে "{correct_answer}" অবশ্যই থাকতে হবে! এটি ছাড়া কাজ অসম্পূর্ণ।

কাজ: শুধুমাত্র ৩টি চ্যালেঞ্জিং ভুল উত্তর তৈরি করুন, সঠিক উত্তর "{correct_answer}" ইতিমধ্যে দেওয়া আছে।

বিশেষ নির্দেশনা - ভুল উত্তর তৈরির কৌশল:
1. সাদৃশ্যমূলক বিভ্রান্তি: এমন বস্তু বেছে নিন যা ধাঁধার কিছু বৈশিষ্ট্য ভাগাভাগি করে
2. আক্ষরিক ব্যাখ্যা: ধাঁধার শব্দগুলোর সরাসরি অর্থে মিলে এমন বস্তু
3. গভীর চিন্তায় বিভ্রান্তি: যা গভীরভাবে চিন্তা করলে সঠিক মনে হতে পারে
4. বিভাগীয় সাদৃশ্য: একই শ্রেণীর কিন্তু ভিন্ন বস্তু

উদাহরণ বিশ্লেষণ:
ধাঁধা: "হাত আছে, পা নেই, বুক তার ফাটা, মানুষকে গিলে খায়, নাই তার মাথা।"
সঠিক উত্তর: "শার্ট"
বিভ্রান্তিকর বিকল্প ৩টি:
- "জ্যাকেট" (অনুরূপ পোশাক কিন্তু বুক ফাটে না)
- "বই" (মানুষকে গিলে খায় মানসিকভাবে, পাতা হাতের মতো)
- "ব্যাগ" (মানুষকে ধারণ করে, হাতল আছে)
সম্পূর্ণ বিকল্প: ["শার্ট", "জ্যাকেট", "বই", "ব্যাগ"]

ধাঁধা: "কোন জিনিস কাটলে বাড়ে?"
সঠিক উত্তর: "পুকুর"
বিভ্রান্তিকর বিকল্প ৩টি:
- "নখ" (কাটলে বাড়ে কিন্তু ভিন্ন অর্থে)
- "চুল" (কাটলে দ্রুত বৃদ্ধি পায়)
- "গাছের ডাল" (প্রুনিং করলে নতুন ডাল গজায়)
সম্পূর্ণ বিকল্প: ["পুকুর", "নখ", "চুল", "গাছের ডাল"]

আবশ্যক শর্ত:
- অবশ্যই সঠিক উত্তর "{correct_answer}" ৪টি বিকল্পের একটি হতে হবে
- প্রতিটি ভুল উত্তর যেন কোন না কোনভাবে যুক্তিসংগত মনে হয়
- উত্তরগুলো খুব সহজ বা খুব কঠিন নয়, মধ্যম চ্যালেঞ্জিং হতে হবে
- গভীর বিশ্লেষণ ছাড়া সঠিক উত্তর খুঁজে পাওয়া কঠিন হতে হবে

JSON ফরম্যাট:
{{
  "options": ["বিকল্প ১", "বিকল্প ২", "বিকল্প ৩", "বিকল্প ৪"]
}}"""

        try:
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }, timeout=180)  # Increased to 3 minutes for large model responses
            
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
                        
                        options = parsed_json.get('options', [])
                        
                        # Ensure we have exactly 4 options
                        if len(options) == 4:
                            # If correct answer is missing, replace one random wrong option
                            if correct_answer not in options:
                                print(f"  🔧 Correct answer missing, fixing options...")
                                # Replace last option with correct answer
                                options[-1] = correct_answer
                                print(f"  ✅ Added correct answer: {correct_answer}")
                            
                            # Validate option quality
                            if self.validate_mcq_quality(options, correct_answer, riddle):
                                return options
                            else:
                                print(f"  ⚠️  Options quality check failed - regenerating...")
                                return None
                        elif len(options) == 3:
                            # Model gave 3 wrong options, add the correct one
                            print(f"  🔧 Adding correct answer to 3 generated options...")
                            options.append(correct_answer)
                            
                            # Validate option quality
                            if self.validate_mcq_quality(options, correct_answer, riddle):
                                return options
                            else:
                                print(f"  ⚠️  Options quality check failed - regenerating...")
                                return None
                        else:
                            print(f"  ⚠️  Invalid options count: {len(options)} (expected 3 or 4)")
                            return None
                    else:
                        print(f"  ⚠️  No valid JSON found in response")
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  JSON decode error: {e}")
                    return None
            else:
                print(f"  ⚠️  Ollama request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  ⚠️  Error generating MCQ options: {e}")
            return None
    
    def process_single_riddle(self, riddle_data: Dict) -> Optional[Dict]:
        """Process a single riddle into MCQ format."""
        riddle_id = riddle_data['riddle_id']
        riddle = riddle_data['riddle']
        correct_answer = riddle_data['ans']
        
        if riddle_id in self.processed_riddles:
            print(f"Riddle {riddle_id} already processed. Skipping.")
            return None
        
        if riddle_id in self.failed_riddles:
            print(f"Riddle {riddle_id} previously failed. Skipping.")
            return None
        
        print(f"Processing riddle {riddle_id}: {riddle[:50]}...")
        
        # Generate MCQ options using Ollama with retry for quality
        options = None
        max_retries = 3
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  🔄 Retry attempt {attempt + 1}/{max_retries}")
                time.sleep(3)  # Wait a bit between retries
            
            options = self.generate_mcq_options(riddle, correct_answer)
            if options:
                break
        
        if options:
            mcq_entry = {
                "id": riddle_id,
                "question": riddle,
                "options": options,
                "correct_answer": correct_answer
            }
            
            print(f"✓ Generated MCQ for riddle {riddle_id}")
            print(f"  Options: {options}")
            return mcq_entry
        else:
            print(f"✗ Failed to generate MCQ for riddle {riddle_id}")
            self.failed_riddles.add(riddle_id)
            return None
    
    def save_mcq(self, mcq_data: Dict):
        """Save a single MCQ to the JSON file."""
        # Load existing data
        existing_mcqs = []
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing_mcqs = json.load(f)
            except Exception as e:
                print(f"Error loading existing MCQs: {e}")
        
        # Add new MCQ
        existing_mcqs.append(mcq_data)
        
        # Save updated data
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_mcqs, f, indent=2, ensure_ascii=False)
    
    def process_all_riddles(self, start_id: int = 1, end_id: int = None):
        """Process all riddles in the given range."""
        
        # Check Ollama connection first
        if not self.check_ollama_connection():
            print("❌ Error: Cannot connect to Ollama or model not available.")
            print("Please make sure:")
            print("1. Ollama is running: ollama serve")
            print(f"2. Model is installed: ollama pull {OLLAMA_MODEL}")
            return
        
        print(f"✅ Connected to Ollama with model: {OLLAMA_MODEL}")
        
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
        
        for riddle_data in riddles:
            riddle_id = riddle_data['riddle_id']
            
            # Skip if outside range
            if riddle_id < start_id or riddle_id > end_id:
                continue
            
            try:
                mcq_data = self.process_single_riddle(riddle_data)
                
                if mcq_data:
                    self.save_mcq(mcq_data)
                    self.processed_riddles.add(riddle_id)
                    successful += 1
                else:
                    failed += 1
                
                # Save progress every 10 riddles
                if (successful + failed) % 10 == 0:
                    self.save_progress()
                    print(f"Progress: {successful + failed} processed - Success: {successful}, Failed: {failed}")
                
                # Small delay to avoid overwhelming Ollama
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\nProcess interrupted by user.")
                self.save_progress()
                break
            except Exception as e:
                print(f"Unexpected error processing riddle {riddle_id}: {e}")
                self.failed_riddles.add(riddle_id)
                failed += 1
                continue
        
        # Final save
        self.save_progress()
        
        print(f"\nMCQ Generation completed!")
        print(f"Total processed: {successful + failed}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if self.failed_riddles:
            print(f"Failed riddle IDs: {sorted(list(self.failed_riddles))}")
        print(f"Output saved to: {OUTPUT_FILE}")
    
    def retry_failed_riddles(self):
        """Retry all previously failed riddles."""
        if not self.failed_riddles:
            print("No failed riddles to retry.")
            return
        
        # Load riddles
        try:
            with open(RIDDLES_FILE, 'r', encoding='utf-8') as f:
                riddles = json.load(f)
        except Exception as e:
            print(f"Error loading riddles: {e}")
            return
        
        failed_list = sorted(list(self.failed_riddles.copy()))
        print(f"Retrying {len(failed_list)} failed riddles...")
        
        successful = 0
        still_failed = 0
        
        for riddle_id in failed_list:
            # Find the riddle data
            riddle_data = next((r for r in riddles if r['riddle_id'] == riddle_id), None)
            if not riddle_data:
                print(f"Riddle {riddle_id} not found in dataset")
                continue
            
            # Remove from failed list temporarily
            if riddle_id in self.failed_riddles:
                self.failed_riddles.remove(riddle_id)
            
            mcq_data = self.process_single_riddle(riddle_data)
            
            if mcq_data:
                self.save_mcq(mcq_data)
                self.processed_riddles.add(riddle_id)
                successful += 1
            else:
                still_failed += 1
            
            time.sleep(2)
        
        self.save_progress()
        
        print(f"Retry completed!")
        print(f"Successfully recovered: {successful}")
        print(f"Still failed: {still_failed}")

def main():
    """Main function to run MCQ generation."""
    generator = MCQGenerator()
    
    # Process all riddles
    generator.process_all_riddles()

if __name__ == "__main__":
    main()
