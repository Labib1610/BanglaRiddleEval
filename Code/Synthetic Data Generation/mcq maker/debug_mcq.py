#!/usr/bin/env python3

import json
import requests
from mcq import MCQGenerator

def test_single_generation():
    """Test MCQ generation with debug output."""
    
    riddle = "কোন জিনিস কাটলে বাড়ে?"
    correct_answer = "পুকুর"
    
    generator = MCQGenerator()
    
    # Test Ollama connection
    print("Testing Ollama connection...")
    if not generator.check_ollama_connection():
        print("Connection failed!")
        return
    
    print(f"\nGenerating MCQ for:")
    print(f"Riddle: {riddle}")
    print(f"Answer: {correct_answer}")
    print("="*50)
    
    # Generate options with debug
    try:
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
ধাঁধা: "কোন জিনিস কাটলে বাড়ে?"
সঠিক উত্তর: "পুকুর"
বিভ্রান্তিকর বিকল্প:
- "নখ" (কাটলে বাড়ে কিন্তু ভিন্ন অর্থে)
- "চুল" (কাটলে দ্রুত বৃদ্ধি পায়)
- "গাছের ডাল" (প্রুনিং করলে নতুন ডাল গজায়)

আবশ্যক শর্ত:
- প্রতিটি ভুল উত্তর যেন কোন না কোনভাবে যুক্তিসংগত মনে হয়
- উত্তরগুলো খুব সহজ বা খুব কঠিন নয়, মধ্যম চ্যালেঞ্জিং হতে হবে
- একই ধরনের/শ্রেণীর বস্তু হতে হবে
- গভীর বিশ্লেষণ ছাড়া সঠিক উত্তর খুঁজে পাওয়া কঠিন হতে হবে

JSON ফরম্যাট:
{{
  "options": ["বিকল্প ১", "বিকল্প ২", "বিকল্প ৩", "বিকল্প ৪"]
}}"""
        
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gpt-oss:20b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            
            print("Raw response:")
            print(response_text)
            print("\n" + "="*50)
            
            # Try to extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                print("Extracted JSON:")
                print(json_str)
                print("\n" + "="*50)
                
                try:
                    parsed_json = json.loads(json_str)
                    options = parsed_json.get('options', [])
                    
                    print("Parsed options:")
                    for i, opt in enumerate(options, 1):
                        print(f"{i}. {opt}")
                    
                    print(f"\nCorrect answer included: {correct_answer in options}")
                    
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
            else:
                print("No JSON found in response")
        else:
            print(f"Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_generation()