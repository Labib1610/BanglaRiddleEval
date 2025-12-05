#!/usr/bin/env python3
"""
Debug CoT Generation
===================

Test script to debug the Chain of Thought generation with Ollama.
"""

import requests
import json
import sys
import os

def clean_reasoning_format(text: str) -> str:
    """Clean up the reasoning text to remove unwanted formatting."""
    
    # Remove markdown bold formatting
    text = text.replace('**', '')
    
    # Remove extra newlines and normalize spacing
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:  # Only keep non-empty lines
            # Remove extra spaces and normalize
            line = ' '.join(line.split())
            cleaned_lines.append(line)
    
    # Join with single newlines
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove any remaining formatting artifacts
    cleaned_text = cleaned_text.replace('\\n', '\n')
    cleaned_text = cleaned_text.replace('  ', ' ')  # Remove double spaces
    
    # Ensure proper Bengali numeral format and fix conclusion numbering
    cleaned_text = cleaned_text.replace('১. ', '১. ')
    cleaned_text = cleaned_text.replace('২. ', '২. ')
    cleaned_text = cleaned_text.replace('৩. ', '৩. ')
    cleaned_text = cleaned_text.replace('৪. ', 'সিদ্ধান্ত: ')  # Convert ৪. to সিদ্ধান্ত:
    cleaned_text = cleaned_text.replace('সিদ্ধান্ত: ', 'সিদ্ধান্ত: ')
    
    return cleaned_text

def test_cot_generation():
    """Test CoT generation with a simple riddle."""
    
    riddle = "এক থালা সুপারি, গুনতে নারি।"
    answer = "তারা"
    
    prompt = f"""
আপনি একটি বাংলা ধাঁধার জন্য বিস্তারিত যুক্তি তৈরি করবেন।

ধাঁধা: "{riddle}"
উত্তর: "{answer}"

নিচের সুনির্দিষ্ট ফরম্যাটে বিশ্লেষণ করুন (প্রতিটি লাইনে একটি করে শব্দ বিশ্লেষণ):

1. উত্তর চিহ্নিতকরণ: ধাঁধার নির্দিষ্ট শব্দগুলো উদ্ধৃত করুন
2. রূপকের ব্যাখ্যা: রূপকটি কী প্রতিনিধিত্ব করে তা ব্যাখ্যা করুন  
3. উত্তরের সাথে সংযোগ: "{answer}" এর কোন বৈশিষ্ট্য এই ধাঁধার সাথে মিলে যায় তা ব্যাখ্যা করুন
4. সিদ্ধান্ত: কেন এটাই একমাত্র যুক্তিসংগত উত্তর তা সংক্ষেপে বলুন

উদাহরণ ফরম্যাট:
১. 'এক থালা': এখানে আকাশকে একটি বিশাল থালার সাথে তুলনা করা হয়েছে।
২. 'সুপারি': সুপারি যেমন ছোট ছোট গোল হয়, আকাশের নক্ষত্রগুলোকেও দেখতে ছোট বিন্দুর মতো লাগে।
৩. 'গুনতে নারি': সুপারি গোনা সম্ভব হলেও, আকাশের তারা বা নক্ষত্র অসংখ্য, যা গুনে শেষ করা যায় না।
সিদ্ধান্ত: আকাশের বিশাল থালায় ছড়িয়ে থাকা অগণিত নক্ষত্রই হলো এই ধাঁধার উত্তর।

গুরুত্বপূর্ণ: কোনো ** (বোল্ড), markdown বা বিশেষ ফরম্যাটিং ব্যবহার করবেন না। শুধু সাধারণ বাংলা টেক্সট দিন।
"""
    
    print("=== Testing CoT Generation ===")
    print(f"Riddle: {riddle}")
    print(f"Answer: {answer}")
    print("\n=== Sending request to Ollama ===")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gpt-oss:20b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '').strip()
            
            print("=== Raw Response ===")
            print(response_text)
            
            # Process Bengali reasoning text
            print("\n=== Processing Bengali Reasoning ===")
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
            
            # Clean up formatting
            reasoning_text = clean_reasoning_format(reasoning_text)
            
            print("✓ Text processing successful!")
            
            print("\n=== Formatted Bengali Reasoning ===")
            print(reasoning_text)
            
            # Validate format
            required_elements = ['১.', '২.', '৩.', 'সিদ্ধান্ত']
            missing_elements = []
            
            for element in required_elements:
                if element not in reasoning_text:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"\n⚠️  Missing required elements: {missing_elements}")
            else:
                print(f"\n✅ All required elements found!")
            
            # Save to file for inspection
            debug_file = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/Code/cot maker/debug_cot_response.json"
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "riddle": riddle,
                    "answer": answer,
                    "reasoning": reasoning_text,
                    "raw_response": response_text,
                    "format_check": {
                        "missing_elements": missing_elements,
                        "length": len(reasoning_text),
                        "valid": len(missing_elements) == 0
                    }
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\nDebug output saved to: {debug_file}")
            
        else:
            print(f"✗ Ollama API error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")

if __name__ == "__main__":
    test_cot_generation()