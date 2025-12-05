#!/usr/bin/env python3
"""
Test script to verify semantic ambiguity dataset loading for Gemma3:12b
"""

import json
from pathlib import Path

def test_dataset_loading():
    dataset_path = Path("/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BornoDhada/main_dataset/riddles_semantic_ambiguity.json")
    
    print(f"🔍 Testing dataset loading from: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, "r", encoding="utf8") as f:
            data = json.load(f)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total items: {len(data)}")
        
        if len(data) > 0:
            print("\n📋 Sample item structure:")
            sample = data[0]
            for key, value in sample.items():
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    test_dataset_loading()