#!/usr/bin/env python3
"""
Launcher script for MCQ generation from riddles
"""

import argparse
import sys
from mcq import MCQGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate MCQs from Bengali riddles using Ollama')
    
    parser.add_argument('--start', type=int, default=1, 
                       help='Starting riddle ID (default: 1)')
    parser.add_argument('--end', type=int, 
                       help='Ending riddle ID (default: process all)')
    parser.add_argument('--single', type=int, 
                       help='Process only a single riddle by ID')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: process only first 5 riddles')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry all previously failed riddles')
    
    args = parser.parse_args()
    
    # Create generator
    generator = MCQGenerator()
    
    try:
        if args.single:
            print(f"Processing single riddle: {args.single}")
            # Load riddles to find the specific one
            import json
            from mcq import RIDDLES_FILE
            
            with open(RIDDLES_FILE, 'r', encoding='utf-8') as f:
                riddles = json.load(f)
            
            riddle_data = next((r for r in riddles if r['riddle_id'] == args.single), None)
            if riddle_data:
                mcq_data = generator.process_single_riddle(riddle_data)
                if mcq_data:
                    generator.save_mcq(mcq_data)
                    generator.processed_riddles.add(args.single)
                    print(f"✓ MCQ generated for riddle {args.single}")
                else:
                    print(f"✗ Failed to generate MCQ for riddle {args.single}")
                    sys.exit(1)
            else:
                print(f"✗ Riddle {args.single} not found")
                sys.exit(1)
        
        elif args.retry_failed:
            print("Retrying all previously failed riddles...")
            generator.retry_failed_riddles()
        
        elif args.test:
            print("Running test mode (first 5 riddles)...")
            generator.process_all_riddles(start_id=1, end_id=5)
        
        else:
            end_id = args.end
            print(f"Processing riddles from {args.start}" + (f" to {end_id}" if end_id else " to end"))
            generator.process_all_riddles(start_id=args.start, end_id=end_id)
        
        print("Done!")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()