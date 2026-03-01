#!/usr/bin/env python3
import json
import glob
import sys

def validate_json_files():
    print("Starting JSON validation...")
    valid = True
    files = glob.glob('/home/ubuntu/CybelMemoryConstruct*.json')
    
    if not files:
        print("No JSON files found to validate!")
        return False
    
    for file in files:
        print(f"Validating {file}...")
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Check if the expected structure is present
                if "title" not in data or "content" not in data:
                    print(f"✗ Missing required fields in {file}")
                    valid = False
                else:
                    print(f"✓ Valid JSON with correct structure")
        except json.JSONDecodeError as e:
            valid = False
            print(f"✗ Invalid JSON: {e}")
    
    print("\nOverall validation result:", "PASSED" if valid else "FAILED")
    return valid

if __name__ == "__main__":
    success = validate_json_files()
    sys.exit(0 if success else 1)
