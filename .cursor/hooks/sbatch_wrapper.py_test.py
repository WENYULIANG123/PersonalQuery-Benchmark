#!/usr/bin/env python3
"""
Simple test to verify sbatch_wrapper.py works
"""
import sys
import subprocess

def main():
    # Test writing to sbatch_wrapper.py
    test_data = """
if __name__ == "__main__":
    try:
        # Test if we can write to the file
        with open("/home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py", "a") as f:
            f.write("TEST WRITE\\n")
            f.flush()
            print("Successfully wrote test data to sbatch_wrapper.py")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    main()
