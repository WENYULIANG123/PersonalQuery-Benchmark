import sys
import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Send prompt to claude code cli and receive output.")
    parser.add_argument('prompt', nargs='?', default=None, help='The prompt text to send to Claude.')
    
    args = parser.parse_args()
    prompt = args.prompt

    # If no prompt argument, read from stdin if available
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
    
    if not prompt:
        print("Error: No prompt provided. Please provide a prompt as an argument or via stdin.", file=sys.stderr)
        sys.exit(1)

    try:
        # Construct the command
        # Using -u for unbuffered python output is good, but for subprocess we want streaming if possible
        # claude --print "prompt" seems to be the way
        
        # We'll use Popen to stream output in real-time if it generates slowly
        # However, --print might buffer. Let's try simple run first.
        
        cmd = ['claude', '--print', prompt]
        
        # Check if claude is executable
        # This relies on PATH being set correctly (environment activated)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # Line buffered
        )

        # Stream output
        full_output = []
        for line in process.stdout:
            print(line, end='')
            full_output.append(line)
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code != 0:
            stderr_output = process.stderr.read()
            print(f"\nError executing claude (Exit code {return_code}):", file=sys.stderr)
            print(stderr_output, file=sys.stderr)
            sys.exit(return_code)

    except FileNotFoundError:
        print("Error: 'claude' command not found. Ensure the conda environment is activated.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
