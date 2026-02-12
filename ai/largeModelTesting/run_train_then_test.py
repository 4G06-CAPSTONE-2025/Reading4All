import subprocess
import sys
import os
import time

def run_script(script_path):
    print(f"\n=== Running {script_path} ===\n")
    start_time = time.time()

    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )

    returncode = process.wait()
    elapsed = time.time() - start_time

    if returncode != 0:
        print(f"\n Script {script_path} failed (code {returncode})")
        return False, elapsed

    print(f"\n Finished {script_path} in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    return True, elapsed


if __name__ == "__main__":
    base_dir =os.path.dirname(os.path.abspath(__file__))

    train_script = os.path.join(base_dir, "train", "blip_onPubLayNet.py")
    test_script = os.path.join(base_dir, "test", "blip_onPubLayNet_test.py")

    # Run training
    success, elapsed = run_script(train_script)
    if not success:
        print("Training failed. Aborting testing.")
        sys.exit(1)

    # Run testing
    success, elapsed = run_script(test_script)
    if not success:
        print("Testing failed.")
        sys.exit(1)

    print("\nAll scripts finished successfully!")