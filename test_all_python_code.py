import subprocess
import os

def run_tests_in(directory):
    print(f"\n🧪 Running tests in: {directory}")
    result = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=directory,
    )
    return result.returncode == 0

def run_all_tests():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    test_targets = [
        "LLM-client",
        "memory-server",
        "shared",  # Optional
    ]

    all_passed = True
    for target in test_targets:
        full_path = os.path.join(base_dir, target)
        if os.path.exists(os.path.join(full_path, "tests")):
            passed = run_tests_in(full_path)
            all_passed = all_passed and passed
        else:
            print(f"⚠️  No 'tests' folder found in {target}, skipping.")

    if all_passed:
        print("\n✅ All tests passed.")
        exit(0)
    else:
        print("\n❌ Some tests failed.")
        exit(1)

if __name__ == "__main__":
    run_all_tests()

