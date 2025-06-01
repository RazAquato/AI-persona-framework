# script to check what code lacks unit-testing
import os

def find_python_files(directory):
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_') and file != '__init__.py':
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        contents = f.read()
                        if '#placeholder' not in contents.lower():
                            py_files.append(full_path)
                except Exception as e:
                    print(f"Warning: Could not read {full_path}: {e}")
    return py_files

def corresponding_test_exists(py_file, test_dirs):
    filename = os.path.basename(py_file)
    test_filename = f'test_{filename}'
    for test_dir in test_dirs:
        if os.path.exists(os.path.join(test_dir, test_filename)):
            return True
    return False

def main():
    base_dirs = {
        'LLM-client': ['LLM-client/tests'],
        'memory-server': ['memory-server/tests'],
        'shared': ['shared/tests']
    }

    for base_dir, test_dirs in base_dirs.items():
        print(f'\nChecking {base_dir} for untested modules:')
        py_files = find_python_files(base_dir)
        untested = []
        for py_file in py_files:
            if not corresponding_test_exists(py_file, test_dirs):
                untested.append(py_file)
        if untested:
            for file in untested:
                print(f'  - {file}')
        else:
            print('  All modules have corresponding tests.')

if __name__ == '__main__':
    main()

