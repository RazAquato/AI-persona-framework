import os

def print_tree(start_path='.', indent=''):
    items = sorted(os.listdir(start_path))
    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = (i == len(items) - 1)
        branch = '└── ' if is_last else '├── '
        print(indent + branch + item)

        if os.path.isdir(path):
            extension = '    ' if is_last else '│   '
            print_tree(path, indent + extension)

if __name__ == '__main__':
    print_tree('.')

