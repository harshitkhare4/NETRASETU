"""
scratch/sanitize_paths.py
Replaces local Windows absolute paths and personal usernames with clean,
project-relative paths across documentation and report files.
"""

import os
import re

def sanitize_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    original = content
    # Replace personal download / machine path
    content = re.sub(r'C:\\Users\\[^\\]+\\Downloads\\26038\\', r'data_sources/', content)
    content = re.sub(r'C:/Users/[^/]+/Downloads/26038/', r'data_sources/', content)
    content = re.sub(r'C:\\Users\\[^\\]+\\', r'user_home/', content)
    
    # Replace project root absolute path with relative
    content = content.replace(r'C:\NetraSetu\models\\', r'models/')
    content = content.replace(r'C:\NetraSetu\models/', r'models/')
    content = content.replace(r'C:\NetraSetu\models', r'models')
    content = content.replace(r'C:\NetraSetu\data\\', r'data/')
    content = content.replace(r'C:\NetraSetu\data/', r'data/')
    content = content.replace(r'C:\NetraSetu\data', r'data')
    content = content.replace(r'C:\NetraSetu\results\\', r'results/')
    content = content.replace(r'C:\NetraSetu\results/', r'results/')
    content = content.replace(r'C:\NetraSetu\results', r'results')
    content = content.replace(r'C:\NetraSetu\\', r'./')
    content = content.replace(r'C:\NetraSetu', r'.')
    content = content.replace(r'C:/NetraSetu/', r'./')
    content = content.replace(r'C:/NetraSetu', r'.')
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Sanitized: {filepath}")

def main():
    root = r"C:\NetraSetu"
    targets = [
        os.path.join(root, "results"),
        os.path.join(root, "README.md"),
    ]
    for target in targets:
        if os.path.isfile(target):
            sanitize_file(target)
        else:
            for dirpath, _, filenames in os.walk(target):
                for fname in filenames:
                    if fname.endswith(('.txt', '.md', '.json', '.csv', '.html')):
                        sanitize_file(os.path.join(dirpath, fname))

if __name__ == '__main__':
    main()
