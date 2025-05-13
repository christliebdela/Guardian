import os
import re
import sys
import glob
import shutil

def remove_comments(content):

    content = re.sub(r'', '', content)
    content = re.sub(r"", '', content)

    result = []
    for line in content.split('\n'):

        if '

            line = line.split('
        result.append(line)

    cleaned = '\n'.join(line.rstrip() for line in result)

    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)

    return cleaned

def process_file(file_path, backup=True):
    print(f"Processing: {file_path}")

    if backup:
        backup_path = file_path + '.bak'
        shutil.copy2(file_path, backup_path)
        print(f"  Backup created: {backup_path}")

    try:

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        cleaned = remove_comments(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        print(f"  Removed comments successfully")
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        if backup:

            shutil.copy2(backup_path, file_path)
            print(f"  Restored from backup due to error")

def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_comments.py <file_pattern>")
        print("Example: python remove_comments.py *.py")
        print("         python remove_comments.py guardian.py")
        return

    file_pattern = sys.argv[1]
    files = glob.glob(file_pattern)

    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(files)} files matching {file_pattern}")
    for file_path in files:
        process_file(file_path)

    print("Done!")

if __name__ == "__main__":
    main()