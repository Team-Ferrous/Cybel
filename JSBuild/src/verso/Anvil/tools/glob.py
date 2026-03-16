import os
import fnmatch


def glob_files(pattern: str, path: str = ".", recursive: bool = True) -> str:
    """
    List files matching a glob pattern.
    """
    matches = []
    if recursive:
        for root, dirs, files in os.walk(path):
            for filename in fnmatch.filter(files, pattern):
                matches.append(os.path.join(root, filename))
    else:
        for filename in fnmatch.filter(os.listdir(path), pattern):
            matches.append(os.path.join(path, filename))

    if not matches:
        return "No files matched."

    return "\n".join(matches)
