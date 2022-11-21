from pathlib import Path


def find_directory_root(starting_path: Path, anchor_file_name: str) -> Path:
    """Basic recursion to find the path of an anchor file.
    This is useful to infer the root of the project repository
    """
    if starting_path.is_file():
        starting_path = starting_path.parent

    file_paths = [path for path in starting_path.iterdir() if path.is_file()]
    target_path = [path for path in file_paths if path.name == anchor_file_name]

    if len(target_path) == 1:
        root_path = target_path[0].parent
    elif len(target_path) > 1:
        raise ValueError("The anchor file is present multiple times")
    else:
        return find_directory_root(
            starting_path=starting_path.parent, anchor_file_name=anchor_file_name
        )

    return root_path
