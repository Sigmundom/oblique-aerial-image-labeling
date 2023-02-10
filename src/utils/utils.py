from os import makedirs, path


def ensure_folder_exists(folder):
    if not path.isdir(folder):
        makedirs(folder)