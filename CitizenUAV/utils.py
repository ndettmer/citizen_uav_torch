import os


def get_pid_from_path(path):
    """
    Takes a path to an INaturalist image file and removes the directories and file extension leaving only the photo ID.
    :param path: Path to the file
    :return int: INaturalist photo ID
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    return int(filename)
