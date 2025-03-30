"""
File handling module
 Manage file operations for text data files.
"""
import os

def bulk_rename(directory, prefix, extension=None):
    """Rename all files in a directory using a prefix and incrementing index"""
    file_count = 0  # for returning how many files were renamed

    # ensure directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # homogenize extension
    if extension and not extension.startswith('.'):
        extension = '.' + extension.lower()

    # loop over files in directory and generate index to add to file name
    for index, file in enumerate(os.listdir(directory)):
        file_path = os.path.join(directory, file)

        # skip directories
        if not os.path.isfile(file_path):
            continue

        # skip files that don't match the extension if one is specified
        if extension and not file.lower().endswith(extension):
            continue

        # determine file extension if not specified
        if not extension:
            _, file_ext = os.path.splitext(file)
        else:
            file_ext = extension

        # create new name for file
        new_path = os.path.join(directory, prefix + str(index) + file_ext)

        counter = 1 # set counter to append to duplicates
        # check if new file name exists in directory
        while os.path.exists(new_path) and new_path != file_path:
            # add counter value to make it unique
            new_name = prefix + str(index) + "_" + str(counter) + file_ext
            new_path = os.path.join(directory, new_name)
            counter += 1 # repeat until file name is unique

        if new_path != file_path: # check if file name changed
            os.rename(file_path, new_path)
            file_count += 1 # add to counter of file names changed

    print(f"Renamed {file_count} files.") # print statement for user
    return file_count

def move_files(source_dir, target_dir, extension=None):
    """Move files from source to target directory with optional extension"""
    file_count = 0

    # ensure directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directory '{source_dir}' does not exist.")

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # homogenize extension
    if extension and not extension.startswith('.'):
        extension = '.' + extension.lower()

    for file in os.listdir(source_dir):

        modified_file = file

        # skip directories
        if not os.path.isfile(os.path.join(source_dir, file)):
            continue

        # skip files that don't match the extension if one is specified
        if extension and not file.lower().endswith(extension):
            continue

        # split file name and extension
        file_name, file_ext = os.path.splitext(file)

        # check if file is in the target directory and rename if so
        counter = 1
        while modified_file in os.listdir(target_dir):
            modified_file = file_name + "_" + str(counter) + file_ext
            counter += 1

        try:
            os.rename(os.path.join(source_dir, file), os.path.join(target_dir, modified_file))
            file_count += 1
        except Exception as e:
            print(f"Error moving {file}: {e}")

    print(f"Moved {file_count} files.")
    return file_count


def delete_files(directory, extension):
    """Delete all files in a directory matching an extension"""
    file_count = 0

    # ensure directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    for file in os.listdir(directory): # loop over files in directory
        file_path = os.path.join(directory, file)
        if file.endswith(extension) and os.path.isfile(file_path):
            try: # delete file
                os.remove(file_path)
                file_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    print(f"Deleted {file_count} files.")
    return file_count

def list_files(directory, extension=None):
    """List all files in a directory with optional extension filtering"""
    files_list = []

    # ensure directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # homogenize extension
    if extension and not extension.startswith('.'):
        extension = '.' + extension.lower()

    for file in os.listdir(directory):

        # skip directories
        if not os.path.isfile(os.path.join(directory, file)):
            continue

        # skip files that don't match the extension if one is specified
        if extension and not file.lower().endswith(extension):
            continue

        files_list.append(file)
        print(file)

    return files_list