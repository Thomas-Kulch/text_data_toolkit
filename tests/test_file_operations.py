"""Unit tests for file operations module"""
from text_data_toolkit import file_operations as fop
import os
import pytest

CONTENT = "Content"

@pytest.fixture
def create_test_directory(tmp_path):
    """create temp directory with test files"""
    source_directory = tmp_path / "source"
    target_directory = tmp_path / "target"
    source_directory.mkdir()
    target_directory.mkdir()

    for i in range(1, 4):
        test_file = source_directory / f"test{i}.txt"
        test_file.write_text(CONTENT, encoding="utf-8")

    return source_directory, target_directory

# bulk_rename testing
def test_bulk_rename_extension(create_test_directory):
    source, _ = create_test_directory

    # testing rename with extension passed
    fop.bulk_rename(source, "new", ".txt")
    assert all(name.startswith("new") and name.endswith(".txt") for name in os.listdir(source))

    # testing homogenization of extension
    fop.bulk_rename(source, "new_new", "txt")
    assert all(name.startswith("new_new") and name.endswith(".txt") for name in os.listdir(source))

def test_bulk_rename_no_extension(create_test_directory):
    source, _ = create_test_directory

    # testing bulk rename with no extension passed
    file_count = fop.bulk_rename(source, "new")
    assert all(name.startswith("new") and name.endswith(".txt") for name in os.listdir(source))
    assert file_count == 3

def test_bulk_rename_directory_exist(create_test_directory):
    source, _ = create_test_directory

    # testing if we get error for non existing directory
    with pytest.raises(FileNotFoundError):
        fop.bulk_rename("nonexistent_dir", "prefix")

def test_bulk_rename_dup_file_name(create_test_directory):
    source, _ = create_test_directory
    # add new file
    new_file = source / f"new1.txt"
    new_file.write_text(CONTENT, encoding="utf-8")
    # bulk rename the files in the directory to test duplicate naming
    fop.bulk_rename(source, "new", ".txt")

    # check if duplicate renaming worked
    assert os.path.exists(source / f"new1_1.txt")

# move_files testing
def test_move_files(create_test_directory):
    source, target = create_test_directory

    file_count = fop.move_files(source, target)

    # check if files got moved
    source_file = source / "test1.txt"
    dest_file = target / "test1.txt"
    assert dest_file.exists(), "File not found in destination"

    # check if file is gone from old location
    assert not source_file.exists(), "File still exists in source location"

    assert file_count == 3

def test_move_files_duplicates(create_test_directory):
    source, target = create_test_directory
    fop.move_files(source, target)
    new_file = source / f"test1.txt"
    new_file.write_text(CONTENT, encoding="utf-8")
    fop.move_files(source, target)

    assert os.path.exists(target / f"test1_1.txt")

# delete_files testing
def test_delete_files(create_test_directory):
    source, _ = create_test_directory
    file_count = fop.delete_files(source, "txt")
    assert not os.path.exists(source / f"test2.txt")
    assert file_count == 3

# list files testing
def test_list_files_extension(create_test_directory):
    source, _ = create_test_directory
    files_list = fop.list_files(source, ".txt")
    assert f"test1.txt" in files_list

def test_list_files_no_extension(create_test_directory):
    source, _ = create_test_directory
    files_list = fop.list_files(source)
    assert f"test3.txt" in files_list