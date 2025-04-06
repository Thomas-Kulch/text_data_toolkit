"""Unit tests for file operations module"""
from text_data_toolkit import file_operations as fop
import pytest
import os

@pytest.fixture
def create_test_directory(tmp_path):
    """create temp directory with test files"""
    CONTENT = "Content"
    source_directory = tmp_path / "source"
    target_directory = tmp_path / "target"
    source_directory.mkdir()
    target_directory.mkdir()

    for i in range(1, 4):
        test_file = source_directory / f"test{i}.txt"
        test_file.write_text(CONTENT, encoding="utf-8")

    return source_directory, target_directory


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
    fop.bulk_rename(source, "new")
    assert all(name.startswith("new") and name.endswith(".txt") for name in os.listdir(source))

def test_bulk_rename_directory_exist(create_test_directory):
    source, _ = create_test_directory

    # testing if we get error for non existing directory
    with pytest.raises(FileNotFoundError):
        fop.bulk_rename("nonexistent_dir", "prefix")

def test_bulk_rename_dup_file_name(create_test_directory):
    source, _ = create_test_directory
    pass

def test_move_files(create_test_directory):
    source, target = create_test_directory
    pass

def test_delete_files(create_test_directory):
    source, _ = create_test_directory
    pass

def test_list_files(create_test_directory):
    source, _ = create_test_directory
    pass

