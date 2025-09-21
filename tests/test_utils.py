from pathlib import Path

from app.utils import DataPaths, american_to_probability, make_merge_key


def test_make_merge_key_handles_suffixes():
    assert make_merge_key("Smith Jr., John") == make_merge_key("John Smith")


def test_american_to_probability_positive():
    prob = american_to_probability(150)
    assert round(prob, 4) == 0.4


def test_data_paths_create_directories(tmp_path):
    paths = DataPaths.from_root(tmp_path)
    assert paths.raw_dir.exists()
    assert paths.processed_dir.exists()
    assert paths.metadata_path.parent == tmp_path / "data"
