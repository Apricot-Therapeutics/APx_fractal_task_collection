import shutil
from pathlib import Path

import pytest
from devtools import debug

from custom_fractal_task_collection.thresholding_task import thresholding_task


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    source_dir = (Path(__file__).parent / "data/ngff_example").as_posix()
    dest_dir = (tmp_path / "ngff_example").as_posix()
    debug(source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def test_thresholding_task(test_data_dir):
    thresholding_task(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component="my_image",
        metadata={},
    )
