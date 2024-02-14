import shutil
from pathlib import Path

import pytest
from devtools import debug
from fractal_tasks_core.lib_input_models import Channel

from apx_fractal_task_collection.measure_features import measure_features
from apx_fractal_task_collection.clip_label_image import clip_label_image
from apx_fractal_task_collection.segment_secondary_objects import segment_secondary_objects
from apx_fractal_task_collection.calculate_illumination_profiles import calculate_illumination_profiles
from apx_fractal_task_collection.apply_basicpy_illumination_model import apply_basicpy_illumination_model

WELL_COMPONENT = "hcs_ngff.zarr/A/2"
IMAGE_COMPONENT = "hcs_ngff.zarr/A/2/0"

@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    source_dir = (Path(__file__).parent / "data").as_posix()
    dest_dir = (tmp_path / "data").as_posix()
    debug(source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def test_measure_features(test_data_dir):
    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=IMAGE_COMPONENT,
        metadata={},
        label_image_name='Label A',
        label_image_cycle=0,
        measure_intensity=True,
        measure_morphology=True,
        measure_texture=True,
        output_table_name='feature_table',
        level=0,
        overwrite=True
    )


def test_clip_label_image(test_data_dir):
    clip_label_image(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=IMAGE_COMPONENT,
        metadata={},
        label_image_name='Label A',
        clipping_mask_name='Label D',
        label_image_cycle=0,
        clipping_mask_cycle=2,
        output_label_cycle=0,
        output_label_name='clipped_label',
        level=0,
        overwrite=True
    )


def test_segment_secondary_objects(test_data_dir):
    segment_secondary_objects(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=WELL_COMPONENT,
        metadata={},
        label_image_name='Label A',
        channel=Channel(label='0_DAPI', wavelength_id=None),
        label_image_cycle=0,
        intensity_image_cycle=0,
        min_threshold=10,
        max_threshold=20,
        gaussian_blur=2,
        fill_holes_area=10,
        contrast_threshold=5,
        output_label_cycle=0,
        output_label_name='watershed_result',
        level=0,
        overwrite=True
    )


def test_illumination_correction(test_data_dir):
    calculate_illumination_profiles(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        illumination_profiles_folder=test_data_dir,
        n_images=1,
        overwrite=True
    )

    apply_basicpy_illumination_model(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=IMAGE_COMPONENT,
        metadata={},
        illumination_profiles_folder=test_data_dir,
        overwrite_input=True,
        new_component=None
    )





