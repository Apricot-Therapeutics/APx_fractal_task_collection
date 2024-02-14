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


def generate_test_data() -> None:
    """
    Generate some NGFF data.
    """
    path = Path(INPUT_PATH).joinpath("hcs_ngff.zarr")
    row_names = ["A", "B"]
    col_names = ["1", "2", "3"]
    well_paths = ["A/2", "B/3"]
    field_paths = ["0", "1", "2"]

    # generate data
    mean_val = 10
    num_wells = len(well_paths)
    num_fields = len(field_paths)
    size_xy = 128
    size_z = 1
    size_c = 2
    num_labels = 2
    rng = np.random.default_rng(0)
    data = rng.poisson(mean_val, size=(num_wells,
                                       num_fields,
                                       size_c,
                                       size_z,
                                       size_xy,
                                       size_xy)).astype(np.uint8)

    # write the plate of images and corresponding metadata
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store, overwrite=True)
    write_plate_metadata(root, row_names, col_names, well_paths)
    for wi, wp in enumerate(well_paths):
        row, col = wp.split("/")
        row_group = root.require_group(row)
        well_group = row_group.require_group(col)
        write_well_metadata(well_group, field_paths)
        for fi, field in enumerate(field_paths):
            image_group = well_group.require_group(str(field))
            write_image(image=data[wi, fi], group=image_group, axes="czyx",
                        storage_options=dict(chunks=(1, size_xy, size_xy)))

            image_group.attrs["omero"] = {
                "channels": [
                    {
                        "color": "00FFFF",
                        "window": {"start": 0, "end": 20, "min": 0,
                                   "max": 255},
                        "label": f"{fi}_DAPI",
                        "active": True,
                        "wavelength_id": 'UV - DAPI'
                    },
                    {
                        "color": "008000",
                        "window": {"start": 0, "end": 20, "min": 0,
                                   "max": 255},
                        "label": f"{fi}_GFP",
                        "active": True,
                        "wavelength_id": 'Blue - FITC'
                    }
                ]
            }

            if fi == 0:
                # add labels...
                blobs = [binary_blobs(length=size_xy,
                                      volume_fraction=0.4
                                      , n_dim=3).astype('uint8')
                         for n in range(0, num_labels)]
                label_images = [label(b[:size_z, :, :]) for b in blobs]
                label_names = ["Label A", "Label B"]
                for i, label_name in enumerate(label_names):
                    write_labels(label_images[i], image_group, axes="zyx",
                                 name=label_name)

            if fi == 2:
                # add labels...
                blobs = [binary_blobs(length=size_xy,
                                      volume_fraction=0.4
                                      , n_dim=3).astype('uint8')
                         for n in range(0, num_labels)]
                label_images = [label(b[:size_z, :, :]) for b in blobs]
                label_names = ["Label C", "Label D"]
                for i, label_name in enumerate(label_names):
                    write_labels(label_images[i], image_group, axes="zyx",
                                 name=label_name)



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





