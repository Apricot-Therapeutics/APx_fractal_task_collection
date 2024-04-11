import shutil
from pathlib import Path
import os
import anndata as ad
import numpy as np
import pandas as pd

import pytest
from devtools import debug
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.tasks.copy_ome_zarr import copy_ome_zarr

from apx_fractal_task_collection.utils import TextureFeatures, FEATURE_LABELS
from apx_fractal_task_collection.tasks.measure_features import measure_features
from apx_fractal_task_collection.tasks.clip_label_image import clip_label_image
from apx_fractal_task_collection.tasks.segment_secondary_objects import segment_secondary_objects
from apx_fractal_task_collection.tasks.calculate_illumination_profiles import calculate_illumination_profiles
from apx_fractal_task_collection.tasks.apply_basicpy_illumination_model import apply_basicpy_illumination_model
from apx_fractal_task_collection.tasks.convert_channel_to_label import convert_channel_to_label
from apx_fractal_task_collection.tasks.filter_label_by_size import filter_label_by_size
from apx_fractal_task_collection.tasks.label_assignment_by_overlap import label_assignment_by_overlap
from apx_fractal_task_collection.tasks.aggregate_tables_to_well_level import aggregate_tables_to_well_level
from apx_fractal_task_collection.tasks.chromatic_shift_correction import chromatic_shift_correction
from apx_fractal_task_collection.tasks.compress_zarr_for_visualization import compress_zarr_for_visualization
from apx_fractal_task_collection.tasks.create_ome_zarr_multiplex_IC6000 import create_ome_zarr_multiplex_IC6000
from apx_fractal_task_collection.tasks.IC6000_to_ome_zarr import IC6000_to_ome_zarr
from apx_fractal_task_collection.tasks.multiplexed_pixel_clustering import multiplexed_pixel_clustering
from apx_fractal_task_collection.tasks.stitch_fovs_with_overlap import stitch_fovs_with_overlap
from apx_fractal_task_collection.tasks.detect_blob_centroids import detect_blob_centroids
from apx_fractal_task_collection.tasks.apply_mask import apply_mask
#from apx_fractal_task_collection.tasks.ashlar_stitching_and_registration import ashlar_stitching_and_registration
#from apx_fractal_task_collection.tasks.ashlar_stitching_and_registration_pure import ashlar_stitching_and_registration

WELL_COMPONENT_2D = "hcs_ngff_2D.zarr/A/2"
IMAGE_COMPONENT_2D = "hcs_ngff_2D.zarr/A/2/0"
WELL_COMPONENT_3D = "hcs_ngff_3D.zarr/A/2"
IMAGE_COMPONENT_3D = "hcs_ngff_3D.zarr/A/2/0"

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

@pytest.fixture(scope="function")
def fixed_test_data_dir() -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    source_dir = Path('/data/homes/atschan/PhD/Code/Python/apx_fractal_task_collection/tests/data').as_posix()
    dest_dir = Path('/data/active/atschan/apx_fractal_task_collection/tests/data').as_posix()
    debug(source_dir, dest_dir)
    try:
        shutil.rmtree(dest_dir)
    except:
        pass
    shutil.copytree(source_dir, dest_dir)
    return dest_dir

@pytest.mark.parametrize("component", [IMAGE_COMPONENT_2D, IMAGE_COMPONENT_3D])
def test_measure_features(test_data_dir, component):
    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component,
        metadata={},
        label_image_name='Label A',
        measure_intensity=True,
        measure_morphology=True,
        channels_to_include=None,
        channels_to_exclude=[ChannelInputModel(label='0_GFP', wavelength_id=None)],
        measure_texture=TextureFeatures(
            haralick=True,
            laws_texture_energy=True,
            clip_value=3000,
            clip_value_exceptions={'0_DAPI': 5000}
        ),
        measure_population=True,
        ROI_table_name='FOV_ROI_table',
        calculate_internal_borders=True,
        output_table_name='feature_table',
        level=0,
        overwrite=True,
    )

    # assert that the feature table exists in correct location
    feature_table_path = Path(test_data_dir).joinpath(
        component,
        "tables/feature_table")
    assert feature_table_path.exists(),\
        f"Feature table not found at {feature_table_path}"

    # assert that the obs contain correct columns
    feature_table = ad.read_zarr(feature_table_path)
    expected_columns = ['label',
                        'well_name',
                        'ROI',
                        'is_border_internal',
                        'is_border_external']

    assert feature_table.obs.columns.tolist() == expected_columns, \
        f"Expected columns {expected_columns}," \
        f" but got {feature_table.obs.columns.tolist()}"

    # assert that the feature table contains correct columns
    morphology_labels = FEATURE_LABELS['morphology'].copy()
    intensity_labels = FEATURE_LABELS['intensity'].copy()
    texture_labels = FEATURE_LABELS['texture'].copy()
    population_labels = FEATURE_LABELS['population'].copy()

    # remove some morphology features in 3D case
    if component == IMAGE_COMPONENT_3D:
        morphology_labels.remove('Morphology_eccentricity')
        morphology_labels.remove('Morphology_orientation')
        morphology_labels.remove('Morphology_perimeter')
        morphology_labels.remove('Morphology_roundness')

    channels = ['0_DAPI']

    morphology_columns = ["Label A_" + c for c in morphology_labels]
    intensity_columns = []
    texture_columns = []
    population_columns = ["Label A_" + c for c in population_labels]

    for channel in channels:

        intensity_columns.extend(["Label A_" + c + f"_{channel}" for c in intensity_labels])
        texture_columns.extend(["Label A_" + c + f"_{channel}" for c in texture_labels])

    expected_columns = morphology_columns +\
                       intensity_columns +\
                       texture_columns +\
                       population_columns

    print(feature_table.to_df().columns)
    assert np.isin(feature_table.var.index.tolist(), expected_columns).all(), \
        f"Expected columns {expected_columns}," \
        f" but got {feature_table.var.index.tolist()}"

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_clip_label_image(test_data_dir, component):
    clip_label_image(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component,
        metadata={},
        label_image_name='Label A',
        clipping_mask_name='Label D',
        output_label_cycle=0,
        output_label_name='clipped_label',
        level=0,
        overwrite=True
    )

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_apply_mask(test_data_dir, component):
    apply_mask(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component,
        metadata={},
        label_image_name='Label A',
        mask_label_name='Label D',
        output_label_cycle=0,
        output_label_name='masked_label',
        level=0,
        overwrite=True
    )

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_segment_secondary_objects(test_data_dir, component):
    segment_secondary_objects(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component,
        metadata={},
        label_image_name='Label A',
        channel=ChannelInputModel(label='0_DAPI', wavelength_id=None),
        ROI_table_name='FOV_ROI_table',
        min_threshold=10,
        max_threshold=20,
        gaussian_blur=2,
        fill_holes_area=10,
        contrast_threshold=5,
        mask=None,
        output_label_cycle=0,
        output_label_name='watershed_result',
        level=0,
        overwrite=True
    )

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_detect_blob_centroids(test_data_dir, component):
    detect_blob_centroids(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component,
        metadata={},
        channel=ChannelInputModel(label='0_DAPI', wavelength_id=None),
        ROI_table_name='FOV_ROI_table',
        min_sigma=1,
        max_sigma=10,
        num_sigma=1,
        threshold=0.002,
        output_label_cycle=0,
        output_label_name='blobs_centroids',
        level=0,
        relabeling=True,
        overwrite=True
    )

@pytest.mark.parametrize("component", [IMAGE_COMPONENT_2D, IMAGE_COMPONENT_3D])
def test_illumination_correction(test_data_dir, component):
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
        component=component,
        metadata={},
        illumination_profiles_folder=test_data_dir,
        overwrite_input=True,
        new_component=None
    )

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_convert_channel_to_label(test_data_dir, component):
    convert_channel_to_label(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component,
        metadata={},
        channel_label='0_DAPI',
        output_label_name='DAPI',
        output_cycle=0,
        overwrite=True
    )

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_filter_label_by_size(test_data_dir, component):
    filter_label_by_size(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        component=component,
        label_name='Label A',
        output_label_name='filtered_label',
        min_size=10,
        max_size=100,
        overwrite=True
    )

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_label_assignment_by_overlap(test_data_dir, component):

    # create a feature table
    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component + '/0',
        metadata={},
        label_image_name='Label B',
        measure_intensity=True,
        measure_morphology=True,
        measure_texture=TextureFeatures(
            texture_features=["haralick", "lte"],
            clip_value=3000,
            clip_value_exceptions={'0_DAPI': 5000}
        ),
        measure_population=False,
        ROI_table_name='FOV_ROI_table',
        calculate_internal_borders=True,
        output_table_name='feature_table',
        level=0,
        overwrite=True
    )

    child_table = ad.read_zarr(Path(test_data_dir).joinpath(
        component,
        "0/tables/feature_table")
    )
    old_obs_columns = list(child_table.obs.columns)

    label_assignment_by_overlap(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        component=component,
        parent_label_image='Label A',
        child_label_image='Label B',
        child_table_name='feature_table',
        level=0,
        overlap_threshold=0.6,
    )

    child_table = ad.read_zarr(Path(test_data_dir).joinpath(
        component,
        "0/tables/feature_table")
    )

    print(child_table.obs)

    # assert that child_table.obs contains the expected columns
    expected_obs_columns = old_obs_columns + \
                           ['Label A_label', 'Label B_Label A_overlap']
    new_obs_columns = list(child_table.obs.columns)
    assert new_obs_columns == expected_obs_columns, \
        f"Expected obs columns {expected_obs_columns}," \
        f" but got {new_obs_columns}"

    # assert whether label assignments are correct
    if component == WELL_COMPONENT_2D:
        label_assignments = {9: 6,
                             14: 6,
                             31: 19,
                             38: 19,
                             36: pd.NA,
                             20: pd.NA,
                             12: 8}
    elif component == WELL_COMPONENT_3D:
        label_assignments = {4: 2,
                             10: 2,
                             1: pd.NA,
                             }

    for child_label, parent_label in label_assignments.items():
        value = child_table.obs.loc[
            child_table.obs.label == child_label, 'Label A_label'].values[0]
        try:
            assert value == parent_label, \
                f"Label assignment failed, expected {parent_label}," \
                f" but got {value}"
        except TypeError:
            assert pd.isna(value), \
                f"Label assignment failed, expected {parent_label}," \
                f" but got {value}"

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_aggregate_tables_to_well_level(test_data_dir, component):

    # create a feature table
    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component + "/0",
        metadata={},
        label_image_name='Label B',
        measure_intensity=True,
        measure_morphology=True,
        measure_texture=TextureFeatures(
            texture_features=["haralick", "lte"],
            clip_value=3000,
            clip_value_exceptions={'0_DAPI': 5000}
        ),
        measure_population=True,
        ROI_table_name='FOV_ROI_table',
        calculate_internal_borders=True,
        output_table_name='feature_table',
        level=0,
        overwrite=True
    )

    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component + "/1",
        metadata={},
        label_image_name='Label B',
        measure_intensity=True,
        measure_morphology=True,
        measure_texture=TextureFeatures(
            texture_features=["haralick", "lte"],
            clip_value=3000,
            clip_value_exceptions={'0_DAPI': 5000}
        ),
        measure_population=True,
        ROI_table_name='FOV_ROI_table',
        calculate_internal_borders=True,
        output_table_name='feature_table',
        level=0,
        overwrite=True
    )

    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=component + "/2",
        metadata={},
        label_image_name='Label B',
        measure_intensity=True,
        measure_morphology=True,
        measure_texture=TextureFeatures(
            texture_features=["haralick", "lte"],
            clip_value=3000,
            clip_value_exceptions={'0_DAPI': 5000}
        ),
        measure_population=True,
        ROI_table_name='FOV_ROI_table',
        calculate_internal_borders=True,
        output_table_name='feature_table',
        level=0,
        overwrite=True
    )

    aggregate_tables_to_well_level(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        component=component,
        input_table_name='feature_table',
        output_table_name='feature_table',
        overwrite=True
    )

@pytest.mark.parametrize("component", [IMAGE_COMPONENT_2D, IMAGE_COMPONENT_3D])
def test_chromatic_shift_correction(test_data_dir, component):
    chromatic_shift_correction(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        component=component,
        correction_channel_labels=['0_DAPI', '0_GFP'],
        reference_channel_label='0_DAPI'
    )

    Path(os.getcwd()).joinpath("TransformParameters.0.txt").unlink(
        missing_ok=True)

@pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
def test_compress_zarr_for_visualization(test_data_dir, component):

    copy_ome_zarr(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={'plate': 'hcs_ngff',
                  'well': ['A/2', 'B/3'],
                  'image': ['0', '1', '2']},
        project_to_2D=False,
        suffix="vis",
        ROI_table_names=("well_ROI_table", "FOV_ROI_table")
    )

    compress_zarr_for_visualization(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={'plate': 'hcs_ngff',
                  'copy_ome_zarr': {'suffix': 'vis'}},
        component=component,
        output_zarr_path=Path(test_data_dir).joinpath(
            "hcs_ngff_vis.zarr").as_posix(),
        overwrite=True
    )

def test_IC6000_conversion(test_data_dir):

    create_ome_zarr_multiplex_IC6000(
        input_paths=[
            Path(test_data_dir).joinpath("IC6000_data/cycle_0").as_posix(),
            Path(test_data_dir).joinpath("IC6000_data/cycle_1").as_posix()],
        output_path=Path(test_data_dir).joinpath("IC6000_data").as_posix(),
        metadata={},
        allowed_channels={'0':[OmeroChannel(label='0_DAPI',
                                            wavelength_id='UV - DAPI'),
                               OmeroChannel(label='0_GFP',
                                            wavelength_id='Blue - FITC'),
                               OmeroChannel(label='0_RFP',
                                            wavelength_id='Green - dsRed'),
                               OmeroChannel(label='0_FR',
                                            wavelength_id='Red - Cy5')],
                          '1': [OmeroChannel(label='1_DAPI',
                                             wavelength_id='UV - DAPI'),
                                OmeroChannel(label='1_GFP',
                                             wavelength_id='Blue - FITC'),
                                OmeroChannel(label='1_RFP',
                                             wavelength_id='Green - dsRed'),
                                OmeroChannel(label='1_FR',
                                             wavelength_id='Red - Cy5')
                                ]
                          },
        image_glob_patterns=None,
        num_levels=5,
        coarsening_xy=2,
        image_extension='tif',
        overwrite=True
    )

    IC6000_to_ome_zarr(
        input_paths=[Path(test_data_dir).joinpath("IC6000_data").as_posix()],
        output_path=Path(test_data_dir).joinpath("IC6000_data").as_posix(),
        component="test_plate.zarr/C/03/0",
        metadata={
            'original_paths': [
            Path(test_data_dir).joinpath("IC6000_data", "cycle_0").as_posix(),
            Path(test_data_dir).joinpath("IC6000_data", "cycle_1").as_posix()
            ],
            'image_extension': 'tif',
            'image_glob_patterns': None,
        },
        overwrite=True
    )


def test_multiplexed_pixel_clustering(test_data_dir):
    multiplexed_pixel_clustering(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        label_image_name='Label A',
        channels_to_use=['0_DAPI', '0_GFP', '1_GFP'],
        well_names=['A/2', 'B/3'],
        som_shape=(10, 10),
        phenograph_neighbours=15,
        enforce_equal_object_count=True,
        coords=None,
        level=0,
        output_table_name='mcu_table',
        output_label_name='mcu_label',
        overwrite=True
    )


def test_stitch_fovs_with_overlap(test_data_dir):

    create_ome_zarr_multiplex_IC6000(
        input_paths=[
            Path(test_data_dir).joinpath("IC6000_data/cycle_0").as_posix(),
            Path(test_data_dir).joinpath("IC6000_data/cycle_1").as_posix()],
        output_path=Path(test_data_dir).joinpath("IC6000_data").as_posix(),
        metadata={},
        allowed_channels={'0': [OmeroChannel(label='0_DAPI',
                                             wavelength_id='UV - DAPI'),
                                OmeroChannel(label='0_GFP',
                                             wavelength_id='Blue - FITC'),
                                OmeroChannel(label='0_RFP',
                                             wavelength_id='Green - dsRed'),
                                OmeroChannel(label='0_FR',
                                             wavelength_id='Red - Cy5')],
                          '1': [OmeroChannel(label='1_DAPI',
                                             wavelength_id='UV - DAPI'),
                                OmeroChannel(label='1_GFP',
                                             wavelength_id='Blue - FITC'),
                                OmeroChannel(label='1_RFP',
                                             wavelength_id='Green - dsRed'),
                                OmeroChannel(label='1_FR',
                                             wavelength_id='Red - Cy5')
                                ]
                          },
        image_glob_patterns=None,
        num_levels=5,
        coarsening_xy=2,
        image_extension='tif',
        overwrite=True
    )

    IC6000_to_ome_zarr(
        input_paths=[Path(test_data_dir).joinpath("IC6000_data").as_posix()],
        output_path=Path(test_data_dir).joinpath("IC6000_data").as_posix(),
        component="test_plate.zarr/C/03/0",
        metadata={
            'original_paths': [
                Path(test_data_dir).joinpath("IC6000_data",
                                             "cycle_0").as_posix(),
                Path(test_data_dir).joinpath("IC6000_data",
                                             "cycle_1").as_posix()
            ],
            'image_extension': 'tif',
            'image_glob_patterns': None,
        },
        overwrite=True
    )
    stitch_fovs_with_overlap(
        input_paths=[Path(test_data_dir).joinpath("IC6000_data").as_posix()],
        output_path=Path(test_data_dir).joinpath("IC6000_data").as_posix(),
        component="test_plate.zarr/C/03/0",
        metadata={},
        overlap=0.1,
        filter_sigma=10
    )

#
# def test_ashlar_stitching_and_registration(test_data_dir):
#
#     create_ome_zarr_multiplex_IC6000(
#         input_paths=[
#             Path(test_data_dir).joinpath("IC6000_data/cycle_0").as_posix(),
#             Path(test_data_dir).joinpath("IC6000_data/cycle_1").as_posix()],
#         output_path=Path(test_data_dir).joinpath("IC6000_data").as_posix(),
#         metadata={},
#         allowed_channels={'0': [OmeroChannel(label='0_DAPI',
#                                              wavelength_id='UV - DAPI'),
#                                 OmeroChannel(label='0_GFP',
#                                              wavelength_id='Blue - FITC'),
#                                 OmeroChannel(label='0_RFP',
#                                              wavelength_id='Green - dsRed'),
#                                 OmeroChannel(label='0_FR',
#                                              wavelength_id='Red - Cy5')],
#                           '1': [OmeroChannel(label='1_DAPI',
#                                              wavelength_id='UV - DAPI'),
#                                 OmeroChannel(label='1_GFP',
#                                              wavelength_id='Blue - FITC'),
#                                 OmeroChannel(label='1_RFP',
#                                              wavelength_id='Green - dsRed'),
#                                 OmeroChannel(label='1_FR',
#                                              wavelength_id='Red - Cy5')
#                                 ]
#                           },
#         image_glob_patterns=None,
#         num_levels=5,
#         coarsening_xy=2,
#         image_extension='tif',
#         overwrite=True
#     )
#
#     IC6000_to_ome_zarr(
#         input_paths=[Path(test_data_dir).joinpath("IC6000_data").as_posix()],
#         output_path=Path(test_data_dir).joinpath("IC6000_data").as_posix(),
#         component="test_plate.zarr/C/03/0",
#         metadata={
#             'original_paths': [
#                 Path(test_data_dir).joinpath("IC6000_data",
#                                              "cycle_0").as_posix(),
#                 Path(test_data_dir).joinpath("IC6000_data",
#                                              "cycle_1").as_posix()
#             ],
#             'image_extension': 'tif',
#             'image_glob_patterns': None,
#         },
#         overwrite=True
#     )
#     ashlar_stitching_and_registration(
#         input_paths=[Path(test_data_dir).joinpath("IC6000_data").as_posix()],
#         output_path=Path(test_data_dir).joinpath("IC6000_data").as_posix(),
#         component="test_plate.zarr/C/03/0",
#         metadata={},
#         overlap=0.1,
#         filter_sigma=10,
#         ref_channel_id = 'UV - DAPI',
#         ref_cycle=0
#     )

#
# from apx_fractal_task_collection.tasks.measure_features import measure_features
# measure_features(
#     input_paths=[r"J:\general\20240124_Arpan_4channel_20x_02272024Rad51foci_4cell_line_1\Output_2"],
#     output_path=r"J:\general\20240124_Arpan_4channel_20x_02272024Rad51foci_4cell_line_1\Output_2",
#     component="02272024Rad51foci_4cell_line.zarr/C/03/0",
#     metadata={},
#     label_image_name='Nuclei',
#     channels_to_include=[ChannelInputModel(label='568', wavelength_id=None)],
#     measure_intensity=False,
#     measure_morphology=False,
#     measure_texture=TextureFeatures(
#         haralick=False,
#         laws_texture_energy=True,
#         clip_value=3000,
#         clip_value_exceptions={'0_DAPI': 5000}
#     ),
#     measure_population=False,
#     ROI_table_name='FOV_ROI_table',
#     calculate_internal_borders=False,
#     output_table_name='feature_table',
#     level=0,
#     overwrite=True
# )
