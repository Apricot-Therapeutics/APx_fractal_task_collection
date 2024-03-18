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

from apx_fractal_task_collection.utils import TextureFeatures
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


def test_measure_features(test_data_dir):
    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=IMAGE_COMPONENT,
        metadata={},
        label_image_name='Label A',
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
        overwrite=True,
    )

    # assert that the feature table exists in correct location
    feature_table_path = Path(test_data_dir).joinpath(
        IMAGE_COMPONENT,
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
    morphology_labels = [
        'Morphology_area',
        'Morphology_centroid-0',
        'Morphology_centroid-1',
        'Morphology_well_centroid-0',
        'Morphology_well_centroid-1',
        'Morphology_bbox_area',
        'Morphology_bbox-0',
        'Morphology_bbox-1',
        'Morphology_bbox-2',
        'Morphology_bbox-3',
        'Morphology_convex_area',
        'Morphology_eccentricity',
        'Morphology_equivalent_diameter',
        'Morphology_euler_number',
        'Morphology_extent',
        'Morphology_filled_area',
        'Morphology_major_axis_length',
        'Morphology_minor_axis_length',
        'Morphology_orientation',
        'Morphology_perimeter',
        'Morphology_solidity',
        'Morphology_roundness',
        'Morphology_circularity'
    ]

    intensity_labels = [
        'Intensity_max_intensity',
        'Intensity_mean_intensity',
        'Intensity_min_intensity',
        'Intensity_weighted_moments_hu-0',
        'Intensity_weighted_moments_hu-1',
        'Intensity_weighted_moments_hu-2',
        'Intensity_weighted_moments_hu-3',
        'Intensity_weighted_moments_hu-4',
        'Intensity_weighted_moments_hu-5',
        'Intensity_weighted_moments_hu-6',
        'Intensity_sum_intensity',
        'Intensity_std_intensity'
    ]

    texture_labels = [
        'Texture_Haralick-Mean-angular-second-moment-2',
        'Texture_Haralick-Mean-contrast-2',
        'Texture_Haralick-Mean-correlation-2',
        'Texture_Haralick-Mean-sum-of-squares-2',
        'Texture_Haralick-Mean-inverse-diff-moment-2',
        'Texture_Haralick-Mean-sum-avg-2',
        'Texture_Haralick-Mean-sum-var-2',
        'Texture_Haralick-Mean-sum-entropy-2',
        'Texture_Haralick-Mean-entropy-2',
        'Texture_Haralick-Mean-diff-var-2',
        'Texture_Haralick-Mean-diff-entropy-2',
        'Texture_Haralick-Mean-info-measure-corr-1-2',
        'Texture_Haralick-Mean-info-measure-corr-2-2',
        'Texture_Haralick-Range-angular-second-moment-2',
        'Texture_Haralick-Range-contrast-2',
        'Texture_Haralick-Range-correlation-2',
        'Texture_Haralick-Range-sum-of-squares-2',
        'Texture_Haralick-Range-inverse-diff-moment-2',
        'Texture_Haralick-Range-sum-avg-2',
        'Texture_Haralick-Range-sum-var-2',
        'Texture_Haralick-Range-sum-entropy-2',
        'Texture_Haralick-Range-entropy-2',
        'Texture_Haralick-Range-diff-var-2',
        'Texture_Haralick-Range-diff-entropy-2',
        'Texture_Haralick-Range-info-measure-corr-1-2',
        'Texture_Haralick-Range-info-measure-corr-2-2',
        'Texture_Haralick-Mean-angular-second-moment-5',
        'Texture_Haralick-Mean-contrast-5',
        'Texture_Haralick-Mean-correlation-5',
        'Texture_Haralick-Mean-sum-of-squares-5',
        'Texture_Haralick-Mean-inverse-diff-moment-5',
        'Texture_Haralick-Mean-sum-avg-5',
        'Texture_Haralick-Mean-sum-var-5',
        'Texture_Haralick-Mean-sum-entropy-5',
        'Texture_Haralick-Mean-entropy-5',
        'Texture_Haralick-Mean-diff-var-5',
        'Texture_Haralick-Mean-diff-entropy-5',
        'Texture_Haralick-Mean-info-measure-corr-1-5',
        'Texture_Haralick-Mean-info-measure-corr-2-5',
        'Texture_Haralick-Range-angular-second-moment-5',
        'Texture_Haralick-Range-contrast-5',
        'Texture_Haralick-Range-correlation-5',
        'Texture_Haralick-Range-sum-of-squares-5',
        'Texture_Haralick-Range-inverse-diff-moment-5',
        'Texture_Haralick-Range-sum-avg-5',
        'Texture_Haralick-Range-sum-var-5',
        'Texture_Haralick-Range-sum-entropy-5',
        'Texture_Haralick-Range-entropy-5',
        'Texture_Haralick-Range-diff-var-5',
        'Texture_Haralick-Range-diff-entropy-5',
        'Texture_Haralick-Range-info-measure-corr-1-5',
        'Texture_Haralick-Range-info-measure-corr-2-5',
        'Texture_LTE_LL',
        'Texture_LTE_EE',
        'Texture_LTE_SS',
        'Texture_LTE_LE',
        'Texture_LTE_ES',
        'Texture_LTE_LS'
    ]

    population_labels = [
        'Population_density_bw_0.01',
        'Population_density_bw_0.02',
        'Population_density_bw_0.03',
        'Population_density_bw_0.04',
        'Population_density_bw_0.05',
        'Population_density_bw_0.2',
        'Population_density_bw_0.5',
        'Population_density_bw_1.0',
        'Population_mean_distance_nn_5',
        'Population_mean_distance_nn_10',
        'Population_mean_distance_nn_25',
        #'Population_mean_distance_nn_50',
        #'Population_mean_distance_nn_100',
        'Population_n_neighbours_radius_100',
        'Population_mean_distance_neighbours_radius_100',
        'Population_n_neighbours_radius_200',
        'Population_mean_distance_neighbours_radius_200',
        'Population_n_neighbours_radius_300',
        'Population_mean_distance_neighbours_radius_300',
        'Population_n_neighbours_radius_400',
        'Population_mean_distance_neighbours_radius_400',
        'Population_n_neighbours_radius_500',
        'Population_mean_distance_neighbours_radius_500'
    ]

    channels = ['0_DAPI', '0_GFP']

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

    assert feature_table.var.index.tolist() == expected_columns, \
        f"Expected columns {expected_columns}," \
        f" but got {feature_table.var.index.tolist()}"


def test_clip_label_image(test_data_dir):
    clip_label_image(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=WELL_COMPONENT,
        metadata={},
        label_image_name='Label A',
        clipping_mask_name='Label D',
        output_label_cycle=0,
        output_label_name='clipped_label',
        level=0,
        overwrite=True
    )


def test_apply_mask(test_data_dir):
    apply_mask(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=WELL_COMPONENT,
        metadata={},
        label_image_name='Label A',
        mask_label_name='Label D',
        output_label_cycle=0,
        output_label_name='masked_label',
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


def test_detect_blob_centroids(test_data_dir):
    detect_blob_centroids(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=WELL_COMPONENT,
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


def test_convert_channel_to_label(test_data_dir):
    convert_channel_to_label(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=WELL_COMPONENT,
        metadata={},
        channel_label='0_DAPI',
        output_label_name='DAPI',
        output_cycle=0,
        overwrite=True
    )


def test_filter_label_by_size(test_data_dir):
    filter_label_by_size(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        component=WELL_COMPONENT,
        label_name='Label A',
        output_label_name='filtered_label',
        min_size=10,
        max_size=100,
        overwrite=True
    )


def test_label_assignment_by_overlap(test_data_dir):

    # create a feature table
    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=IMAGE_COMPONENT,
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
        WELL_COMPONENT,
        "0/tables/feature_table")
    )
    old_obs_columns = list(child_table.obs.columns)

    label_assignment_by_overlap(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        component=WELL_COMPONENT,
        parent_label_image='Label A',
        child_label_image='Label B',
        child_table_name='feature_table',
        level=0,
        overlap_threshold=0.6,
    )

    child_table = ad.read_zarr(Path(test_data_dir).joinpath(
        WELL_COMPONENT,
        "0/tables/feature_table")
    )

    # assert that child_table.obs contains the expected columns
    expected_obs_columns = old_obs_columns + \
                           ['Label A_label', 'Label B_Label A_overlap']
    new_obs_columns = list(child_table.obs.columns)
    assert new_obs_columns == expected_obs_columns, \
        f"Expected obs columns {expected_obs_columns}," \
        f" but got {new_obs_columns}"

    # assert whether label assignments are correct
    label_assignments = {9: 6,
                         14: 6,
                         31: 19,
                         38: 19,
                         36: pd.NA,
                         20: pd.NA,
                         12: 8}

    for child_label, parent_label in label_assignments.items():
        value = child_table.obs.loc[
            child_table.obs.label == child_label, 'Label A_label'].values[0]
        print(value)
        try:
            assert value == parent_label, \
                f"Label assignment failed, expected {parent_label}," \
                f" but got {value}"
        except TypeError:
            assert pd.isna(value), \
                f"Label assignment failed, expected {parent_label}," \
                f" but got {value}"


def test_aggregate_tables_to_well_level(test_data_dir):

    # create a feature table
    measure_features(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        component=IMAGE_COMPONENT,
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
        component=Path(IMAGE_COMPONENT).parent.joinpath('1').as_posix(),
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
        component=Path(IMAGE_COMPONENT).parent.joinpath('2').as_posix(),
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
        component=WELL_COMPONENT,
        input_table_name='feature_table',
        output_table_name='feature_table',
        overwrite=True
    )


def test_chromatic_shift_correction(test_data_dir):
    chromatic_shift_correction(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        component=IMAGE_COMPONENT,
        correction_channel_labels=['0_DAPI', '0_GFP'],
        reference_channel_label='0_DAPI'
    )

    Path(os.getcwd()).joinpath("TransformParameters.0.txt").unlink(
        missing_ok=True)


def test_compress_zarr_for_visualization(test_data_dir):

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
        component=WELL_COMPONENT,
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


# from apx_fractal_task_collection.tasks.measure_features import measure_features
# measure_features(
#     input_paths=[r"J:\general\20240124_Arpan_4channel_20x_02272024Rad51foci_4cell_line_1\Output_2"],
#     output_path=r"J:\general\20240124_Arpan_4channel_20x_02272024Rad51foci_4cell_line_1\Output_2",
#     component="02272024Rad51foci_4cell_line.zarr/C/04/0",
#     metadata={},
#     label_image_name='Nuclei',
#     measure_intensity=False,
#     measure_morphology=True,
#     measure_texture=False,
#     measure_population=True,
#     ROI_table_name='well_ROI_table',
#     calculate_internal_borders=True,
#     output_table_name='feature_table',
#     level=0,
#     overwrite=True
# )
