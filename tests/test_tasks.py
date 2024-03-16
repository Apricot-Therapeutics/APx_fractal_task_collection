import shutil
from pathlib import Path
import os

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