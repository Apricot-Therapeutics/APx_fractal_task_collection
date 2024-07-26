import shutil
from pathlib import Path
import os
import anndata as ad
import numpy as np
import pandas as pd
import dask.array as da

import pytest
from devtools import debug
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.tasks.io_models import MultiplexingAcquisition
#from fractal_tasks_core.tasks.copy_ome_zarr import copy_ome_zarr

from apx_fractal_task_collection.utils import TextureFeatures, FEATURE_LABELS, BaSiCPyModelParams
from apx_fractal_task_collection.tasks.measure_features import measure_features
from apx_fractal_task_collection.tasks.clip_label_image import clip_label_image
from apx_fractal_task_collection.tasks.init_clip_label_image import init_clip_label_image
from apx_fractal_task_collection.tasks.segment_secondary_objects import segment_secondary_objects
from apx_fractal_task_collection.tasks.init_segment_secondary_objects import init_segment_secondary_objects
from apx_fractal_task_collection.tasks.calculate_basicpy_illumination_models import calculate_basicpy_illumination_models
from apx_fractal_task_collection.tasks.init_calculate_basicpy_illumination_models import init_calculate_basicpy_illumination_models
from apx_fractal_task_collection.tasks.apply_basicpy_illumination_models import apply_basicpy_illumination_models
from apx_fractal_task_collection.tasks.convert_channel_to_label import convert_channel_to_label
from apx_fractal_task_collection.tasks.init_convert_channel_to_label import init_convert_channel_to_label
from apx_fractal_task_collection.tasks.filter_label_by_size import filter_label_by_size
from apx_fractal_task_collection.tasks.init_filter_label_by_size import init_filter_label_by_size
from apx_fractal_task_collection.tasks.init_label_assignment_by_overlap import init_label_assignment_by_overlap
from apx_fractal_task_collection.tasks.label_assignment_by_overlap import label_assignment_by_overlap
from apx_fractal_task_collection.tasks.aggregate_feature_tables import aggregate_feature_tables
from apx_fractal_task_collection.tasks.init_aggregate_feature_tables import init_aggregate_feature_tables
from apx_fractal_task_collection.tasks.correct_chromatic_shift import correct_chromatic_shift
from apx_fractal_task_collection.tasks.init_correct_chromatic_shift import init_correct_chromatic_shift
from apx_fractal_task_collection.tasks.compress_zarr_for_visualization import compress_zarr_for_visualization
from apx_fractal_task_collection.tasks.init_convert_IC6000_to_ome_zarr import init_convert_IC6000_to_ome_zarr
from apx_fractal_task_collection.tasks.convert_IC6000_to_ome_zarr import convert_IC6000_to_ome_zarr
from apx_fractal_task_collection.tasks.multiplexed_pixel_clustering import multiplexed_pixel_clustering
from apx_fractal_task_collection.tasks.stitch_fovs_with_overlap import stitch_fovs_with_overlap
from apx_fractal_task_collection.tasks.detect_blob_centroids import detect_blob_centroids
from apx_fractal_task_collection.tasks.init_detect_blob_centroids import init_detect_blob_centroids
from apx_fractal_task_collection.tasks.mask_label_image import mask_label_image
from apx_fractal_task_collection.tasks.init_mask_label_image import init_mask_label_image
from apx_fractal_task_collection.tasks.calculate_registration_image_based_chi_squared_shift import calculate_registration_image_based_chi_squared_shift
from apx_fractal_task_collection.tasks.ashlar_stitching_and_registration import ashlar_stitching_and_registration
from apx_fractal_task_collection.tasks.init_ashlar_stitching_and_registration import init_ashlar_stitching_and_registration
from apx_fractal_task_collection.tasks.init_expand_labels import init_expand_labels
from apx_fractal_task_collection.tasks.expand_labels_skimage import expand_labels_skimage
#from apx_fractal_task_collection.tasks.ashlar_stitching_and_registration_pure import ashlar_stitching_and_registration

WELL_COMPONENT_2D = "hcs_ngff_2D.zarr/A/2"
IMAGE_COMPONENT_2D = "hcs_ngff_2D.zarr/A/2/0"
WELL_COMPONENT_3D = "hcs_ngff_3D.zarr/A/2"
IMAGE_COMPONENT_3D = "hcs_ngff_3D.zarr/A/2/0"

IMAGE_LIST_2D = ["hcs_ngff_2D.zarr/A/2/0",
                 "hcs_ngff_2D.zarr/A/2/1",
                 "hcs_ngff_2D.zarr/A/2/2",
                 "hcs_ngff_2D.zarr/B/3/0",
                 "hcs_ngff_2D.zarr/B/3/1",
                 "hcs_ngff_2D.zarr/B/3/2"]

IMAGE_LIST_3D = ["hcs_ngff_3D.zarr/A/2/0",
                 "hcs_ngff_3D.zarr/A/2/1",
                 "hcs_ngff_3D.zarr/A/2/2",
                 "hcs_ngff_3D.zarr/B/3/0",
                 "hcs_ngff_3D.zarr/B/3/1",
                 "hcs_ngff_3D.zarr/B/3/2"]


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

@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_measure_features(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    measure_features(
        zarr_url=image_list[0],
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
    feature_table_path = Path(image_list[0]).joinpath("tables/feature_table")
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
    if image_list[0] == IMAGE_LIST_3D[0]:
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

@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_clip_label_image(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_clip_label_image(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        label_name='Label A',
        clipping_mask_name='Label D',
        output_label_image_name="0"
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    clip_label_image(
        zarr_url=zarr_url,
        init_args=init_args,
        output_label_name='clipped_label',
        level=0,
        overwrite=True
    )

    # assert whether the clipped label image was created
    clipped_label_path = Path(zarr_url).joinpath("labels/clipped_label/0")
    assert clipped_label_path.exists(),\
        f"Clipped label image not found at {clipped_label_path}"


@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_mask_label_image(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_mask_label_image(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        label_name='Label A',
        mask_name='Label D',
        output_label_image_name="0"
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    mask_label_image(
        zarr_url=zarr_url,
        init_args=init_args,
        output_label_name='masked_label',
        level=0,
        overwrite=True
    )

    # assert whether the clipped label image was created
    masked_label_path = Path(zarr_url).joinpath("labels/masked_label/0")
    assert masked_label_path.exists(),\
        f"Masked label image not found at {masked_label_path}"

@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_segment_secondary_objects(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_segment_secondary_objects(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        channel_label='0_DAPI',
        label_name='Label A',
        output_label_image_name="0",
        mask='Label B',
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    segment_secondary_objects(
        zarr_url=zarr_url,
        init_args=init_args,
        ROI_table_name='FOV_ROI_table',
        min_threshold=10,
        max_threshold=20,
        gaussian_blur=2,
        fill_holes_area=10,
        contrast_threshold=5,
        output_label_name='watershed_result',
        level=0,
        overwrite=True
    )

    # assert whether the label image was created
    label_path = Path(zarr_url).joinpath("labels/watershed_result/0")
    assert label_path.exists(), \
        f"label image not found at {label_path}"


@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_expand_labels(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_expand_labels(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        label_name='Label A',
        output_label_image_name="0",
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    expand_labels_skimage(
        zarr_url=zarr_url,
        init_args=init_args,
        ROI_table_name='FOV_ROI_table',
        distance=10,
        output_label_name='expansion_result',
        level=0,
        overwrite=True,
    )

    # assert whether the label image was created
    label_path = Path(zarr_url).joinpath("labels/expansion_result/0")
    assert label_path.exists(), \
        f"label image not found at {label_path}"

@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_detect_blob_centroids(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_detect_blob_centroids(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        channel_label='0_DAPI',
        output_label_image_name="0",
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    detect_blob_centroids(
        zarr_url=zarr_url,
        init_args=init_args,
        ROI_table_name='FOV_ROI_table',
        min_sigma=1,
        max_sigma=10,
        num_sigma=1,
        threshold=0.002,
        output_label_name='blobs_centroids',
        level=0,
        relabeling=True,
        overwrite=True
    )

    # assert whether the label image was created
    label_path = Path(zarr_url).joinpath("labels/blobs_centroids")
    assert label_path.exists(), \
        f"label image not found at {label_path}"

@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_illumination_correction(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]
    compute_per_well = False

    parallelization_list = init_calculate_basicpy_illumination_models(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        n_images=1,
        compute_per_well=compute_per_well,
    )

    for channel in parallelization_list['parallelization_list']:
        calculate_basicpy_illumination_models(
            zarr_url=channel['zarr_url'],
            init_args=channel['init_args'],
            illumination_profiles_folder=f"{test_data_dir}/illumination_profiles",
            advanced_basicpy_model_params=BaSiCPyModelParams(),
            overwrite=True
        )

    apply_basicpy_illumination_models(
        zarr_url=image_list[0],
        illumination_profiles_folder=f"{test_data_dir}/illumination_profiles",
        illumination_exceptions=["0_DAPI"],
        overwrite_input=False,
    )

    # assert that the illumination corrected image was created
    corrected_image_path = Path(image_list[0]).parent.joinpath("0_illum_corr")
    assert corrected_image_path.exists(),\
        f"Corrected image not found at {corrected_image_path}"

    # assert that illumination profiles folder was created
    illumination_profiles_folder = Path(f"{test_data_dir}/illumination_profiles")
    assert illumination_profiles_folder.exists(),\
        f"Illumination profiles not found at {illumination_profiles_folder}"

    # assert that illumination profiles folder contains the expected subfolders

    if compute_per_well:
        expected_subfolders = ['well_A2_ch_lbl_2_DAPI',
                               'well_A2_ch_lbl_0_GFP',
                               'well_A2_ch_lbl_0_DAPI',
                               'well_A2_ch_lbl_1_GFP',
                               'well_A2_ch_lbl_1_DAPI',
                               'well_B3_ch_lbl_2_GFP',
                               'well_B3_ch_lbl_1_DAPI',
                               'well_B3_ch_lbl_1_GFP',
                               'well_A2_ch_lbl_2_GFP',
                               'well_B3_ch_lbl_0_GFP',
                               'well_B3_ch_lbl_2_DAPI',
                               'well_B3_ch_lbl_0_DAPI']
    else:
        expected_subfolders = ['0_DAPI', '0_GFP',
                               '1_DAPI', '1_GFP',
                               '2_DAPI', '2_GFP']


    subfolders = [f.name for f in illumination_profiles_folder.iterdir()]
    assert sorted(subfolders) == sorted(expected_subfolders), \
        f"Expected subfolders {expected_subfolders}," \
        f" but got {subfolders}"


@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_convert_channel_to_label(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_convert_channel_to_label(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        channel_label='0_DAPI',
        output_label_image_name="0",
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    convert_channel_to_label(
        zarr_url=zarr_url,
        init_args=init_args,
        output_label_name='DAPI',
        overwrite=True
    )

    # assert whether the label image was created
    label_path = Path(zarr_url).joinpath("labels/DAPI")
    assert label_path.exists(), \
        f"label image not found at {label_path}"


@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_filter_label_by_size(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_filter_label_by_size(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        label_name='Label A',
        output_label_image_name="0"
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    filter_label_by_size(
        zarr_url=zarr_url,
        init_args=init_args,
        output_label_name='filtered_label',
        min_size=10,
        max_size=50,
        level=0,
        overwrite=True
    )

    # assert whether the clipped label image was created
    filtered_label_path = Path(zarr_url).joinpath("labels/filtered_label/0")
    assert filtered_label_path.exists(),\
        f"Clipped label image not found at {filtered_label_path}"

    # assert whether filtered label image contains fewer objects
    label = da.from_zarr(Path(
        init_args["label_zarr_url"]).joinpath("labels/Label A/0"))
    filtered_label = da.from_zarr(filtered_label_path)

    print(label.max(), filtered_label.max())

    assert filtered_label.max() < label.max(), \
        (f"Filtered label image does not contain less objects "
         f"than original label image")

@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_label_assignment_by_overlap(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]
    image_list = [i for i in image_list if "A/2" in i]
    zarr_url = image_list[0]
    # create a feature table
    measure_features(
        zarr_url=zarr_url,
        label_image_name='Label B',
        measure_intensity=True,
        measure_morphology=True,
        channels_to_include=None,
        channels_to_exclude=[
            ChannelInputModel(label='0_GFP', wavelength_id=None)],
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

    child_table = ad.read_zarr(Path(zarr_url).joinpath(
        "tables/feature_table"))
    old_obs_columns = list(child_table.obs.columns)

    parallelization_list = init_label_assignment_by_overlap(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        parent_label_name="Label A",
        child_label_name="Label B",
    )

    new_zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0][
        'init_args']

    label_assignment_by_overlap(
        zarr_url=new_zarr_url,
        init_args=init_args,
        child_table_name='feature_table',
        level=0,
        overlap_threshold=0.6,
    )

    child_table = ad.read_zarr(Path(zarr_url).joinpath(
        "tables/feature_table"))

    # assert that child_table.obs contains the expected columns
    expected_obs_columns = old_obs_columns + \
                           ['Label A_label', 'Label B_Label A_overlap']
    new_obs_columns = list(child_table.obs.columns)
    assert new_obs_columns == expected_obs_columns, \
        f"Expected obs columns {expected_obs_columns}," \
        f" but got {new_obs_columns}"

    # assert whether label assignments are correct
    if zarr_url.split("/data/")[1] == IMAGE_LIST_2D[0]:
        label_assignments = {9: 6,
                             14: 6,
                             31: 19,
                             38: 19,
                             36: pd.NA,
                             20: pd.NA,
                             12: 8}
    elif zarr_url.split("/data/")[1] == IMAGE_LIST_3D[0]:
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

@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_aggregate_tables_to_well_level(test_data_dir, image_list):
    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    measure_features(
        zarr_url=image_list[0],
        label_image_name='Label B',
        measure_intensity=True,
        measure_morphology=True,
        channels_to_include=None,
        channels_to_exclude=[
            ChannelInputModel(label='0_GFP', wavelength_id=None)],
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

    measure_features(
        zarr_url=image_list[1],
        label_image_name='Label B',
        measure_intensity=True,
        measure_morphology=True,
        channels_to_include=None,
        channels_to_exclude=[
            ChannelInputModel(label='0_GFP', wavelength_id=None)],
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

    measure_features(
        zarr_url=image_list[2],
        label_image_name='Label B',
        measure_intensity=True,
        measure_morphology=True,
        channels_to_include=None,
        channels_to_exclude=[
            ChannelInputModel(label='0_GFP', wavelength_id=None)],
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

    parallelization_list = init_aggregate_feature_tables(
        zarr_urls=image_list[0:3],
        zarr_dir=test_data_dir,
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    aggregate_feature_tables(
        zarr_url=zarr_url,
        init_args=init_args,
        input_table_name='feature_table',
        output_table_name='feature_table',
        output_image='2',
        overwrite=True
    )

    # assert that the aggregated feature table exists in correct location
    feature_table_path = Path(zarr_url).joinpath("tables/feature_table")
    assert feature_table_path.exists(),\
        f"Feature table not found at {feature_table_path}"

@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_correct_chromatic_shift(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_correct_chromatic_shift(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        reference_zarr_image="0",
        reference_channel_label='0_DAPI',
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']
    print(f"zarr_url is {zarr_url}")
    correct_chromatic_shift(
        zarr_url=zarr_url,
        init_args=init_args,
        overwrite_input=False,
    )

    Path(os.getcwd()).joinpath("TransformParameters.0.txt").unlink(
        missing_ok=True)

    # assert that the corrected image was created
    corrected_image_path = Path(zarr_url).parent.joinpath("0_chromatic_shift_corr")
    assert corrected_image_path.exists(),\
        f"Corrected image not found at {corrected_image_path}"


@pytest.mark.parametrize("component", [IMAGE_COMPONENT_2D, IMAGE_COMPONENT_3D])
def test_registration(test_data_dir, component):
    calculate_registration_image_based_chi_squared_shift(
        input_paths=[test_data_dir],
        output_path=test_data_dir,
        metadata={},
        component=component,
        wavelength_id='A01_C01',
        roi_table='FOV_ROI_table',
        level=0,
    )


# @pytest.mark.parametrize("component", [WELL_COMPONENT_2D, WELL_COMPONENT_3D])
# def test_compress_zarr_for_visualization(test_data_dir, component):
#
#     copy_ome_zarr(
#         input_paths=[test_data_dir],
#         output_path=test_data_dir,
#         metadata={'plate': 'hcs_ngff',
#                   'well': ['A/2', 'B/3'],
#                   'image': ['0', '1', '2']},
#         project_to_2D=False,
#         suffix="vis",
#         ROI_table_names=("well_ROI_table", "FOV_ROI_table")
#     )
#
#     compress_zarr_for_visualization(
#         input_paths=[test_data_dir],
#         output_path=test_data_dir,
#         metadata={'plate': 'hcs_ngff',
#                   'copy_ome_zarr': {'suffix': 'vis'}},
#         component=component,
#         output_zarr_path=Path(test_data_dir).joinpath(
#             "hcs_ngff_vis.zarr").as_posix(),
#         overwrite=True
#     )

def test_IC6000_conversion(test_data_dir):

    parallelization_list = init_convert_IC6000_to_ome_zarr(
        zarr_urls=[],
        zarr_dir=test_data_dir,
        acquisitions={"0":
                          MultiplexingAcquisition(
                              image_dir=Path(test_data_dir).joinpath("IC6000_data/cycle_0").as_posix(),
                              allowed_channels=[
                                  OmeroChannel(label='0_DAPI',
                                               wavelength_id='UV - DAPI'),
                                  OmeroChannel(label='0_GFP',
                                               wavelength_id='Blue - FITC'),
                                  OmeroChannel(label='0_RFP',
                                               wavelength_id='Green - dsRed'),
                                  OmeroChannel(label='0_FR',
                                               wavelength_id='Red - Cy5')]),
                      "1":
                          MultiplexingAcquisition(
                              image_dir=Path(test_data_dir).joinpath("IC6000_data/cycle_1").as_posix(),
                              allowed_channels=[
                                  OmeroChannel(label='1_DAPI',
                                               wavelength_id='UV - DAPI'),
                                  OmeroChannel(label='1_GFP',
                                               wavelength_id='Blue - FITC'),
                                  OmeroChannel(label='1_RFP',
                                               wavelength_id='Green - dsRed'),
                                  OmeroChannel(label='1_FR',
                                               wavelength_id='Red - Cy5')])},
        image_glob_patterns=None,
        num_levels=5,
        coarsening_xy=2,
        image_extension='tif',
        overwrite=True,
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    convert_IC6000_to_ome_zarr(
        zarr_url=zarr_url,
        init_args=init_args,
    )


@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_multiplexed_pixel_clustering(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    multiplexed_pixel_clustering(
        zarr_urls=image_list,
        label_image_name='Label A',
        channels_to_use=['0_DAPI', '0_GFP', '1_GFP'],
        well_names=['A2', 'B3'],
        som_shape=(10, 10),
        phenograph_neighbours=15,
        enforce_equal_object_count=True,
        coords=None,
        level=0,
        output_table_name='mcu_table',
        output_label_name='mcu_label',
        overwrite=True
    )

    # assert that output table was created
    mcu_table_path = Path(image_list[0]).parents[2].joinpath("tables/mcu_table")
    assert mcu_table_path.exists(),\
        f"MCU table not found at {mcu_table_path}"

    # assert that the mcu label image was created
    mcu_label_path = Path(image_list[0]).parent.joinpath("0/labels/mcu_label/0")
    assert mcu_label_path.exists(),\
        f"MCU label image not found at {mcu_label_path}"


def test_stitch_fovs_with_overlap(test_data_dir):

    parallelization_list = init_convert_IC6000_to_ome_zarr(
        zarr_urls=[],
        zarr_dir=test_data_dir,
        acquisitions={"0":
            MultiplexingAcquisition(
                image_dir=Path(test_data_dir).joinpath(
                    "IC6000_data/cycle_0").as_posix(),
                allowed_channels=[
                    OmeroChannel(label='0_DAPI',
                                 wavelength_id='UV - DAPI'),
                    OmeroChannel(label='0_GFP',
                                 wavelength_id='Blue - FITC'),
                    OmeroChannel(label='0_RFP',
                                 wavelength_id='Green - dsRed'),
                    OmeroChannel(label='0_FR',
                                 wavelength_id='Red - Cy5')]),
            "1":
                MultiplexingAcquisition(
                    image_dir=Path(test_data_dir).joinpath(
                        "IC6000_data/cycle_1").as_posix(),
                    allowed_channels=[
                        OmeroChannel(label='1_DAPI',
                                     wavelength_id='UV - DAPI'),
                        OmeroChannel(label='1_GFP',
                                     wavelength_id='Blue - FITC'),
                        OmeroChannel(label='1_RFP',
                                     wavelength_id='Green - dsRed'),
                        OmeroChannel(label='1_FR',
                                     wavelength_id='Red - Cy5')])},
        image_glob_patterns=None,
        num_levels=5,
        coarsening_xy=2,
        image_extension='tif',
        overwrite=True,
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    convert_IC6000_to_ome_zarr(
        zarr_url=zarr_url,
        init_args=init_args,
    )

    stitch_fovs_with_overlap(
        zarr_url=f"{test_data_dir}/test_plate.zarr/C/03/0",
        overlap=0.1,
        filter_sigma=10,
        overwrite_input=False,
        tmp_dir=None,
    )

    # assert whether stitched images was saved in a new zarr image
    stitched_image_path = Path(test_data_dir).joinpath(
        "test_plate.zarr/C/03/0_stitched")
    assert stitched_image_path.exists(),\
        f"Stitched image not found at {stitched_image_path}"

@pytest.mark.parametrize("ref_wavelength_id", ['UV - DAPI', 'Red - Cy5'])
def test_ashlar_stitching_and_registration(test_data_dir, ref_wavelength_id):

    parallelization_list = init_convert_IC6000_to_ome_zarr(
        zarr_urls=[],
        zarr_dir=test_data_dir,
        acquisitions={"0":
            MultiplexingAcquisition(
                image_dir=Path(test_data_dir).joinpath(
                    "IC6000_data/cycle_0").as_posix(),
                allowed_channels=[
                    OmeroChannel(label='0_DAPI',
                                 wavelength_id='UV - DAPI'),
                    OmeroChannel(label='0_GFP',
                                 wavelength_id='Blue - FITC'),
                    OmeroChannel(label='0_RFP',
                                 wavelength_id='Green - dsRed'),
                    OmeroChannel(label='0_FR',
                                 wavelength_id='Red - Cy5')]),
            "1":
                MultiplexingAcquisition(
                    image_dir=Path(test_data_dir).joinpath(
                        "IC6000_data/cycle_1").as_posix(),
                    allowed_channels=[
                        OmeroChannel(label='1_DAPI',
                                     wavelength_id='UV - DAPI'),
                        OmeroChannel(label='1_GFP',
                                     wavelength_id='Blue - FITC'),
                        OmeroChannel(label='1_RFP',
                                     wavelength_id='Green - dsRed'),
                        OmeroChannel(label='1_FR',
                                     wavelength_id='Red - Cy5')])},
        image_glob_patterns=None,
        num_levels=5,
        coarsening_xy=2,
        image_extension='tif',
        overwrite=True,
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    print(parallelization_list['parallelization_list'])

    convert_IC6000_to_ome_zarr(
        zarr_url=zarr_url,
        init_args=init_args,
    )

    zarr_url = parallelization_list['parallelization_list'][2]['zarr_url']
    init_args = parallelization_list['parallelization_list'][2]['init_args']

    print(parallelization_list['parallelization_list'])

    convert_IC6000_to_ome_zarr(
        zarr_url=zarr_url,
        init_args=init_args,
    )

    parallelization_list = init_ashlar_stitching_and_registration(
        zarr_dir=test_data_dir,
        zarr_urls=[f"{test_data_dir}/test_plate.zarr/C/03/0",
                   f"{test_data_dir}/test_plate.zarr/C/03/1"]
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']

    ashlar_stitching_and_registration(
        zarr_url=zarr_url,
        init_args=init_args,
        overlap=0.1,
        filter_sigma=10,
        tmp_dir=None,
        overwrite_input=False,
        suffix="_stitched",
        ref_wavelength_id=ref_wavelength_id,
        ref_cycle=0,
    )

    # assert whether stitched images was saved in a new zarr image
    stitched_image_path = Path(test_data_dir).joinpath(
        "test_plate.zarr/C/03/0_stitched")
    assert stitched_image_path.exists(),\
        f"Stitched image not found at {stitched_image_path}"

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
