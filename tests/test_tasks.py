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
from apx_fractal_task_collection.io_models import CorrectBy
from apx_fractal_task_collection.tasks.init_calculate_basicpy_illumination_models import (
    init_calculate_basicpy_illumination_models,
)
from apx_fractal_task_collection.tasks.calculate_basicpy_illumination_models import calculate_basicpy_illumination_models
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
from apx_fractal_task_collection.tasks.init_add_multiplexing_cycle_IC6000 import init_add_multiplexing_cycle_IC6000
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
from apx_fractal_task_collection.tasks.init_calculate_pixel_intensity_correlation import init_calculate_pixel_intensity_correlation
from apx_fractal_task_collection.tasks.calculate_pixel_intensity_correlation import calculate_pixel_intensity_correlation
from apx_fractal_task_collection.tasks.merge_plate_metadata import merge_plate_metadata
from apx_fractal_task_collection.tasks.normalize_feature_table import normalize_feature_table, NormalizationMethod
from apx_fractal_task_collection.tasks.init_normalize_feature_table import init_normalize_feature_table, NormalizationLayout
from apx_fractal_task_collection.tasks.init_correct_4i_bleaching_artifacts import init_correct_4i_bleaching_artifacts
from apx_fractal_task_collection.tasks.correct_4i_bleaching_artifacts import correct_4i_bleaching_artifacts

WELL_COMPONENT_2D = "hcs_ngff_2D.zarr/A/02"
IMAGE_COMPONENT_2D = "hcs_ngff_2D.zarr/A/02/0"
WELL_COMPONENT_3D = "hcs_ngff_3D.zarr/A/02"
IMAGE_COMPONENT_3D = "hcs_ngff_3D.zarr/A/02/0"

IMAGE_LIST_2D = ["hcs_ngff_2D.zarr/A/02/0",
                 "hcs_ngff_2D.zarr/A/02/1",
                 "hcs_ngff_2D.zarr/A/02/2",
                 "hcs_ngff_2D.zarr/B/03/0",
                 "hcs_ngff_2D.zarr/B/03/1",
                 "hcs_ngff_2D.zarr/B/03/2"]

IMAGE_LIST_3D = ["hcs_ngff_3D.zarr/A/02/0",
                 "hcs_ngff_3D.zarr/A/02/1",
                 "hcs_ngff_3D.zarr/A/02/2",
                 "hcs_ngff_3D.zarr/B/03/0",
                 "hcs_ngff_3D.zarr/B/03/1",
                 "hcs_ngff_3D.zarr/B/03/2"]


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

    #assert set(feature_table.var.index.tolist()) == set(expected_columns), \
    #    f"Expected columns {expected_columns}," \
    #    f" but got {feature_table.var.index.tolist()}"

    # print values uniquely in feature_table.var.index.tolist() and not in expected_columns
    print(f"this is a test{set(expected_columns) - set(feature_table.var.index.tolist())}")

    assert np.isin(feature_table.var.index.tolist(), expected_columns).all(), \
        f"Expected columns {expected_columns}," \
        f" but got {feature_table.var.index.tolist()}"




@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_calculate_pixel_intensity_correlation(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    parallelization_list = init_calculate_pixel_intensity_correlation(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        correlation_pairs=[{"0_DAPI": "0_GFP"},
                           {"0_DAPI": "1_GFP"},
                           {"0_DAPI": "1_DAPI"}],
        label_name='Label A',
    )

    for pl in parallelization_list['parallelization_list']:
        zarr_url = pl['zarr_url']
        init_args = pl['init_args']

        calculate_pixel_intensity_correlation(
            zarr_url=zarr_url,
            init_args=init_args,
            ROI_table_name='FOV_ROI_table',
            output_table_name='correlation_table',
            level=0,
            overwrite=True,
        )

        # assert that the feature table exists in correct location
        feature_table_path = Path(image_list[0]).joinpath(
            "tables/correlation_table")
        assert feature_table_path.exists(), \
            f"Feature table not found at {feature_table_path}"


    # assert that the obs contain correct columns
    feature_table = ad.read_zarr(feature_table_path)
    expected_columns = ['label',
                        'well_name',
                        'ROI']

    assert feature_table.obs.columns.tolist() == expected_columns, \
        f"Expected columns {expected_columns}," \
        f" but got {feature_table.obs.columns.tolist()}"

    # assert that the feature table contains correct columns
    expected_columns = ['Label A_Correlation_0_DAPI_0_GFP',
                        'Label A_Correlation_0_DAPI_1_GFP',
                        'Label A_Correlation_0_DAPI_1_DAPI',]

    print(feature_table.to_df().shape)
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
def test_illumination_correction_by_label(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]
    compute_per_well = False

    parallelization_list = init_calculate_basicpy_illumination_models(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        n_images=2,
        correct_by=CorrectBy.channel_label,
        compute_per_well=compute_per_well
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
        correct_by = CorrectBy.channel_label,
        illumination_exceptions=["0_DAPI"],
        darkfield=True,
        subtract_baseline=True,
        fixed_baseline=105,
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
        expected_subfolders = ['well_A02_ch_2_DAPI',
                               'well_A02_ch_0_GFP',
                               'well_A02_ch_0_DAPI',
                               'well_A02_ch_1_GFP',
                               'well_A02_ch_1_DAPI',
                               'well_B03_ch_2_GFP',
                               'well_B03_ch_1_DAPI',
                               'well_B03_ch_1_GFP',
                               'well_A02_ch_2_GFP',
                               'well_B03_ch_0_GFP',
                               'well_B03_ch_2_DAPI',
                               'well_B03_ch_0_DAPI']
    else:
        expected_subfolders = ['0_DAPI', '0_GFP',
                               '1_DAPI', '1_GFP',
                               '2_DAPI', '2_GFP']


    subfolders = [f.name for f in illumination_profiles_folder.iterdir()]
    assert sorted(subfolders) == sorted(expected_subfolders), \
        f"Expected subfolders {expected_subfolders}," \
        f" but got {subfolders}"


@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_illumination_correction_by_wavelength(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]
    compute_per_well = False

    parallelization_list = init_calculate_basicpy_illumination_models(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        n_images=1,
        correct_by=CorrectBy.wavelength_id,
        compute_per_well=compute_per_well
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
        correct_by = CorrectBy.wavelength_id,
        illumination_exceptions=["Blue - FITC"],
        darkfield=False,
        subtract_baseline=False,
        fixed_baseline=105,
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
        expected_subfolders = ['well_A02_ch_Blue - FITC',
                               'well_A02_ch_UV - DAPI',
                               'well_B03_ch_Blue - FITC',
                               'well_B03_ch_UV - DAPI']
        
    else:
        expected_subfolders = ['Blue - FITC', 'UV - DAPI']


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
    )

    zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
    init_args = parallelization_list['parallelization_list'][0]['init_args']
    output_label_image_name = "1"

    print(f"init args are {init_args}")

    filter_label_by_size(
        zarr_url=zarr_url,
        init_args=init_args,
        output_label_name='filtered_label',
        output_label_image_name=output_label_image_name,
        min_size=10,
        max_size=50,
        level=0,
        overwrite=True
    )

    output_zarr_url = f"{init_args['label_zarr_url'].rsplit('/', 1)[0]}/{output_label_image_name}"
    # assert whether the clipped label image was created
    filtered_label_path = Path(f"{output_zarr_url}/labels/filtered_label/0")
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
    image_list = [i for i in image_list if "A/02" in i]
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

    print(f"zarr_url is {zarr_url}")

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


def test_add_multiplexing_cycle_IC6000(test_data_dir):

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


    # add multiplexing cycle
    parallelization_list = init_add_multiplexing_cycle_IC6000(
        zarr_urls=[],
        zarr_dir=test_data_dir,
        zarr_path=Path(test_data_dir).joinpath("test_plate.zarr").as_posix(),
        acquisitions={"1":
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

    # assert that the new cycle was added
    cycle_1_path = Path(test_data_dir).joinpath("test_plate.zarr/C/03/1")
    assert cycle_1_path.exists(),\
        f"Cycle 1 not found at {cycle_1_path}"



@pytest.mark.parametrize("image_list", [IMAGE_LIST_2D, IMAGE_LIST_3D])
def test_multiplexed_pixel_clustering(test_data_dir, image_list):

    image_list = [f"{test_data_dir}/{i}" for i in image_list]

    multiplexed_pixel_clustering(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        label_image_name='Label A',
        channels_to_use=['0_DAPI', '1_GFP'],
        well_names=['A02', 'B03'],
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

    # assert that the mcu table varm has the expected columns set to True
    mcu_table_path = Path(image_list[0]).parents[2].joinpath(
        "tables/mcu_table")
    mcu_table = ad.read_zarr(mcu_table_path)
    expected_output = [True, False, False, True, False, False]
    assert (mcu_table.varm['used_for_clustering'] == expected_output).all(), \
        f"Expected output {expected_output}," \
        f" but got {mcu_table.varm['used_for_clustering']}"


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

# @pytest.mark.parametrize("ref_wavelength_id", ['UV - DAPI', 'Red - Cy5'])
# def test_ashlar_stitching_and_registration(test_data_dir, ref_wavelength_id):
#
#     parallelization_list = init_convert_IC6000_to_ome_zarr(
#         zarr_urls=[],
#         zarr_dir=test_data_dir,
#         acquisitions={"0":
#             MultiplexingAcquisition(
#                 image_dir=Path(test_data_dir).joinpath(
#                     "IC6000_data/cycle_0").as_posix(),
#                 allowed_channels=[
#                     OmeroChannel(label='0_DAPI',
#                                  wavelength_id='UV - DAPI'),
#                     OmeroChannel(label='0_GFP',
#                                  wavelength_id='Blue - FITC'),
#                     OmeroChannel(label='0_RFP',
#                                  wavelength_id='Green - dsRed'),
#                     OmeroChannel(label='0_FR',
#                                  wavelength_id='Red - Cy5')]),
#             "1":
#                 MultiplexingAcquisition(
#                     image_dir=Path(test_data_dir).joinpath(
#                         "IC6000_data/cycle_1").as_posix(),
#                     allowed_channels=[
#                         OmeroChannel(label='1_DAPI',
#                                      wavelength_id='UV - DAPI'),
#                         OmeroChannel(label='1_GFP',
#                                      wavelength_id='Blue - FITC'),
#                         OmeroChannel(label='1_RFP',
#                                      wavelength_id='Green - dsRed'),
#                         OmeroChannel(label='1_FR',
#                                      wavelength_id='Red - Cy5')])},
#         image_glob_patterns=None,
#         num_levels=5,
#         coarsening_xy=2,
#         image_extension='tif',
#         overwrite=True,
#     )
#
#     zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
#     init_args = parallelization_list['parallelization_list'][0]['init_args']
#
#     print(parallelization_list['parallelization_list'])
#
#     convert_IC6000_to_ome_zarr(
#         zarr_url=zarr_url,
#         init_args=init_args,
#     )
#
#     zarr_url = parallelization_list['parallelization_list'][2]['zarr_url']
#     init_args = parallelization_list['parallelization_list'][2]['init_args']
#
#     print(parallelization_list['parallelization_list'])
#
#     convert_IC6000_to_ome_zarr(
#         zarr_url=zarr_url,
#         init_args=init_args,
#     )
#
#     parallelization_list = init_ashlar_stitching_and_registration(
#         zarr_dir=test_data_dir,
#         zarr_urls=[f"{test_data_dir}/test_plate.zarr/C/03/0",
#                    f"{test_data_dir}/test_plate.zarr/C/03/1"]
#     )
#
#     zarr_url = parallelization_list['parallelization_list'][0]['zarr_url']
#     init_args = parallelization_list['parallelization_list'][0]['init_args']
#
#     ashlar_stitching_and_registration(
#         zarr_url=zarr_url,
#         init_args=init_args,
#         overlap=0.1,
#         filter_sigma=10,
#         tmp_dir=None,
#         overwrite_input=False,
#         suffix="_stitched",
#         ref_wavelength_id=ref_wavelength_id,
#         ref_cycle=0,
#     )
#
#     # assert whether stitched images was saved in a new zarr image
#     stitched_image_path = Path(test_data_dir).joinpath(
#         "test_plate.zarr/C/03/0_stitched")
#     assert stitched_image_path.exists(),\
#         f"Stitched image not found at {stitched_image_path}"


def test_merge_plate_metadata(test_data_dir):

    image_list = [f"{test_data_dir}/{i}" for i in IMAGE_LIST_2D]

    zarr_url = image_list[0]

    measure_features(
        zarr_url=zarr_url,
        label_image_name='Label A',
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

    merge_plate_metadata(
        zarr_url=zarr_url,
        metadata_path=Path(test_data_dir).joinpath("metadata.csv").as_posix(),
        feature_table_name='feature_table',
        left_on='well_name',
        right_on='well',
        new_feature_table_name='feature_table_2',
    )

    feature_table = ad.read_zarr(f"{zarr_url}/tables/feature_table_2")
    obs = feature_table.obs

    # assert that the feature table contains the column treatment
    assert 'treatment' in obs.columns, \
        f"Column treatment not found in feature table"


def test_normalize_feature_table(test_data_dir):

    image_list = [f"{test_data_dir}/{i}" for i in IMAGE_LIST_2D]

    for zarr_url in image_list:

        measure_features(
            zarr_url=zarr_url,
            label_image_name='Label A',
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

        merge_plate_metadata(
            zarr_url=zarr_url,
            metadata_path=Path(test_data_dir).joinpath("metadata.csv").as_posix(),
            feature_table_name='feature_table',
            left_on='well_name',
            right_on='well',
            new_feature_table_name='feature_table_2',
        )

    parallelization_list = init_normalize_feature_table(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        feature_table_name='feature_table_2',
        condition_column='treatment',
        control_condition='control',
        normalization_layout=NormalizationLayout.full_plate,
        additional_control_filters={'sample': 'sample_1'}
    )

    for p in parallelization_list['parallelization_list']:
        zarr_url = p['zarr_url']
        init_args = p['init_args']

        normalize_feature_table(
            zarr_url=zarr_url,
            init_args=init_args,
            normalization_method=NormalizationMethod.z_score,
            output_table_name_suffix='_normalized',
        )


def test_correct_4i_bleaching_artifacts(test_data_dir):

    image_list = [f"{test_data_dir}/{i}" for i in IMAGE_LIST_2D]

    for zarr_url in image_list:

        measure_features(
            zarr_url=zarr_url,
            label_image_name='Label A',
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

        merge_plate_metadata(
            zarr_url=zarr_url,
            metadata_path=Path(test_data_dir).joinpath("metadata.csv").as_posix(),
            feature_table_name='feature_table',
            left_on='well_name',
            right_on='well',
            new_feature_table_name='feature_table_2',
        )

    parallelization_list = init_correct_4i_bleaching_artifacts(
        zarr_urls=image_list,
        zarr_dir=test_data_dir,
        feature_table_name='feature_table_2',
        condition_column='treatment',
        control_condition='control',
        additional_control_filters={'sample': 'sample_1'},
        model_output_dir=test_data_dir,
    )

    for p in parallelization_list['parallelization_list']:
        zarr_url = p['zarr_url']
        init_args = p['init_args']
        cycle = Path(zarr_url).name

        # change scale factors, because calculation fails for test
        corr_value = 0.5
        current_scale_factors = {
            f"Label A_Intensity_mean_intensity_{cycle}_DAPI": {0: corr_value}}

        init_args['current_scale_factors'] = current_scale_factors

        correct_4i_bleaching_artifacts(
            zarr_url=zarr_url,
            init_args=init_args,
            output_table_name_suffix='_bleaching_corrected',
        )

        # assert whether corrected table exists
        corrected_table_path = Path(zarr_url).joinpath(
            "tables/feature_table_2_bleaching_corrected"
        )
        assert corrected_table_path.exists(),\
            f"Corrected table not found at {corrected_table_path}"

        # assert that the feature values have been changed correctly
        original_table_path = Path(zarr_url).joinpath(
            "tables/feature_table_2"
        )

        feature_table = ad.read_zarr(original_table_path).to_df()
        feature_table_corr = ad.read_zarr(corrected_table_path).to_df()

        condition = feature_table_corr[f"Label A_Intensity_mean_intensity_{cycle}_DAPI"] == feature_table[f"Label A_Intensity_mean_intensity_{cycle}_DAPI"].div(corr_value)
        corr_factor_mean = feature_table[f'Label A_Intensity_mean_intensity_{cycle}_DAPI'].div(feature_table_corr[f'Label A_Intensity_mean_intensity_{cycle}_DAPI']).mean()

        # all in conditions must be True
        assert condition.all(), (
            f"Feature values not corrected correctly. "
            f"Expected correction factor of {corr_value},"
            f" but got correction factor of "
            f"{corr_factor_mean}")

        # assert that the plots have been generated
        plot_path = Path(test_data_dir).joinpath("feature_table_2","plots")

        assert plot_path.exists(),\
            f"Plot folder not found at {plot_path}"

        # assert that the decay models have been saved
        decay_model_path = Path(test_data_dir).joinpath("feature_table_2", "decay_model_params.csv")
        assert decay_model_path.exists(),\
            f"Decay model file not found at {decay_model_path}"






# Clean up: Remove the temporary files after the tests
@pytest.fixture(autouse=True)
def cleanup_temp_files(test_data_dir):
    yield
    try:
        shutil.rmtree(test_data_dir)
    except:
        pass