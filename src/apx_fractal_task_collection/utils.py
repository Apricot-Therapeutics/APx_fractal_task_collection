import zarr
import logging
from natsort import natsorted
from pathlib import Path
from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.channels import get_channel_from_image_zarr
import dask.array as da
from pydantic import BaseModel, model_validator
from typing_extensions import Self, Literal
from basicpy.basicpy import FittingMode, ResizeMode

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    'morphology': [
        'Morphology_area',
        'Morphology_centroid-0',
        'Morphology_centroid-1',
        'Morphology_centroid-2',
        'Morphology_well_centroid-0',
        'Morphology_well_centroid-1',
        'Morphology_well_centroid-2',
        'Morphology_bbox_area',
        'Morphology_bbox-0',
        'Morphology_bbox-1',
        'Morphology_bbox-2',
        'Morphology_bbox-3',
        'Morphology_bbox-4',
        'Morphology_bbox-5',
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
        'Morphology_circularity'],
    'intensity': [
        'Intensity_max_intensity',
        'Intensity_mean_intensity',
        'Intensity_min_intensity',
        'Intensity_sum_intensity',
        'Intensity_std_intensity'],
    'texture': [
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
        'Texture_LTE_LS'],
    'population': [
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
        'Population_mean_distance_nn_50',
        'Population_mean_distance_nn_100',
        'Population_n_neighbours_radius_100',
        'Population_mean_distance_neighbours_radius_100',
        'Population_n_neighbours_radius_200',
        'Population_mean_distance_neighbours_radius_200',
        'Population_n_neighbours_radius_300',
        'Population_mean_distance_neighbours_radius_300',
        'Population_n_neighbours_radius_400',
        'Population_mean_distance_neighbours_radius_400',
        'Population_n_neighbours_radius_500',
        'Population_mean_distance_neighbours_radius_500']
}


def get_acquisition_from_channel_label(zarrurl: Path,
                                       channel_label: str) -> str:
    """
    Get the acquisition name from the channel label.

    Args:
        channel_label: Label of the channel.

    Returns:
        The acquisition name.
    """

    zarr_group = zarr.open(zarrurl, mode="r")
    first_well_url = zarr_group.attrs['plate']['wells'][0]['path']
    well_group = zarr.open(zarrurl.joinpath(first_well_url), mode="r")

    img_paths = natsorted(
        [image['path'] for image in well_group.attrs['well']["images"]])

    actual_img_path = None
    for img_path in img_paths:
        image_path = zarrurl.joinpath(
            first_well_url,
            img_path,
        )
        channel_list = get_omero_channel_list(image_zarr_path=image_path)
        if channel_label in [c.label for c in channel_list]:
            actual_img_path = img_path
            break

    return actual_img_path


def get_acquisition_from_label_name(zarrurl: Path,
                                    label_name: str) -> str:
    """
    Get the acquisition name from the label name.

    Args:
        label_name: Name of the label image.

    Returns:
        The acquisition name.
    """
    well_group = zarr.open(zarrurl, mode="r")
    img_paths = natsorted(
        [image['path'] for image in well_group.attrs['well']["images"]])

    actual_img_path = None
    for img_path in img_paths:
        label_path = zarrurl.joinpath(
            img_path,
            "labels",
            label_name,
        )
        if label_path.exists():
            actual_img_path = img_path
            break

    return actual_img_path


def get_label_image_from_well(wellurl: Path, label_name: str, level: int = 0):
    '''
    Get the image data for a specific channel from an OME-Zarr file.

    Args:
        zarrurl: Path to the OME-Zarr file.
        channel_label: Label of the channel to extract.

    Returns:
        The image data for the specified channel as dask array
    '''

    well_group = zarr.open_group(wellurl, mode="r+")
    for image in well_group.attrs['well']['images']:
        try:
            img_zarr_path = wellurl.joinpath(wellurl, image['path'])
            data_zyx = da.from_zarr(
                img_zarr_path.joinpath("labels", label_name, str(level)))
            break
        except:
            continue

    return data_zyx, img_zarr_path


def get_channel_image_from_well(wellurl: Path,
                                channel_label: str,
                                level: int = 0):
    '''
    Get the image data for a specific channel from an OME-Zarr file.

    Args:
        wellurl: Path to the OME-Zarr file.
        channel_label: Label of the channel to extract.

    Returns:
        The image data for the specified channel as dask array
    '''


    well_group = zarr.open(wellurl, mode='r')
    for image in well_group.attrs['well']['images']:
        img_zarr_path = wellurl.joinpath(wellurl, image['path'])
        channel_list = get_omero_channel_list(
            image_zarr_path=img_zarr_path)

        if channel_label in [c.label for c in channel_list]:
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=img_zarr_path,
                wavelength_id=None,
                label=channel_label
            )

            ind_channel = tmp_channel.index
            data_zyx = \
                da.from_zarr(img_zarr_path.joinpath(str(level)))[ind_channel]

            return data_zyx, img_zarr_path
        

def get_channel_image_from_image(img_url: Path,
                                 channel_label: str,
                                 level: int = 0):
    '''
    Get the image data for a specific channel from an OME-Zarr file.

    Args:
        img_url: Path to the OME-Zarr file.
        channel_label: Label of the channel to extract.

    Returns:
        The image data for the specified channel as dask array
    '''

    channel_list = get_omero_channel_list(
        image_zarr_path=img_url)

    if channel_label in [c.label for c in channel_list]:
        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=img_url,
            wavelength_id=None,
            label=channel_label
        )

        ind_channel = tmp_channel.index
        data_zyx = \
            da.from_zarr(img_url.joinpath(str(level)))[ind_channel]

        return data_zyx


def get_zarr_urls(zarr_dir: str) -> list[str]:
    """
    Get all zarr urls in a directory.

    Args:
        zarr_dir: Path to the directory containing the zarr files.

    Returns:
        List of zarr urls.
    """
    zarr_urls = list(Path(zarr_dir).glob(
        pattern=f"[A-Z]/[0-9][0-9]/[0-9]"))
    zarr_urls = [str(zarr_url) for zarr_url in zarr_urls]

    return zarr_urls


class TextureFeatures(BaseModel):
    """
    Validator to handle texture feature selection.

    Attributes:
        haralick: Flag to calculate Haralick texture features.
        laws_texture_energy: Flag to calculate Law's Texture Energy features.
        clip_value: Value to which to clip the intensity image for haralick
            texture feature calculation. Will be applied to all channels
            except the ones specified in clip_value_exceptions.
        clip_value_exceptions: Dictionary of exceptions for the clip value.
            The dictionary should have the channel name as key and the
            clip value as value.
    """

    haralick: bool = False
    laws_texture_energy: bool = False
    clip_value: int = 5000
    clip_value_exceptions: dict[str, int] = {}

    @model_validator(mode="after")
    def validate_conditions(self: Self) -> Self:
        # Extract values
        haralick = self.haralick
        laws_texture_energy = self.laws_texture_energy
        lower_percentile = self.clip_value
        upper_percentile = self.clip_value_exceptions

        return self


class BaSiCPyModelParams(BaseModel):
    """
    Advanced parameters for BaSiCPy illumination correction.

    Attributes:
        autosegment: When not False, automatically segment the image before fitting.When True, threshold_otsu from scikit-image is used and the brighter pixels are taken.When a callable is given, it is used as the segmentation function.
        autosegment_margin: Margin of the segmentation mask to the thresholded region.
        epsilon: Weight regularization term.
        fitting_mode: Must be one of [‘ladmap’, ‘approximate’]
        get_darkfield: When True, will estimate the darkfield shading component.
        max_iterations: Maximum number of iterations for single optimization.
        max_mu_coef: Maximum allowed value of mu, divided by the initial value.
        max_reweight_iterations: Maximum number of reweighting iterations.
        max_reweight_iterations_baseline: Maximum number of reweighting iterations for baseline.
        max_workers: Maximum number of threads used for processing.
        mu_coef: Coefficient for initial mu value.
        optimization_tol: Optimization tolerance.
        optimization_tol_diff: Optimization tolerance for update diff.
        resize_mode: Resize mode to use when downsampling images. Must be one of ‘jax’, ‘skimage’, and ‘skimage_dask’
        #resize_params: Parameters for the resize function when downsampling images.
        reweighting_tol: Reweighting tolerance in mean absolute difference of images.
        rho: Parameter rho for mu update.
        smoothness_darkfield: Weight of the darkfield term in the Lagrangian.
        smoothness_flatfield: Weight of the flatfield term in the Lagrangian.
        sort_intensity: Whether or not to sort the intensities of the image.
        sparse_cost_darkfield: Size for running computations. None means no rescaling.
        working_size: Working
    """

    autosegment: bool = False
    autosegment_margin: int = 10
    epsilon: float = 0.1
    fitting_mode: Literal[FittingMode.ladmap, FittingMode.approximate] = FittingMode.ladmap
    get_darkfield: bool = True
    max_iterations: int = 500
    max_mu_coef: float = 10000000.0
    max_reweight_iterations: int = 10
    max_reweight_iterations_baseline: int = 5
    max_workers: int = 2
    mu_coef: float = 12.5
    optimization_tol: float = 0.001
    optimization_tol_diff: float = 0.01
    resize_mode: Literal[ResizeMode.jax, ResizeMode.skimage, ResizeMode.skimage_dask] = ResizeMode.jax
    #resize_params: dict[str, Any] = {}
    reweighting_tol: float = 0.01
    rho: float = 1.5
    smoothness_darkfield: float = 1.0
    smoothness_flatfield: float = 1.0
    sort_intensity: bool = False
    sparse_cost_darkfield: float = 0.01
    working_size: int = 128


