from typing import Optional
from pydantic import BaseModel
from enum import Enum


class InitArgsSegmentSecondary(BaseModel):
    """
    Segment Secondary HCS Plate init args.

    Passed from `group_by_well_segment_secondary` to
    `segment_secondary_compute`.

    Attributes:
        label_name: Name of the label image that contains the seeds.
            Needs to exist in OME-Zarr file.
        label_zarr_url: Zarr url indicating the zarr image containing the
            label image.
        channel_label: Label of the intensity image used for watershed.
            Needs to exist in OME-Zarr file.
        channel_zarr_url: Zarr url indicating the zarr image containing the
            channel image.
        mask: label image to use as mask. Only areas where the mask is
            non-zero will be considered for the watershed.
        mask_zarr_url: Zarr url indicating the zarr image containing the mask.
    """

    label_name: str
    label_zarr_url: str
    channel_label: str
    channel_zarr_url: str
    mask: Optional[str] = None
    mask_zarr_url: Optional[str] = None


class InitArgsExpandLabels(BaseModel):
    """
    Expand Labels init args.

    Attributes:
        label_name: Name of the label image to expand.
            Needs to exist in OME-Zarr file.
        label_zarr_url: Zarr url indicating the zarr image containing the
            label image.
    """

    label_name: str
    label_zarr_url: str


class InitArgsIC6000(BaseModel):
    """
    Arguments to be passed from IC6000 converter init to compute

    Attributes:
        image_dir: Directory where the raw images are found
        plate_prefix: part of the image filename needed for finding the
            right subset of image files
        well_ID: part of the image filename needed for finding the
            right subset of image files
        image_extension: part of the image filename needed for finding the
            right subset of image files
        image_glob_patterns: Additional glob patterns to filter the available
            images with
        acquisition: Acquisition metadata needed for multiplexing
    """

    image_dir: str
    plate_prefix: str
    well_ID: str
    image_extension: str
    image_glob_patterns: Optional[list[str]]
    acquisition: Optional[int]


class InitArgsLabelAssignment(BaseModel):
    """

    Arguments to be passed from Label Assignment init to compute

    Attributes:
        parent_label_name: Name of the parent label.
        parent_label_zarr_url: Zarr url indicating the zarr image containing the
            parent label image.
        child_label_name: Name of the child label. This label will be assigned
            to the parent label based on overlap. The parent label will appear
            in the child feature table as the "(parent_label_name)_label"
            column in the obs table of the anndata table.
        child_label_zarr_url: Zarr url indicating the zarr image containing the
            child label image.
    """

    parent_label_name: str
    parent_label_zarr_url: str
    child_label_name: str
    child_label_zarr_url: str


class InitArgsClipLabelImage(BaseModel):
    """

    Arguments to be passed from Clip Label Image init to compute

    Attributes:
        label_name: Name of the label image to be clipped.
            Needs to exist in OME-Zarr file.
        label_zarr_url: Zarr url indicating the zarr image containing the
            label image.
        clipping_mask_name: Name of the label image used as mask for clipping.
            This image will be binarized. Needs to exist in OME-Zarr file.
        clipping_mask_zarr_url: Zarr url indicating the zarr image containing
            the clipping mask image.
    """

    label_name: str
    label_zarr_url: str
    clipping_mask_name: str
    clipping_mask_zarr_url: str


class InitArgsMaskLabelImage(BaseModel):
    """

    Arguments to be passed from Mask Label Image init to compute

    Attributes:
        label_name: Name of the label image to be masked.
            Needs to exist in OME-Zarr file.
        label_zarr_url: Zarr url indicating the zarr image containing the
            label image.
        mask_name: Name of the label image used as mask.
            This image will be binarized. Needs to exist in OME-Zarr file.
        mask_zarr_url: Zarr url indicating the zarr image containing
            the mask image.
    """

    label_name: str
    label_zarr_url: str
    mask_name: str
    mask_zarr_url: str


class InitArgsFilterLabelBySize(BaseModel):
    """

    Arguments to be passed from Filter Label by Size init to compute

    Attributes:
        label_name: Name of the label image to be filtered by size.
            Needs to exist in OME-Zarr file.
        label_zarr_url: Zarr url indicating the zarr image containing the
            label image.
    """

    label_name: str
    label_zarr_url: str

class CorrectBy(Enum):
    """
    Enum for BaSiCPy correction options.
    """

    wavelength_id = "wavelength id"
    channel_label = "channel label"


class InitArgsBaSiCPyCalculate(BaseModel):
    """

    Arguments to be passed from BaSiCPy Calculate init to compute

    Attributes:
        channel_name: name of the channel for which the illumination model
            will be calculated. can be channel label or wavelength id.
        correct_by: if illumination profile will be calculated per channel_label
            or wavelength id.
        channel_zarr_urls: list of zarr urls specifying the images that
            contain the channel and will be used to calculate the illumination
            model.
        channel_zarr_dict: dictionary specifying how often each zarr url
            should be sampled.
        compute_per_well: If True, calculate illumination profiles per well.
            This can be useful if your experiment contains different stainings
            in each well (e.g., different antibodies with varying intensity
            ranges). Defaults to False.
        exclude_border_FOVs: If True, exclude border FOVs from the calculation.

    """

    channel_name: str
    correct_by: str
    channel_zarr_urls: list[str]
    channel_zarr_dict: dict[str, int]
    compute_per_well: bool
    exclude_border_FOVs: bool


class InitArgsAggregateFeatureTables(BaseModel):
    """

    Arguments to be passed from BaSiCPy Calculate init to compute

    Attributes:
        zarr_urls: list of zarr urls specifying the images that
            are present in the well.

    """
    zarr_urls: list[str]


class InitArgsCorrectChromaticShift(BaseModel):
    """

    Arguments to be passed from Correct Chromatic Shift init to compute

    Attributes:
        zarr_urls: list of zarr urls to be processed.
        transformation_maps: transformation maps dictionary to be applied to
            the images. Key is the wavelength id of the channel to be
            corrected. Value is  another dictionary containing the
            transformation map (value) for each ROI (key).

    """
    zarr_urls: list[str]
    transformation_maps: dict[str, dict]


class InitArgsConvertChannelToLabel(BaseModel):
    """

    Arguments to be passed from Convert Channel to Label init to compute

    Attributes:
        channel_label: label of the channel to convert into a label image.
        channel_zarr_url: list of zarr urls specifying the images that
            contain the channel and will be converted to a label image.

    """

    channel_label: str
    channel_zarr_url: str


class InitArgsDetectBlobCentroids(BaseModel):
    """

    Arguments to be passed from Convert Channel to Label init to compute

    Attributes:
        channel_label: label of the channel to convert into a label image.
        channel_zarr_url: list of zarr urls specifying the images that
            contain the channel and will be converted to a label image.

    """

    channel_label: str
    channel_zarr_url: str


class InitArgsAshlarStitchingAndRegistration(BaseModel):
    """

    Arguments to be passed from Ashlar Stitching and Regisrtation init to compute

    Attributes:
        zarr_urls: list of zarr urls specifying the images that
            are present in the well.

    """
    zarr_urls: list[str]


class InitArgsCalculatePixelIntensityCorrelation(BaseModel):
    """
    Calculate Pixel Intensity Correlation init args.

    Attributes:
        corr_channel_urls: List of dictionaries. Key and value represent
        the url of the intensity images that should be correlated.
        corr_channel_labels: List of dictionaries. Key and value represent
        the label names of two channels that should be correlated.
        label_name: Name of the label image that contains the seeds.
            Needs to exist in OME-Zarr file.
        label_zarr_url: Zarr url indicating the zarr image containing the
            label image.
    """
    corr_channel_urls: list[dict[str, str]]
    corr_channel_labels: list[dict[str, str]]
    label_name: str
    label_zarr_url: str


class InitArgsNormalizeFeatureTable(BaseModel):
    """
    Normalize Measurements init args.

    Attributes:
        ctrl_zarr_urls: List of paths or urls to the individual OME-Zarr image
            to be used as control conditions.
        feature_table_name: Name of the feature table that contains the
            measurements to be normalized.
    """
    ctrl_zarr_urls: list[str]
    feature_table_name: str


class InitArgsCorrect4iBleachingArtifacts(BaseModel):
    """
    Correct 4i Bleaching Artifacts init args.

    Attributes:
        current_scale_factors: scale factors for intensity features of the
            zarr-url to be corrected.
        feature_table_name: Name of the feature table that contains the
            measurements to be corrected.
    """
    current_scale_factors: dict[str, dict[int, float]]
    feature_table_name: str
