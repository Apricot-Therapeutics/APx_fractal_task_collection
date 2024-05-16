from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.channels import OmeroChannel


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


class InitArgsBaSiCPyCalculate(BaseModel):
    """

    Arguments to be passed from BaSiCPy Calculate init to compute

    Attributes:
        channel_label: label of the channel for which the illumination model
            will be calculated.
        channel_zarr_urls: list of zarr urls specifying the images that
            contain the channel and will be used to calculate the illumination
            model.
        channel_zarr_dict: dictionary specifying how often each zarr url
            should be sampled.

    """

    channel_label: str
    channel_zarr_urls: list[str]
    channel_zarr_dict: dict[str, int]