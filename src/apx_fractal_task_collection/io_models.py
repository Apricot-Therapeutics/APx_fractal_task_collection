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