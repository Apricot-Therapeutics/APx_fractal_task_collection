import zarr
import logging
from natsort import natsorted
from pathlib import Path
from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.channels import get_channel_from_image_zarr
import dask.array as da

logger = logging.getLogger(__name__)

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

    zarr_group = zarr.open(zarrurl, mode="r")
    first_well_url = zarr_group.attrs['plate']['wells'][0]['path']
    well_group = zarr.open(zarrurl.joinpath(first_well_url), mode="r")

    img_paths = natsorted(
        [image['path'] for image in well_group.attrs['well']["images"]])

    actual_img_path = None
    for img_path in img_paths:
        label_path = zarrurl.joinpath(
            first_well_url,
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




