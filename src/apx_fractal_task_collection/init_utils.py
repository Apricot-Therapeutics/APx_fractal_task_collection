from fractal_tasks_core.channels import get_omero_channel_list
from pathlib import Path

import pandas as pd
import numpy as np

from defusedxml import ElementTree

import fractal_tasks_core


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__



def group_by_well(zarr_urls: list[str]):
    """
    Create well dictionaries for the zarr_urls

    Keys are well ids, values are a list of zarr_urls that belong to that well.

    zarr_urls: List of zarr_urls. Each zarr_url is a string that defines the
        path to an individual zarr_url in an HCS plate
    """
    well_dict = {}
    for zarr_url in zarr_urls:
        well_id = zarr_url.rsplit("/", 1)[0]

        if well_id not in well_dict:
            well_dict[well_id] = [zarr_url]
        else:
            well_dict[well_id].append(zarr_url)
    return well_dict


def group_by_channel(zarr_urls: list[str]):
    """
    Create channel dictionaries for the zarr_urls

    Keys are channel ids, values are a list of zarr_urls that belong to that channel.

    zarr_urls: List of zarr_urls. Each zarr_url is a string that defines the
        path to an individual zarr_url in an HCS plate
    """
    channel_dict = {}
    for zarr_url in zarr_urls:
        channels = get_omero_channel_list(image_zarr_path=zarr_url)
        for channel in channels:
            if channel.label not in channel_dict:
                channel_dict[channel.label] = [zarr_url]
            else:
                channel_dict[channel.label].append(zarr_url)
    return channel_dict


def group_by_well_and_channel(zarr_urls: list[str]):
    """
    Create channel dictionaries for the zarr_urls

    Keys are channel ids, values are a list of zarr_urls that belong to that channel.

    zarr_urls: List of zarr_urls. Each zarr_url is a string that defines the
        path to an individual zarr_url in an HCS plate
    """
    channel_dict = {}
    for zarr_url in zarr_urls:
        channels = get_omero_channel_list(image_zarr_path=zarr_url)
        well_id = zarr_url.rsplit("/", 3)[1] + zarr_url.rsplit("/", 3)[2]
        for channel in channels:
            if channel.label not in channel_dict:
                channel_dict[f"well_{well_id}_ch_{channel.label}"] = [zarr_url]
            else:
                channel_dict[f"well_{well_id}_ch_{channel.label}"].append(zarr_url)
    return channel_dict


def group_by_wavelength(zarr_urls: list[str]):
    """
    Create wavelength dictionaries for the zarr_urls

    Keys are wavelength ids, values are a list of zarr_urls that belong to that wavelength.

    zarr_urls: List of zarr_urls. Each zarr_url is a string that defines the
        path to an individual zarr_url in an HCS plate
    """
    channel_dict = {}
    for zarr_url in zarr_urls:
        channels = get_omero_channel_list(image_zarr_path=zarr_url)
        for channel in channels:
            if channel.wavelength_id not in channel_dict:
                channel_dict[channel.wavelength_id] = [zarr_url]
            else:
                channel_dict[channel.wavelength_id].append(zarr_url)
    return channel_dict


def group_by_well_and_wavelength(zarr_urls: list[str]):
    """
    Create wavelength dictionaries for the zarr_urls

    Keys are wavelength ids, values are a list of zarr_urls that belong to that wavelength.

    zarr_urls: List of zarr_urls. Each zarr_url is a string that defines the
        path to an individual zarr_url in an HCS plate
    """
    channel_dict = {}
    for zarr_url in zarr_urls:
        channels = get_omero_channel_list(image_zarr_path=zarr_url)
        well_id = zarr_url.rsplit("/", 3)[1] + zarr_url.rsplit("/", 3)[2]
        for channel in channels:
            if channel.wavelength_id not in channel_dict:
                channel_dict[f"well_{well_id}_ch_{channel.wavelength_id}"] = [zarr_url]
            else:
                channel_dict[f"well_{well_id}_ch_{channel.wavelength_id}"].append(zarr_url)
    return channel_dict


def get_label_zarr_url(well_list: list, label_name: str) -> str:
    """
    Get the label zarr url for a well

    Args:
        well_list: list of zarr urls for a well
        label_name: name of the label

    Returns:
        label_zarr_url: Label zarr url for the well
    """
    out_urls = []
    for zarr_url in well_list:
        label_path = Path(f"{zarr_url}/labels/{label_name}")
        if label_path.exists():
            out_urls.append(zarr_url)

    if len(out_urls) > 1:
        raise ValueError(f"Multiple label zarr urls found for {label_name}")
    else:
        return out_urls[0]


def get_channel_zarr_url(well_list: list, channel_label: str) -> str:
    """
    Get the channel zarr url for a well

    Args:
        well_list: list of zarr urls for a well
        channel_label: name of the channel

    Returns:
        channel_zarr_url: channel zarr url for the well
    """
    out_urls = []
    for zarr_url in well_list:
        channels = get_omero_channel_list(image_zarr_path=zarr_url)
        channel_labels = [channel.label for channel in channels]
        if channel_label in channel_labels:
            out_urls.append(zarr_url)

    if len(out_urls) > 1:
        raise ValueError(f"Multiple channel zarr urls "
                         f"found for {channel_label}")
    else:
        return out_urls[0]


def parse_platename(filename: str) -> dict[str, str]:
    metadata = ElementTree.parse(filename).getroot()
    return metadata.get("PlateID")


def parse_filename(filename: str) -> dict[str, str]:
    """
    Parse image metadata from filename.

    Args:
        filename: Name of the image.

    Returns:
        Metadata dictionary.
    """

    # Remove extension and folder from filename
    filename = Path(filename).with_suffix("").name
    # Remove plate prefix
    filename_split = filename.split("_")
    if len(filename_split) > 1:
        filename = filename_split[-1]
    else:
        filename = filename_split[0]

    output = {}
    output["well"] = filename.split("(")[0].split(" - ")[0] + \
                     filename.split("(")[0].split(" - ")[1]
    output["T"] = '0000'
    output["F"] = filename.split("(fld ")[1].split(" wv")[0]
    output["L"] = '01'
    output["A"] = '01'
    output["Z"] = '01'
    output["C"] = filename.split(" wv ")[1].split(")")[0]

    return output

def parse_IC6000_metadata(metadata_path, filename_patterns):
    metadata = ElementTree.parse(metadata_path)
    obj_calibration = \
        metadata.findall("AutoLeadAcquisitionProtocol")[0].findall(
            "ObjectiveCalibration")[0]
    pixel_size_x = float(obj_calibration.get("pixel_width"))
    pixel_size_y = float(obj_calibration.get("pixel_height"))

    camera_size = \
        metadata.findall("AutoLeadAcquisitionProtocol")[0].findall("Camera")[
            0].findall("Size")[0]
    x_pixel = int(camera_size.get("width"))
    y_pixel = int(camera_size.get("height"))

    wavelengths = \
        metadata.findall("AutoLeadAcquisitionProtocol")[0].findall(
            "Wavelengths")[0].findall("Wavelength")
    pixel_size_z = [float(w.get("z_step")) for w in wavelengths]
    # warn if >1 unique values
    pixel_size_z = np.unique(pixel_size_z)[0]

    # hardcoded stuff
    bit_depth = 16
    z_micrometer = 0
    z_pixel = 1

    images = metadata.findall("Images")[0].findall("Image")
    well_id = [
        f"{image.findall('Well')[0].get('label').split(' -')[0]}{int(image.findall('Well')[0].get('label').split('- ')[1]):02d}"
        for image in images]
    field_index = [
        int(image.findall("Identifier")[0].get("field_index")) + 1 for
        image in images]
    x_micrometer = [float(image.findall("PlatePosition_um")[0].get('x'))
                    for image in images]
    y_micrometer = [float(image.findall("PlatePosition_um")[0].get('y'))
                    for image in images]
    timestamp = [
        pd.to_datetime(float(image.get("timestamp_sec")), unit="s") for
        image in images]
    filenames = [image.get("filename") for image in images]

    df = pd.DataFrame(data={
        'well_id': well_id,
        'FieldIndex': field_index,
        'x_micrometer': x_micrometer,
        'y_micrometer': y_micrometer,
        'z_micrometer': z_micrometer,
        'pixel_size_z': pixel_size_z,
        'z_pixel': z_pixel,
        'Time': timestamp,
        'pixel_size_x': pixel_size_x,
        'pixel_size_y': pixel_size_y,
        'x_pixel': x_pixel,
        'y_pixel': y_pixel,
        'bit_depth': bit_depth,
        'filename': filenames})

    if filename_patterns is not None:

        patterns = [pattern.replace('*', r'') for pattern in
                    filename_patterns]
        for pattern in patterns:
            df = df[df['filename'].str.contains(pattern)]

    site_metadata = df.groupby(['well_id', 'FieldIndex']).apply(
        lambda x: x.iloc[0])
    site_metadata = site_metadata.set_index(['well_id', 'FieldIndex'])

    total_files = df.groupby('well_id')['FieldIndex'].count().to_dict()

    return site_metadata, total_files

