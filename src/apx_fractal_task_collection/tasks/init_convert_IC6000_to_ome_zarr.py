# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# Adapted by:
# Adrian Tschan <adrian.tschan@uzh.ch>
#
# This file is based on Fractal code originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Create OME-NGFF zarr group, for multiplexing dataset.
"""
import os
from pathlib import Path
from typing import Any
from typing import Optional

import zarr
from pydantic import validate_call
from zarr.errors import ContainsGroupError

from apx_fractal_task_collection.init_utils import (parse_platename,
                                                    parse_filename,
                                                    parse_IC6000_metadata)
from apx_fractal_task_collection.io_models import InitArgsIC6000

import fractal_tasks_core
from fractal_tasks_core.channels import check_unique_wavelength_ids
from fractal_tasks_core.channels import check_well_channel_labels
from fractal_tasks_core.channels import define_omero_channels
from fractal_tasks_core.cellvoyager.filenames import glob_with_multiple_patterns
from fractal_tasks_core.roi import prepare_FOV_ROI_table
from fractal_tasks_core.roi import prepare_well_ROI_table
from fractal_tasks_core.roi import remove_FOV_overlaps
from fractal_tasks_core.tasks.io_models import MultiplexingAcquisition
from fractal_tasks_core.zarr_utils import open_zarr_group_with_overwrite
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.ngff.specs import Plate
from fractal_tasks_core.ngff.specs import Well

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

import logging


logger = logging.getLogger(__name__)


@validate_call
def init_convert_IC6000_to_ome_zarr(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    acquisitions: dict[str, MultiplexingAcquisition],
    # Advanced parameters
    image_glob_patterns: Optional[list[str]] = None,
    num_levels: int = 5,
    coarsening_xy: int = 2,
    image_extension: str = "tif",
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create OME-NGFF structure and metadata to host a multiplexing dataset.

    This task takes a set of image folders (i.e. different acquisition cycles)
    and build the internal structure and metadata of a OME-NGFF zarr group,
    without actually loading/writing the image data.

    Each element in input_paths should be treated as a different acquisition.

    Args:
        input_paths: List of input paths where the image data from the
            microscope is stored (as TIF or PNG).  Each element of the list is
            treated as another cycle of the multiplexing data, the cycles are
            ordered by their order in this list.  Should point to the parent
            folder containing the images and the metadata files
            `MeasurementData.mlf` and `MeasurementDetail.mrf` (if present).
            Example: `["/path/cycle1/", "/path/cycle2/"]`. (standard argument
            for Fractal tasks, managed by Fractal server).
        output_path: Path were the output of this task is stored.
            Example: `"/some/path/"` => puts the new OME-Zarr file in the
            `/some/path/`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        acquisitions: dictionary of acquisitions. Each key is the acquisition
            identifier (normally 0, 1, 2, 3 etc.). Each item defines the
            acquisition by providing the image_dir and the allowed_channels.
        image_glob_patterns: If specified, only parse images with filenames
            that match with all these patterns. Patterns must be defined as in
            https://docs.python.org/3/library/fnmatch.html, Example:
            `image_glob_pattern=["*B - 03*"]` => only process well B03
            `image_glob_pattern=["*C - 09*", "*fld 16*"]` =>
            only process well C09, field of view 16.
        num_levels: Number of resolution-pyramid levels. If set to `5`, there
            will be the full-resolution level and 4 levels of downsampled
            images.
        coarsening_xy: Linear coarsening factor between subsequent levels.
            If set to `2`, level 1 is 2x downsampled, level 2 is 4x downsampled
            etc.
        image_extension: Filename extension of images
            (e.g. `"tif"` or `"png"`).
        overwrite: If `True`, overwrite the task output.

    Returns:
        A metadata dictionary containing important metadata about the OME-Zarr
            plate, the images and some parameters required by downstream tasks
            (like `num_levels`).
    """

    # Preliminary checks on allowed_channels
    # Note that in metadata the keys of dictionary arguments should be
    # strings (and not integers), so that they can be read from a JSON file
    for key, values in acquisitions.items():
        if not isinstance(key, str):
            raise ValueError(f"{acquisitions=} has non-string keys")
        check_unique_wavelength_ids(values.allowed_channels)

    # Identify all plates and all channels, per input folders
    dict_acquisitions: dict = {}

    for acquisition, acq_input in acquisitions.items():
        dict_acquisitions[acquisition] = {}

        # IC6000 may not contain plate name in filename, getting it from
        # metadata file instead
        xml_path = list(Path(acq_input.image_dir).glob("*.xdce"))[0]

        plate = parse_platename(xml_path)
        plate_prefix = ""

        actual_wavelength_ids = []
        plates = []
        plate_prefixes = []

        # Loop over all images
        patterns = [f"*.{image_extension}"]
        if image_glob_patterns:
            patterns.extend(image_glob_patterns)
        input_filenames = glob_with_multiple_patterns(
            folder=acq_input.image_dir,
            include_patterns=patterns,
        )

        for fn in input_filenames:
            try:
                filename_metadata = parse_filename(Path(fn).name)
                plates.append(plate)
                plate_prefixes.append(plate_prefix)
                actual_wavelength_ids.append(filename_metadata['C'])
            except ValueError as e:
                logger.warning(
                    f'Skipping "{Path(fn).name}". Original error: ' + str(e)
                )

        plates = sorted(list(set(plates)))
        actual_wavelength_ids = sorted(list(set(actual_wavelength_ids)))

        info = (
            "Listing all plates/channels:\n"
            f"Patterns: {patterns}\n"
            f"Plates:   {plates}\n"
            f"Actual wavelength IDs: {actual_wavelength_ids}\n"
        )

        # Check that a folder includes a single plate
        if len(plates) > 1:
            raise ValueError(f"{info}ERROR: {len(plates)} plates detected")
        elif len(plates) == 0:
            raise ValueError(f"{info}ERROR: No plates detected")
        original_plate = plates[0]
        plate_prefix = plate_prefixes[0]

        # Replace plate with the one of acquisition 0, if needed
        if int(acquisition) > 0:
            plate = dict_acquisitions["0"]["plate"]
            logger.warning(
                f"For {acquisition=}, we replace {original_plate=} with "
                f"{plate=} (the one for acquisition 0)"
            )

        # Check that all channels are in the allowed_channels
        allowed_wavelength_ids = [
            c.wavelength_id for c in acq_input.allowed_channels
        ]
        if not set(actual_wavelength_ids).issubset(
                set(allowed_wavelength_ids)
        ):
            msg = "ERROR in create_ome_zarr\n"
            msg += f"actual_wavelength_ids: {actual_wavelength_ids}\n"
            msg += f"allowed_wavelength_ids: {allowed_wavelength_ids}\n"
            raise ValueError(msg)

        # Create actual_channels, i.e. a list of the channel dictionaries which
        # are present
        actual_channels = [
            channel
            for channel in acq_input.allowed_channels
            if channel.wavelength_id in actual_wavelength_ids
        ]

        logger.info(f"plate: {plate}")
        logger.info(f"actual_channels: {actual_channels}")

        dict_acquisitions[acquisition] = {}
        dict_acquisitions[acquisition]["plate"] = plate
        dict_acquisitions[acquisition]["original_plate"] = original_plate
        dict_acquisitions[acquisition]["plate_prefix"] = plate_prefix
        dict_acquisitions[acquisition][
            "image_folder"] = acq_input.image_dir
        dict_acquisitions[acquisition]["original_paths"] = [
            acq_input.image_dir
        ]
        dict_acquisitions[acquisition]["actual_channels"] = actual_channels
        dict_acquisitions[acquisition][
            "actual_wavelength_ids"
        ] = actual_wavelength_ids

        dict_acquisitions[acquisition]["input_filenames"] = input_filenames

    # create parallelization list
    parallelization_list = []
    acquisitions_sorted = sorted(list(acquisitions.keys()))
    current_plates = [item["plate"] for item in dict_acquisitions.values()]
    if len(set(current_plates)) > 1:
        raise ValueError(f"{current_plates=}")
    plate = current_plates[0]

    zarrurl = dict_acquisitions[acquisitions_sorted[0]]["plate"] + ".zarr"
    full_zarrurl = str(Path(zarr_dir) / zarrurl)
    logger.info(f"Creating {full_zarrurl=}")
    # Call zarr.open_group wrapper, which handles overwrite=True/False
    group_plate = open_zarr_group_with_overwrite(
        full_zarrurl, overwrite=overwrite
    )

    group_plate.attrs["plate"] = {
        "acquisitions": [
            {
                "id": int(acquisition),
                "name": dict_acquisitions[acquisition]["original_plate"],
            }
            for acquisition in acquisitions_sorted
        ]
    }

    zarrurls: dict[str, list[str]] = {"well": [], "image": []}
    zarrurls["plate"] = [f"{plate}.zarr"]

    ################################################################
    logging.info(f"{acquisitions_sorted=}")

    for acquisition in acquisitions_sorted:

        # Define plate zarr
        image_folder = dict_acquisitions[acquisition]["image_folder"]
        logger.info(f"Looking at {image_folder=}")

        # Obtain FOV-metadata dataframe
        xml_path = list(Path(acq_input.image_dir).glob("*.xdce"))[0]
        site_metadata, total_files = parse_IC6000_metadata(
            xml_path, filename_patterns=image_glob_patterns
        )
        site_metadata = remove_FOV_overlaps(site_metadata)

        # Extract pixel sizes and bit_depth
        pixel_size_z = site_metadata["pixel_size_z"][0]
        pixel_size_y = site_metadata["pixel_size_y"][0]
        pixel_size_x = site_metadata["pixel_size_x"][0]
        bit_depth = site_metadata["bit_depth"][0]

        if min(pixel_size_z, pixel_size_y, pixel_size_x) < 1e-9:
            raise ValueError(pixel_size_z, pixel_size_y, pixel_size_x)

        # Identify all wells
        # patterns = [f"*.{image_extension}"]
        # if image_glob_patterns:
        #     patterns.extend(image_glob_patterns)
        # plate_images = glob_with_multiple_patterns(
        #     folder=str(image_folder),
        #     patterns=patterns,
        # )

        plate_images = dict_acquisitions[acquisition]["input_filenames"]

        wells = [
            parse_filename(os.path.basename(fn))["well"] for fn in
            plate_images
        ]

        wells = sorted(list(set(wells)))
        logger.info(f"{wells=}")

        # Verify that all wells have all channels
        actual_channels = dict_acquisitions[acquisition]["actual_channels"]

        for well in wells:
            # patterns = [f"*{well[0]} - {well[1:]}(*.{image_extension}"]
            # if image_glob_patterns:
            #     patterns.extend(image_glob_patterns)
            # well_images = glob_with_multiple_patterns(
            #     folder=str(image_folder),
            #     patterns=patterns,
            # )

            well_images = [img for img in plate_images if
                           f"{well[0]} - {well[1:]}(" in img]

            well_wavelength_ids = []
            for fpath in well_images:
                try:
                    filename_metadata = parse_filename(
                        os.path.basename(fpath))
                    well_wavelength_ids.append(filename_metadata["C"])
                except IndexError:
                    logger.info(f"Skipping {fpath}")
            well_wavelength_ids = sorted(list(set(well_wavelength_ids)))
            actual_wavelength_ids = dict_acquisitions[acquisition][
                "actual_wavelength_ids"
            ]

            if well_wavelength_ids != actual_wavelength_ids:
                raise ValueError(
                    f"ERROR: well {well} in plate {plate} (prefix: "
                    f"{plate_prefix}) has missing channels.\n"
                    f"Expected: {actual_wavelength_ids}\n"
                    f"Found: {well_wavelength_ids}.\n"
                )

        well_rows_columns = [
            ind for ind in sorted([(n[0], n[1:]) for n in wells])
        ]
        row_list = [
            well_row_column[0] for well_row_column in well_rows_columns
        ]
        col_list = [
            well_row_column[1] for well_row_column in well_rows_columns
        ]
        row_list = sorted(list(set(row_list)))
        col_list = sorted(list(set(col_list)))

        plate_attrs = group_plate.attrs["plate"]
        plate_attrs["columns"] = [{"name": col} for col in col_list]
        plate_attrs["rows"] = [{"name": row} for row in row_list]
        plate_attrs["wells"] = [
            {
                "path": well_row_column[0] + "/" + well_row_column[1],
                "rowIndex": row_list.index(well_row_column[0]),
                "columnIndex": col_list.index(well_row_column[1]),
            }
            for well_row_column in well_rows_columns
        ]
        plate_attrs["version"] = __OME_NGFF_VERSION__
        # Validate plate attrs
        Plate(**plate_attrs)
        group_plate.attrs["plate"] = plate_attrs

        for row, column in well_rows_columns:
            parallelization_list.append(
                {
                    "zarr_url": (
                        f"{zarr_dir}/{plate}.zarr/{row}/{column}/"
                        f"{acquisition}/"
                    ),
                    "init_args": InitArgsIC6000(
                        image_dir=acquisitions[acquisition].image_dir,
                        plate_prefix=plate_prefix,
                        well_ID=row+column,
                        image_extension=image_extension,
                        image_glob_patterns=image_glob_patterns,
                        acquisition=acquisition,
                    ).dict(),
                }
            )
            try:
                group_well = group_plate.create_group(f"{row}/{column}/")
                logging.info(f"Created new group_well at {row}/{column}/")
                well_attrs = {
                    "images": [
                        {
                            "path": f"{acquisition}",
                            "acquisition": int(acquisition),
                        }
                    ],
                    "version": __OME_NGFF_VERSION__,
                }
                # Validate well attrs:
                Well(**well_attrs)
                group_well.attrs["well"] = well_attrs
                zarrurls["well"].append(f"{plate}.zarr/{row}/{column}")
            except ContainsGroupError:
                group_well = zarr.open_group(
                    f"{full_zarrurl}/{row}/{column}/", mode="r+"
                )
                logging.info(
                    f"Loaded group_well from {full_zarrurl}/{row}/{column}"
                )
                current_images = group_well.attrs["well"]["images"] + [
                    {"path": f"{acquisition}",
                     "acquisition": int(acquisition)}
                ]
                well_attrs = dict(
                    images=current_images,
                    version=group_well.attrs["well"]["version"],
                )
                # Validate well attrs:
                Well(**well_attrs)
                group_well.attrs["well"] = well_attrs

            group_image = group_well.create_group(
                f"{acquisition}/"
            )  # noqa: F841
            logging.info(
                f"Created image group {row}/{column}/{acquisition}")
            image = f"{plate}.zarr/{row}/{column}/{acquisition}"
            zarrurls["image"].append(image)

            group_image.attrs["multiscales"] = [
                {
                    "version": __OME_NGFF_VERSION__,
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {
                            "name": "z",
                            "type": "space",
                            "unit": "micrometer",
                        },
                        {
                            "name": "y",
                            "type": "space",
                            "unit": "micrometer",
                        },
                        {
                            "name": "x",
                            "type": "space",
                            "unit": "micrometer",
                        },
                    ],
                    "datasets": [
                        {
                            "path": f"{ind_level}",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [
                                        1,
                                        pixel_size_z,
                                        pixel_size_y
                                        * coarsening_xy ** ind_level,
                                        pixel_size_x
                                        * coarsening_xy ** ind_level,
                                    ],
                                }
                            ],
                        }
                        for ind_level in range(num_levels)
                    ],
                }
            ]

            group_image.attrs["omero"] = {
                "id": 1,  # FIXME does this depend on the plate number?
                "name": "TBD",
                "version": __OME_NGFF_VERSION__,
                "channels": define_omero_channels(
                    channels=actual_channels,
                    bit_depth=bit_depth,
                    label_prefix=acquisition,
                ),
            }

            # Validate Image attrs
            NgffImageMeta(**group_image.attrs)

            # Prepare AnnData tables for FOV/well ROIs
            well_id = row + column
            FOV_ROIs_table = prepare_FOV_ROI_table(
                site_metadata.loc[well_id])
            well_ROIs_table = prepare_well_ROI_table(
                site_metadata.loc[well_id]
            )

            # Write AnnData tables into the `tables` zarr group
            write_table(
                group_image,
                "FOV_ROI_table",
                FOV_ROIs_table,
                overwrite=overwrite,
                table_attrs={"type": "roi_table"},
            )
            write_table(
                group_image,
                "well_ROI_table",
                well_ROIs_table,
                overwrite=overwrite,
                table_attrs={"type": "roi_table"},
            )

    # Check that the different images (e.g. different acquisitions) in the each
    # well have unique labels
    for well_path in zarrurls["well"]:
        check_well_channel_labels(
            well_zarr_path=str(Path(zarr_dir) / well_path)
        )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_convert_IC6000_to_ome_zarr,
        logger_name=logger.name,
    )

