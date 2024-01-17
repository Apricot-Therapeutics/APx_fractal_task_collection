"""
# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
# Joel Lüthi  <joel.luethi@fmi.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""

import logging
from typing import Any, Dict, Sequence

import fractal_tasks_core
import zarr
import anndata as ad
from typing import Optional
from fractal_tasks_core.lib_write import write_table
from pydantic.decorator import validate_arguments


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

@validate_arguments
def aggregate_tables_to_well_level(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        input_paths: Sequence[str],
        output_path: str,
        component: str,
        metadata: Dict[str, Any],
        # Task-specific arguments:
        input_table_name: str,
        output_table_name: Optional[str] = None,
        overwrite: bool = True

) -> None:
    """
    Aggregate acquisition (image) based features to the well level.

    Args:
        input_paths: Path to the parent folder of the NGFF image.
            This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This argument is not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path of the NGFF image, relative to `input_paths[0]`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This argument is not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        input_table_name: Name of the feature table.
        output_table_name: Name of the aggregated feature table.
        overwrite: If True, overwrite existing label image.
    """

    logger.info(
        f"Aggregating features from feature table {input_table_name} to well level.")

    # collect paths to all feature tables
    in_path = input_paths[0]
    well_zarr_path = f"{in_path}/{component}"
    well_group = zarr.open_group(well_zarr_path, mode="r+")
    image_paths = [image["path"] for image in
                   well_group.attrs["well"]["images"]]

    feature_tables = [
        ad.read_zarr(
            f"{in_path}/{component}/{image}/tables/{input_table_name}"
        ) for image in image_paths
    ]

    # concatenate feature tables
    well_table = ad.concat(feature_tables, axis=1)
    well_table.var_names_make_unique(join="trash")

    vars_to_keep = [v for v in well_table.var_names if not "trash" in v]
    morphology = [v for v in vars_to_keep if "Morphology" in v]
    intensity = [v for v in vars_to_keep if "Intensity" in v]
    texture = [v for v in vars_to_keep if "Texture" in v]

    vars_to_keep = morphology + intensity + texture

    well_table = well_table[:, vars_to_keep]


    well_zarr_path = f"{output_path}/{component}"
    well_group = zarr.open_group(well_zarr_path, mode="r+")
    # Write to zarr group
    write_table(
        well_group,
        output_table_name,
        well_table,
        overwrite=overwrite,
        logger=logger,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=aggregate_tables_to_well_level,
        logger_name=logger.name,
    )

