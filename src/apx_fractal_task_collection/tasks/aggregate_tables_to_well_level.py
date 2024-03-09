# Original authors:
# Adrian Tschan <adrian.tschan@uzh.ch>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.

import logging
from typing import Any, Dict, Sequence
import warnings

import fractal_tasks_core
import zarr
import anndata as ad
from fractal_tasks_core.tables import write_table
from pydantic.decorator import validate_arguments


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

def concat_features(feature_tables):

    well_table = ad.concat(feature_tables, axis=1, merge="same")
    well_table.var_names_make_unique(join="trash")

    vars_to_keep = [v for v in well_table.var_names if "trash" not in v]
    morphology = [v for v in vars_to_keep if "Morphology" in v]
    intensity = [v for v in vars_to_keep if "Intensity" in v]
    texture = [v for v in vars_to_keep if "Texture" in v]
    population = [v for v in vars_to_keep if "Population" in v]

    vars_to_keep = morphology + intensity + texture + population

    feature_table = well_table[:, vars_to_keep]

    return feature_table

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
        output_table_name: str,
        output_component: str = 'image',
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
        tables_to_merge: List of feature tables to merge into the main
            feature table. For example, if the input feature table is the table
            for cells, tables to merge could include nuclei and cytoplasm.
            Only use this option if you ran Label Assignment by Overlap first.
        output_component: In which component to store the aggregated feature
            table. Can take values "image" (the table will be saved in the 
            first image/acquisition (0) folder) or "well" (the table will be 
            saved in the well folder).
        overwrite: If True, overwrite existing feature table.
    """

    logger.info(
        f"Aggregating features from feature table"
        f" {input_table_name} to well level.")

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

    well_table = concat_features(feature_tables)

    if output_component == "well":
        out_zarr_path = f"{output_path}/{component}"
    elif output_component == "image":
        out_zarr_path = f"{output_path}/{component}/{image_paths[0]}"
    else:
        warnings.warn(f"Unknown output component {output_component}."
                      f" Please choose between 'image' and 'well'.")
        pass

    out_group = zarr.open_group(out_zarr_path, mode="r+")

    # get original table attributes
    table_group = zarr.open_group(
        f"{in_path}/{component}/0/tables/{input_table_name}",
        mode='r')
    orig_attrs = table_group.attrs.asdict()

    if output_component == "well":
        # update orig_attrs
        orig_attrs["region"]["path"] = \
            orig_attrs["region"]["path"].replace("../../", "../")

    # Write to zarr group
    write_table(
        out_group,
        output_table_name,
        well_table,
        overwrite=overwrite,
        table_attrs=orig_attrs,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=aggregate_tables_to_well_level,
        logger_name=logger.name,
    )

