import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import anndata as ad
import dask.array as da
import fractal_tasks_core
import numpy as np
import pandas as pd
import zarr
from typing import Optional
from skimage.measure import regionprops_table
from fractal_tasks_core.lib_write import write_table

from pydantic.decorator import validate_arguments


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

def assign_objects(
        parent_label: np.ndarray,
        child_label: np.ndarray,
        overlap_threshold=1.0,
) -> pd.DataFrame:
    """
    Calculate the overlap between labels in label_a and label_b,
        and return a DataFrame of matching labels.
    label_a:  4D numpy array.
    label_b:  4D numpy array.
    overlap_threshold: float, the minimum fraction of child label object that
        must be contained in parent label object to be considered a match.
    """
    parent_label = np.squeeze(parent_label)
    child_label = np.squeeze(child_label)
    t = pd.DataFrame(regionprops_table(child_label, parent_label,
                                       properties=[
                                       'label', 'image_intensity', 'area']))

    res = []
    for i, b in t.iterrows():
        sub_region_df = pd.DataFrame(regionprops_table(b.image_intensity,
                                       properties=['label', 'area']))

        sub_region_df.rename(columns={'label': 'parent_label', 'area': 'parent_area'}, inplace=True)
        sub_region_df['child_label'] = b.label
        sub_region_df['child_area'] = b.area

        res.append(sub_region_df)

    res_merged = pd.concat(res, axis=0)
    res_merged['overlap'] = res_merged['parent_area']/res_merged['child_area']
    # keep only parent with highest overlap
    res_merged = res_merged.groupby('child_label', as_index=False).apply(lambda x: x.loc[x.overlap == x.overlap.max()])
    res_merged.set_index('child_label', inplace=True)
    res_merged.loc[res_merged.overlap < overlap_threshold, 'parent_label'] = pd.NA

    return res_merged[['parent_label', 'overlap']]


@validate_arguments
def label_assignment_by_overlap(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments:
    parent_label_image: str,
    child_label_image: str,
    parent_label_cycle: Optional[int] = None,
    child_label_cycle: Optional[int] = None,
    child_table_name: str,
    level: int = 0,
    overlap_threshold: float = 1.0,
):
    """
    Assign labels to each other based on overlap.

    Takes a parent label image and a child label image and calculates
    overlaps between their labels. Child labels will be assigned to parent
    labels based on an overlap threshold.

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: dictionary containing metadata about the OME-Zarr. This task
            requires the following elements to be present in the metadata.
            `coarsening_xy (int)`: coarsening factor in XY of the downsampling
            when building the pyramid. (standard argument for Fractal tasks,
            managed by Fractal server).
        parent_label_image: Name of the parent label image.
            Needs to exist in OME-Zarr file.
        child_label_image: Name of the child label image.
            Needs to exist in OME-Zarr file.
        parent_label_cycle: indicates which cycle contains the parent label image (only needed if multiplexed).
        child_label_cycle: indicates which cycle contains the child label image (only needed if multiplexed).
        child_table_name: Name of the feature table associated with
            the child label image.
        level: Resolution of the label image to calculate overlap.
            Only tested for level 0.
        overlap_threshold: The minimum percentage (between 0 and 1) of child
            label object that must be contained in parent label object to
             be considered a match.
    """
    if parent_label_cycle:
    # update the component for the label image
        parts = component.rsplit("/", 1)
        parent_label_component = parts[0] + "/" + str(parent_label_cycle)
        child_label_component = parts[0] + "/" + str(child_label_cycle)
    else:
        parent_label_component = component
        child_label_component = component

    # define path to feature table
    child_feature_path = f"{Path(output_path)}/{component}/tables/{child_table_name}"

    if Path(child_feature_path).is_dir():
        in_path = Path(input_paths[0]).as_posix()
        parent_label = da.from_zarr(
            f"{in_path}/{parent_label_component}/labels/{parent_label_image}/{level}"
        )

        # load the parent label image
        parent_label = parent_label.compute()

        child_label = da.from_zarr(
            f"{in_path}/{child_label_component}/labels/{child_label_image}/{level}"
        )

        # load the child labe image
        child_label = child_label.compute()
        # load the child feature table
        child_features = ad.read_zarr(child_feature_path)
        # make the assignment
        assignments = assign_objects(parent_label,
                                     child_label,
                                     overlap_threshold,
                                     )

        assignments.rename(
            columns={'parent_label': f'{parent_label_image}_label',
                     'overlap': f'{child_label_image}_{parent_label_image}_overlap'},
        inplace=True)
        # merge with child feature obs data
        merged_data = child_features.obs.merge(assignments, left_on='label',
                                               right_index=True,
                                               how='left')
        merged_data[f'{parent_label_image}_label'] = merged_data[f'{parent_label_image}_label'].astype('Int32')

        child_features.obs = merged_data

        image_group = zarr.group(f"{in_path}/{component}")
        write_table(image_group, child_table_name,
                    child_features, overwrite=True)
    else:
        pass


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=label_assignment_by_overlap,
        logger_name=logger.name,
    )
