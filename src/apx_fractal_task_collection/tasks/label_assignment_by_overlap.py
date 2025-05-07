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

import anndata as ad
import fractal_tasks_core
import numpy as np
import pandas as pd
import zarr
import dask.array as da
from skimage.measure import regionprops_table
from pathlib import Path

from apx_fractal_task_collection.io_models import InitArgsLabelAssignment
from fractal_tasks_core.tables import write_table

from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


def label_overlap(regionmask, intensity_image):
    """
    Scikit-image regionprops_table extra_properties function to compute
    max overlap between two label images.
    """
    parent_labels = np.where(regionmask > 0, intensity_image, 0)

    labels, counts = np.unique(parent_labels[parent_labels>0],
                               return_counts=True)

    if len(labels > 0):
        # if there is a tie in the overlap, the first label is selected
        max_label = labels[np.argmax(counts)]
        max_label_area = counts.max()
        child_area = regionmask[regionmask>0].size
        overlap = max_label_area/child_area
    else:
        max_label = np.nan
        overlap = np.nan

    return [max_label, overlap]


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

    t = pd.DataFrame(regionprops_table(child_label,
                                       parent_label,
                                       properties=['label'],
                                       extra_properties=[label_overlap]))

    t.columns = ['child_label', 'parent_label', 'overlap']
    t.loc[t.overlap < overlap_threshold, 'parent_label'] = np.nan
    t['parent_label'] = t['parent_label'].astype('Int32')
    t.set_index('child_label', inplace=True)

    return t


@validate_call
def label_assignment_by_overlap(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsLabelAssignment,
    # Task-specific arguments:
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
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_label_assignment_by_overlap`.
        child_table_name: Name of the feature table associated with
            the child label image.
        level: Resolution of the label image to calculate overlap.
            Only tested for level 0.
        overlap_threshold: The minimum percentage (between 0 and 1) of child
            label object that must be contained in parent label object to
             be considered a match.
    """
    
    parent_label_name = init_args.parent_label_name
    child_label_name = init_args.child_label_name
    parent_label_zarr_url = init_args.parent_label_zarr_url
    child_label_zarr_url = init_args.child_label_zarr_url

    # load parent and child label image
    parent_label = da.from_zarr(
        f"{parent_label_zarr_url}/labels/"
        f"{parent_label_name}/{level}")
    
    child_label = da.from_zarr(
        f"{child_label_zarr_url}/labels/"
        f"{child_label_name}/{level}")

    # if there are no child labels, assignments will be all NaN
    if np.unique(child_label).compute().size == 1:
        assignments = pd.DataFrame({'parent_label': pd.NA,
                                    'overlap': pd.NA}, index=pd.Index([]))

    else:
        # make the assignment
        assignments = assign_objects(parent_label.compute(),
                                     child_label.compute(),
                                     overlap_threshold,
                                     )

    assignments.rename(
        columns={'parent_label': f'{parent_label_name}_label',
                 'overlap': f'{child_label_name}_{parent_label_name}_overlap'},
    inplace=True)

    # loop over images and write the assignments to the child table
    well_group = zarr.open_group(zarr_url, mode='r')
    for image in well_group.attrs['well']["images"]:

        # define path to feature table
        child_feature_path = Path(zarr_url).joinpath(image['path'],
                                                     'tables',
                                                     child_table_name)

        # load the child feature table
        try:
            child_features = ad.read_zarr(child_feature_path)
        except:
            continue

        if f'{parent_label_name}_label' in child_features.obs.columns:
            child_features.obs.drop(
                columns=[f'{parent_label_name}_label',
                         f'{child_label_name}_{parent_label_name}_overlap'],
                                    inplace=True)

        # remove these two columns that snuck in at some point (to make
        # sure we don't have them in the output in case a user re-runs this tasks
        # on the same data)
        if 'parent_area' in child_features.obs.columns and\
                'child_area' in child_features.obs.columns:
            child_features.obs.drop(
                columns=['parent_area', 'child_area'], inplace=True)


        # merge with child feature obs data
        merged_data = pd.merge(child_features.obs,
                               assignments,
                               left_on='label',
                               right_index=True,
                               how='left')

        child_features.obs = merged_data

        # read original attributes of child table
        child_table_group = zarr.open_group(child_feature_path, mode='r')
        orig_attrs = child_table_group.attrs.asdict()

        img_group = zarr.group(child_feature_path.parents[1])
        write_table(img_group,
                    child_table_name,
                    child_features,
                    overwrite=True,
                    table_attrs=orig_attrs,
                    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=label_assignment_by_overlap,
        logger_name=logger.name,
    )
