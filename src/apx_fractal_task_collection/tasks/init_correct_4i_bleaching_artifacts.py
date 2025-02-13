# Original authors:
# Adrian Tschan <atschan@apricotx.com>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Initializes the parallelization list for Correct 4i Bleaching Artifacts Task.
"""
import logging
from typing import Any, List
from enum import Enum
from functools import reduce
import pandas as pd
import string
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pydantic import validate_call
from pathlib import Path
import zarr
from anndata.experimental import read_elem
import anndata as ad

logger = logging.getLogger(__name__)


class MeanEstimator(Enum):
    """
    Enum for the mean estimator options.

    Attributes:
        mean: use arithmetic mean to estimate the control data mean value
        median: use median to estimate the control data mean value
    """

    mean = "mean"
    median = "median"


class Plate():
    '''
    Class to store metadata of a plate and which wells are imaged
    '''
    def __init__(self):
        self.initialize_wells()
        self.time_index = list(range(0, 384))


    def initialize_wells(self):
        self.wells = []
        rows = list(string.ascii_uppercase[0:16])
        for i_row, row in enumerate(rows):
            if i_row % 2 == 0:
                cols = list(range(1, 25))
            else:
                cols = list(range(24, 0, -1))
            for col in cols:
                self.wells.append(f"{row}{col:02d}")

    def sort_wells_by_imaging_time(self, imaged_wells):
        time_index = [self.wells.index(x) for x in imaged_wells]
        time_sorted_wells =  [val for _, val in sorted(zip(time_index, imaged_wells))]
        return time_sorted_wells

def fit_exp_nonlinear(t, y, bounds=([0, -1, 0], [100, 0, 100])):
    opt_parms, parm_cov = scipy.optimize.curve_fit(model_func, t, y,
                                                   maxfev=10000, bounds=bounds)
    A, K, C = opt_parms
    return A, K, C


def model_func(t, A, K, C):
    return A * np.exp(K * t) + C


def calculate_correction_factors(control_data,
                                 imaged_wells,
                                 mean_estimator,
                                 plot_results=False,
                                 output_path=None):
    '''
    Calculate the correction factor for each timepoint of each 
    intensity measurement
    
    Args:
        control_data: pd.DataFrame containing control data
    '''


    plate = Plate()
    time_index = list(range(0, len(imaged_wells)))
    wells = plate.sort_wells_by_imaging_time(imaged_wells)

    time_df = pd.DataFrame({'well_name': wells, 'time': time_index})
    control_data = pd.merge(control_data, time_df, on='well_name', how='left')

    if mean_estimator == MeanEstimator.median:
        averaged_df = control_data.groupby(['well_name', 'time']).median()

    elif mean_estimator == MeanEstimator.mean:
        averaged_df = control_data.groupby(['well_name', 'time']).mean()

    intensity_columns = [c for c in averaged_df.columns if 'Intensity' in c]

    # minmax scaling
    min_ctrl_time = control_data['time'].min()
    averaged_df = averaged_df.div(averaged_df.loc[
                                      averaged_df.index.get_level_values(
                                          'time') == min_ctrl_time].values)


    averaged_df.reset_index(inplace=True)
    decay_model_params = {}

    bounds = ([0, -1, 0], [1, 0, 1])
    for col in intensity_columns:
        try:
            A, K, C = fit_exp_nonlinear(averaged_df['time'], averaged_df[col],
                                        bounds=bounds)
            decay_model_params[col] = {'A': A, 'K': K, 'C': C}
        except ValueError:
            logger.warning(f"Could not fit decay model for {col}")

    decay_model_scale_factors = {}

    for col in decay_model_params.keys():
        data_fit = model_func(np.array(time_index),
                              decay_model_params[col]['A'],
                              decay_model_params[col]['K'],
                              decay_model_params[col]['C'])
        decay_model_scale_factors[col] = data_fit

    decay_model_scale_factors_df = pd.DataFrame(decay_model_scale_factors)
    decay_model_scale_factors_df['well_name'] = time_df['well_name']

    if plot_results:
        plt.rc('figure', figsize=(6, 4))
        for col in decay_model_params.keys():
            plt.figure()
            A, K, C = decay_model_params[col]['A'], decay_model_params[col]['K'], \
                decay_model_params[col]['C']
            ax = sns.scatterplot(data=averaged_df, x='time', y=col, color='black')
            sns.lineplot(x=averaged_df['time'],
                         y=model_func(averaged_df['time'], A, K, C), color='red',
                         linestyle='--')
            ax.set(xlabel='Imaging Time Index',
                   ylabel=f'{col.split("_intensity_")[-1]} Intensity',
                   ylim=(0, 2.0))

            # make a subfolder plots in the output_path
            output_dir = Path(output_path).joinpath('plots')
            output_dir.mkdir(exist_ok=True)

            plt.savefig(output_dir.joinpath(f'{col}.png'), dpi=300,
                        bbox_inches='tight')

    model_df = pd.DataFrame(decay_model_params).T

    return model_df, decay_model_scale_factors_df



@validate_call
def init_correct_4i_bleaching_artifacts(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    condition_column: str = "condition",
    control_condition: str,
    mean_estimator: MeanEstimator = MeanEstimator.median,
    feature_table_name: str,
    additional_control_filters: dict[str, str] = None,
    plot_results: bool = True,
    model_output_dir: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Initializes normalize feature table task


    This task prepares a parallelization list of all zarr_urls that need to be
    used to perform correction of 4i bleaching artifacts.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
        condition_column: Name of the column in the feature table that contains
            the condition information.
        control_condition: Name of the condition to be used for
            correction.
        mean_estimator: Method to be used for estimating the mean of the control
            population. Choices are:
            - mean: use arithmetic mean to estimate the control data mean value
            - median: use median to estimate the control data mean value
        feature_table_name: Name of the feature table that contains the
            measurements to be corrected.
        additional_control_filters: Dictionary of additional metadata filters
            to be applied to filter. The dictionary should be
            formatted as: { "column_name": "value",}.
        plot_results: Whether to plot the results of the decay model fit.
        model_output_dir: Directory where to save the model parameters and
            scale factors.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(
        f"Running `init_correct_4i_bleaching_artifacts.py` for {zarr_urls=}"
    )

    # append feature table name to model_output_dir
    model_output_dir = f"{model_output_dir}/{feature_table_name}"

    # make sure it exists
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    # filter zarr-urls to only include zarrs that have the feature table
    # (for example, aggregated tables are not present in all zarr files)
    zarr_urls = [zarr_url for zarr_url in zarr_urls if
                 Path(f"{zarr_url}/tables/{feature_table_name}").exists()]

    # for each zarr_url in zarr_urls, load a sample and get the condition
    condition_cycle_dict = {'zarr_url': [],
                            'condition': [],
                            'cycle': [],
                            'row': [],
                            'col': []}

    # add columns from additional_control_filters to the dict
    for col in additional_control_filters:
        condition_cycle_dict[col] = []

    zarr_paths = [Path(z).parents[2] for z in zarr_urls]
    zarr_path = np.unique(zarr_paths)

    if len(zarr_path) > 1:
        raise ValueError("This task currently only supports single plates")

    # using regex, find all subfolders with format '/[A-Z]/[0-9]{2}' in zarr dir
    imaged_wells = list(zarr_path[0].glob(
        pattern="[A-Z]/[0-9][0-9]/"))
    imaged_wells = [well.parent.name + well.name for well in imaged_wells]

    logger.info(f"Found {len(imaged_wells)} imaged wells in the plate "
                f"at {zarr_path[0]}")

    for zarr_url in zarr_urls:
        # get the condition
        zarr_store = zarr.open(f"{zarr_url}/tables/{feature_table_name}",
                               mode="r")
        condition = read_elem(zarr_store[f"obs/{condition_column}"])[0]

        # get the cycle
        cycle_path = Path(zarr_url).name
        row = Path(zarr_url).parents[1].name
        col = Path(zarr_url).parent.name

        condition_cycle_dict['zarr_url'].append(zarr_url)
        condition_cycle_dict['condition'].append(condition)
        condition_cycle_dict['cycle'].append(cycle_path)
        condition_cycle_dict['row'].append(row)
        condition_cycle_dict['col'].append(col)

        # get additional control filters
        for col, value in additional_control_filters.items():
            column_value = read_elem(zarr_store[f"obs/{col}"])[0]
            condition_cycle_dict[col].append(column_value)


    condition_cycle_df = pd.DataFrame(condition_cycle_dict)

    # filter df to only include control conditions
    ctrl_df = condition_cycle_df.loc[
        condition_cycle_df['condition'] == control_condition]

    # if additional control filters are provided, filter the control df
    if additional_control_filters:
        for col, value in additional_control_filters.items():
            ctrl_df = ctrl_df.loc[ctrl_df[col] == value]

    cycle_scale_factors = {}

    model_dfs = []
    scale_factor_dfs = []

    # for each unique imaging cycle, read all control tables
    for cycle in ctrl_df['cycle'].unique():
        logger.info(f"Calculating correction factors for cycle {cycle}")
        cycle_ctrl_df = ctrl_df.loc[ctrl_df['cycle'] == cycle]
        cycle_ctrl_zarr_urls = cycle_ctrl_df['zarr_url'].tolist()
        cycle_ctrl_data = ad.concat([ad.read_zarr(
            f"{zarr_url}/tables/{feature_table_name}")
            for zarr_url in cycle_ctrl_zarr_urls])

        cycle_ctrl_df = cycle_ctrl_data.to_df()
        cycle_ctrl_df['well_name'] = cycle_ctrl_data.obs['well_name']
        #cycle_ctrl_df.reset_index(inplace=True)
        
        # get model_df and decay_model_scale_factors_df
        model_df, scale_factors_df = calculate_correction_factors(
            cycle_ctrl_df,
            imaged_wells,
            mean_estimator,
            plot_results,
            output_path=model_output_dir)

        # append to model_dfs and scale_factor_dfs
        model_dfs.append(model_df)
        scale_factor_dfs.append(scale_factors_df)
        cycle_scale_factors[cycle] = scale_factors_df

    # save model_df and scale_factors_df
    model_df = pd.concat(model_dfs)
    model_df.to_csv(f"{model_output_dir}/decay_model_params.csv")

    scale_factors_df = reduce(lambda x, y: pd.merge(x, y, on='well_name'),
                              scale_factor_dfs)
    scale_factors_df.set_index('well_name', inplace=True)

    scale_factors_df.to_csv(
        f"{model_output_dir}/decay_model_scale_factors.csv")

    # Create the parallelization list
    parallelization_list = []

    for i, row in condition_cycle_df.iterrows():
        # get the scale factors for the cycle
        cycle_scale_factors_df = cycle_scale_factors[row['cycle']]
        well_name = row['row'] + row['col']
        current_scale_factors = cycle_scale_factors_df.loc[
            cycle_scale_factors_df['well_name'] == well_name]
        current_scale_factors.drop(columns='well_name', inplace=True)

        parallelization_list.append(
            dict(
                zarr_url=row.zarr_url,
                init_args=dict(
                    current_scale_factors=current_scale_factors.to_dict(),
                    feature_table_name=feature_table_name,
                ),
            )
        )

    return dict(parallelization_list=parallelization_list)

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_correct_4i_bleaching_artifacts,
        logger_name=logger.name,
    )