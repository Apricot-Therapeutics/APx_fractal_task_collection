from skimage.measure import regionprops_table
import mahotas as mh
import numpy as np
import pandas as pd


def roundness(regionmask):
    return mh.features.roundness(regionmask)


def measure_morphology_features(label_image):
    """
    Measure morphology features for label image.
    """
    morphology_features = pd.DataFrame(regionprops_table(
        np.squeeze(label_image),
        properties=[
            "label",
            "area",
            "centroid",
            "bbox_area",
            "bbox",
            "convex_area",
            "eccentricity",
            "equivalent_diameter",
            "euler_number",
            "extent",
            "filled_area",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "perimeter",
            "solidity",
        ],
        extra_properties=[
            roundness,
        ],
    )
    )
    morphology_features["circularity"] = (4*np.pi*morphology_features.area) \
                                         / (morphology_features.perimeter**2)
    return morphology_features


def get_borders_internal(well_table, fov_table, morphology_features,
                         pixel_size_xy):

    well_table = well_table.to_df()
    fov_table = fov_table.to_df()
    obj_name = morphology_features.columns[0].split("_Morphology")[0]

    def check_range(row, borders, col_start, col_end):
        start_value = row[col_start]
        end_value = row[col_end]
        range_values = np.arange(start_value, end_value + 1)
        is_in_range = np.isin(borders, range_values)
        return np.any(is_in_range)

    # get internal borders of FOVs
    borders_x = np.unique(
        np.round(
            (fov_table['x_micrometer'] / pixel_size_xy)).astype('uint16'))[1:] - \
                np.unique((well_table['x_micrometer'] / pixel_size_xy).astype(
                    'uint16'))[0]
    borders_y = np.unique(
        np.round(
        (fov_table['y_micrometer'] / pixel_size_xy)).astype('uint16'))[1:] - \
                np.unique((well_table['y_micrometer'] / pixel_size_xy).astype(
                    'uint16'))[0]

    e = morphology_features.apply(lambda x:
                              check_range(
                                  x,
                                  borders_y,
                                  f'{obj_name}_Morphology_bbox-0',
                                  f'{obj_name}_Morphology_bbox-2'),
                              axis=1)

    f = morphology_features.apply(lambda x:
                              check_range(
                                  x,
                                  borders_x,
                                  f'{obj_name}_Morphology_bbox-1',
                                  f'{obj_name}_Morphology_bbox-3'),
                              axis=1)

    is_border_internal = e | f

    return is_border_internal



def get_borders_external(ROI_table, morphology_features, pixel_size_xy):

    safety_range = 5
    ROI_df = ROI_table.to_df()
    obj_name = morphology_features.columns[0].split("_Morphology")[0]

    borders_x = np.unique(
        (ROI_df['x_micrometer'] / pixel_size_xy).astype('uint16')) - np.min(
        ROI_df['x_micrometer'] / pixel_size_xy).astype('uint16')
    borders_x = np.append(borders_x, borders_x[-1] + np.round(
        ROI_df['len_x_micrometer'] / pixel_size_xy).astype('uint16')[0])
    borders_x_start = np.arange(borders_x[0], borders_x[0] + safety_range)
    borders_x_end = np.arange(borders_x[-1] - safety_range, borders_x[-1])

    borders_x = np.append(borders_x, borders_x_start)
    borders_x = np.append(borders_x, borders_x_end)

    borders_y = np.unique(
        (ROI_df['y_micrometer'] / pixel_size_xy).astype('uint16')) - np.min(
        ROI_df['y_micrometer'] / pixel_size_xy).astype('uint16')
    borders_y = np.append(borders_y, borders_y[-1] + np.round(
        ROI_df['len_y_micrometer'] / pixel_size_xy).astype('uint16')[0])
    borders_y_start = np.arange(borders_y[0], borders_y[0] + safety_range)
    borders_y_end = np.arange(borders_y[-1] - safety_range, borders_y[-1])

    borders_y = np.append(borders_y, borders_y_start)
    borders_y = np.append(borders_y, borders_y_end)

    a = morphology_features[f'{obj_name}_Morphology_bbox-0'].isin(borders_y)
    b = morphology_features[f'{obj_name}_Morphology_bbox-1'].isin(borders_x)
    c = morphology_features[f'{obj_name}_Morphology_bbox-2'].isin(borders_y)
    d = morphology_features[f'{obj_name}_Morphology_bbox-3'].isin(borders_x)

    is_border_external = a | b | c | d

    return is_border_external
