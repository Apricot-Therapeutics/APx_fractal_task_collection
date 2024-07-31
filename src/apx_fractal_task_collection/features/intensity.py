from skimage.measure import regionprops_table
import numpy as np
import pandas as pd

def sum_intensity(regionmask, intensity_image):
    return np.sum(intensity_image[regionmask])

def std_intensity(regionmask, intensity_image):
    return np.std(intensity_image[regionmask])

def measure_intensity_features(label_image, intensity_image):
    """
    Measure intensity features for label image.
    """
    intensity_features = pd.DataFrame(
        regionprops_table(
            np.squeeze(label_image),
            intensity_image,
            properties=[
                "label",
                "max_intensity",
                "mean_intensity",
                "min_intensity",
                #"weighted_moments_hu",
            ],
            extra_properties=[
                sum_intensity,
                std_intensity
            ],
        )
    )

    return intensity_features


def object_intensities(regionmask, intensity_image):
    return intensity_image[regionmask]


def object_intensity_correlation(labels: np.ndarray,
                                 ref_img: np.ndarray,
                                 img: np.ndarray):
    '''
    Calculate the correlation between the pixel intensities of objects in two
    images (for example the same channel in two different acquisitions).

    Args:
        labels: label image containing the objects
        ref_img: intensity image containing the pixel intensities of the reference image
        img: intensity image containing the pixel intensities of the second image
    '''

    intensities_1 = pd.DataFrame(
        regionprops_table(np.squeeze(labels),
                          intensity_image=np.squeeze(ref_img),
                          properties=['label'],
                          extra_properties=[object_intensities]))

    intensities_2 = pd.DataFrame(
        regionprops_table(np.squeeze(labels),
                          intensity_image=np.squeeze(img),
                          properties=['label'],
                          extra_properties=[object_intensities]))

    intensities = pd.merge(intensities_1, intensities_2, on='label',
                           suffixes=('_ref_img', '_img'))

    correlation = intensities.apply(lambda x: np.corrcoef(
        x['object_intensities_ref_img'], x['object_intensities_img'])[0, 1],
                                    axis=1)

    correlation = pd.DataFrame(
        {'label': intensities['label'], 'intensity_correlation': correlation})

    return correlation

