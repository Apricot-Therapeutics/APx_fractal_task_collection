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
                "weighted_moments_hu",
            ],
            extra_properties=[
                sum_intensity,
                std_intensity
            ],
        )
    )
    return intensity_features
