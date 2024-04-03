from skimage.measure import regionprops_table
import logging
import mahotas as mh
import numpy as np
import pandas as pd
import pyfeats

logger = logging.getLogger(__name__)

def haralick_features(regionmask, intensity_image):
    """
    Scikit-image regionprops_table extra_properties function to compute
    Haralick features for a region.
    """
    masked_image = np.where(regionmask > 0, intensity_image, 0)
    haralick_values_list = []
    for distance in [2, 5]:
        try:
            haralick_values = mh.features.haralick(
                masked_image,
                distance=distance,
                return_mean_ptp=True,
                ignore_zeros=True)
        except ValueError:
            haralick_values = np.full(26, np.NaN, dtype=float)
        except TypeError:
            haralick_values = np.full(26, np.NaN, dtype=float)

        haralick_values_list.extend(haralick_values)
    return haralick_values_list


def measure_haralick_features(label_image, intensity_image, clip_value=10000):
    """
    Measure Haralick features for label image and intensity image.
    """

    # NOTE: Haralick features are computed on 8-bit images.
    rescaled_image = np.clip(intensity_image, 0, clip_value)
    # 8-bit rescaling, but only to 254 to avoid overflow
    # because we add 1 to the rescaled image to avoid 0 values
    # in the intensities (we will ignore 0 values in the haralick features
    # computation to not include regions outside of mask)
    rescaled_image = rescaled_image/clip_value*254
    rescaled_image = rescaled_image.astype('uint8')
    rescaled_image = rescaled_image+1


    names = ['angular-second-moment', 'contrast', 'correlation',
         'sum-of-squares', 'inverse-diff-moment', 'sum-avg',
         'sum-var', 'sum-entropy', 'entropy', 'diff-var',
         'diff-entropy', 'info-measure-corr-1', 'info-measure-corr-2']

    names = [f"Haralick-Mean-{name}" for name in names] \
             + [f"Haralick-Range-{name}" for name in names]

    names = [f"{name}-{distance}" for distance in [2, 5] for name in names]



    features = pd.DataFrame(
        regionprops_table(label_image,
                          rescaled_image,
                          properties=['label'],
                          extra_properties=[haralick_features]))

    features.set_index('label', inplace=True)
    features.columns = names
    return features


def lte_features(label_image, intensity_image):
    """
    Scikit-image regionprops_table extra_properties function to compute
    Law's Texture Energy (LTE) features for a region.
    """
    try:
        features, labels = pyfeats.lte_measures(intensity_image, label_image,
                                                l=3)
    except ValueError:
        features = np.full(6, np.NaN, dtype=float)
    return list(features)

def measure_lte_features(label_image, intensity_image):
    """
    Measure Law's Texture Energy Measures for label image.
    """
    features = pd.DataFrame(
        regionprops_table(label_image,
                          intensity_image,
                          properties=['label'],
                          extra_properties=[lte_features]))

    features.set_index('label', inplace=True)
    features.columns = ["LTE_LL",
                        "LTE_EE",
                        "LTE_SS",
                        "LTE_LE",
                        "LTE_ES",
                        "LTE_LS"]

    return features


def measure_texture_features(label_image,
                             intensity_image,
                             clip_value=10000,
                             feature_selection=["haralick", "lte"]):
    """
    Measure texture features for label image.
    """

    feature_list = []

    if "haralick" in feature_selection:
        logger.info("Measuring Haralick features")
        # get haralick features
        haralick_features = measure_haralick_features(
            label_image=label_image,
            intensity_image=intensity_image,
            clip_value=clip_value)
        feature_list.append(haralick_features)

    if "lte" in feature_selection:
        logger.info("Measuring Law's Texture Energy features")
        # get lte features
        lte_features = measure_lte_features(
            label_image=label_image,
            intensity_image=intensity_image)
        feature_list.append(lte_features)

    texture_features = pd.concat(feature_list, axis=1)
    texture_features.reset_index(inplace=True)
    return texture_features

