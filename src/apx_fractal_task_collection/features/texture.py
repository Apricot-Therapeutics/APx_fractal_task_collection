from skimage.measure import regionprops_table
import mahotas as mh
import numpy as np
import pandas as pd

def haralick_features(regionmask, intensity_image):
    haralick_values_list = []
    masked_image = np.where(regionmask > 0, intensity_image, 0)
    for distance in [2, 5]:
        try:
            haralick_values = mh.features.haralick(
                masked_image.astype('uint8'),
                distance=distance,
                return_mean=True,
                ignore_zeros=True)
        except ValueError:
            haralick_values = np.full(13, np.NaN, dtype=float)

        haralick_values_list.extend(haralick_values)
    return haralick_values_list


def measure_texture_features(label_image, intensity_image):
    """
    Measure texture features for label image.
    """

    # NOTE: Haralick features are computed on 8-bit images.
    clip_value = np.percentile(intensity_image, 99.999)
    clipped_img = np.clip(intensity_image, 0, clip_value).astype('uint16')
    rescaled_img = mh.stretch(clipped_img)

    names = ['angular-second-moment', 'contrast', 'correlation',
             'sum-of-squares', 'inverse-diff-moment', 'sum-avg',
             'sum-var', 'sum-entropy', 'entropy', 'diff-var',
             'diff-entropy', 'info-measure-corr-1', 'info-measure-corr-2']

    names = [
        f"Haralick-{name}-{distance}" for distance in [2, 5] for name in names]

    texture_features = pd.DataFrame(
        regionprops_table(label_image,
                          rescaled_img,
                          properties=['label'],
                          extra_properties=[haralick_features]))

    texture_features.set_index('label', inplace=True)
    texture_features.columns = names
    texture_features.reset_index(inplace=True)
    return texture_features

