### Purpose
- Calculates **morphology**, **intensity**, **texture**, and **population features** for objects in a label image.
- Supports **2D and 3D measurements** across multiple regions of interest (ROIs).
- Extracts features for intensity images using configurable channel inclusion/exclusion.

### Outputs
- A **feature table** saved in the OME-Zarr structure, containing:
  - Morphology features (e.g., size, shape, well coordinates).
  - Intensity features (e.g., mean, max, min intensity per object).
  - Texture features (e.g., Haralick, Laws' texture energy).
  - Population features (e.g., densities and number of neighbours).
- Updated ROI metadata with border and well location information.

### Limitations
- Does not support measurements for label images that do not have the same resolution as the intensity images.