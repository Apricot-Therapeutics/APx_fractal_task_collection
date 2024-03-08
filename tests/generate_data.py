import numpy as np
import zarr

from ome_zarr.io import parse_url
from pathlib import Path
from ome_zarr.writer import (write_image,
                            write_labels,
                             write_plate_metadata,
                             write_well_metadata)

from fractal_tasks_core.tables import write_table
import anndata as ad
import pandas as pd
from skimage.data import binary_blobs
from skimage.morphology import label


def main() -> None:
    """
    Generate some NGFF data.
    """
    path = Path(Path(__file__).parent).joinpath("data/hcs_ngff.zarr")
    row_names = ["A", "B"]
    col_names = ["1", "2", "3"]
    well_paths = ["A/2", "B/3"]
    field_paths = ["0", "1", "2"]

    # generate data
    mean_val = 10
    num_wells = len(well_paths)
    num_fields = len(field_paths)
    size_xy = 128
    size_z = 1
    size_c = 2
    num_labels = 2
    rng = np.random.default_rng(0)
    data = rng.poisson(mean_val, size=(num_wells,
                                       num_fields,
                                       size_c,
                                       size_z,
                                       size_xy,
                                       size_xy)).astype(np.uint8)

    # write the plate of images and corresponding metadata
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store, overwrite=True)
    write_plate_metadata(root, row_names, col_names, well_paths)
    for wi, wp in enumerate(well_paths):
        row, col = wp.split("/")
        row_group = root.require_group(row)
        well_group = row_group.require_group(col)
        write_well_metadata(well_group, field_paths)
        for fi, field in enumerate(field_paths):
            image_group = well_group.require_group(str(field))
            write_image(image=data[wi, fi], group=image_group, axes="czyx",
                        storage_options=dict(chunks=(1, size_xy, size_xy)))

            image_group.attrs["omero"] = {
                "channels": [
                    {
                        "color": "00FFFF",
                        "window": {"start": 0, "end": 20, "min": 0,
                                   "max": 255},
                        "label": f"{fi}_DAPI",
                        "active": True,
                        "wavelength_id": 'UV - DAPI'
                    },
                    {
                        "color": "008000",
                        "window": {"start": 0, "end": 20, "min": 0,
                                   "max": 255},
                        "label": f"{fi}_GFP",
                        "active": True,
                        "wavelength_id": 'Blue - FITC'
                    }
                ]
            }

            # add tables
            var = ["x_micrometer", "y_micrometer", "z_micrometer",
                   "len_x_micrometer", "len_y_micrometer", "len_z_micrometer",
                   "x_micrometer_original", "y_micrometer_original"]
            FOV_obs = ["FOV_1"]
            well_obs = ["well_1"]

            FOV_table = pd.DataFrame(
                np.array([[0, 0, 0, size_xy, size_xy, 1, 0, 0]]),
                columns=var, index=FOV_obs)

            well_table = pd.DataFrame(
                np.array([[0, 0, 0, size_xy, size_xy, 1]]),
                columns=var[0:-2], index=well_obs)

            FOV_table = ad.AnnData(FOV_table)
            well_table = ad.AnnData(well_table)

            write_table(
                image_group,
                "FOV_ROI_table",
                FOV_table,
                overwrite=True,
                table_type="roi_table",
            )

            write_table(
                image_group,
                "well_ROI_table",
                well_table,
                overwrite=True,
                table_type="roi_table",
            )

            if fi == 0:
                # add labels...
                blobs = [binary_blobs(length=size_xy,
                                      volume_fraction=0.6,
                                      blob_size_fraction=0.07,
                                      n_dim=3).astype('uint8')
                         for n in range(0, num_labels)]
                label_images = [label(b[:size_z, :, :]) for b in blobs]
                label_names = ["Label A", "Label B"]
                for i, label_name in enumerate(label_names):
                    write_labels(label_images[i], image_group, axes="zyx",
                                 name=label_name)

            if fi == 2:
                # add labels...
                blobs = [binary_blobs(length=size_xy,
                                      volume_fraction=0.6,
                                      blob_size_fraction=0.07,
                                      n_dim=3).astype('uint8')
                         for n in range(0, num_labels)]
                label_images = [label(b[:size_z, :, :]) for b in blobs]
                label_names = ["Label C", "Label D"]
                for i, label_name in enumerate(label_names):
                    write_labels(label_images[i], image_group, axes="zyx",
                                 name=label_name)

if __name__ == "__main__":
    main()