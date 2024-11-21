"""Fractal Task list for Fractal Helper Tasks."""

from fractal_tasks_core.dev.task_models import (NonParallelTask,
                                                ParallelTask,
                                                CompoundTask)

TASK_LIST = [
    ParallelTask(
        name="Measure Features",
        executable="tasks/measure_features.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Calculate Pixel Intensity Correlation",
        executable_init="tasks/init_calculate_pixel_intensity_correlation.py",
        executable="tasks/calculate_pixel_intensity_correlation.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Segment Secondary Objects",
        executable_init="tasks/init_segment_secondary_objects.py",
        executable="tasks/segment_secondary_objects.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Expand Labels",
        executable_init="tasks/init_expand_labels.py",
        executable="tasks/expand_labels_skimage.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Convert IC6000 to OME-Zarr",
        executable_init="tasks/init_convert_IC6000_to_ome_zarr.py",
        executable="tasks/convert_IC6000_to_ome_zarr.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Label Assignment by Overlap",
        executable_init="tasks/init_label_assignment_by_overlap.py",
        executable="tasks/label_assignment_by_overlap.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Clip Label Image",
        executable_init="tasks/init_clip_label_image.py",
        executable="tasks/clip_label_image.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Mask Label Image",
        executable_init="tasks/init_mask_label_image.py",
        executable="tasks/mask_label_image.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Filter Label by Size",
        executable_init="tasks/init_filter_label_by_size.py",
        executable="tasks/filter_label_by_size.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Calculate BaSiCPy Illumination Models",
        executable_init="tasks/init_calculate_basicpy_illumination_models.py",
        executable="tasks/calculate_basicpy_illumination_models.py",
        meta={"cpus_per_task": 1, "mem": 10000},
    ),
    ParallelTask(
        name="Apply BaSiCPy Illumination Models",
        input_types=dict(illumination_corrected=False),
        executable="tasks/apply_basicpy_illumination_models.py",
        output_types=dict(illumination_corrected=True),
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Aggregate Feature Tables",
        executable_init="tasks/init_aggregate_feature_tables.py",
        executable="tasks/aggregate_feature_tables.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    ParallelTask(
        name="Stitch FOVs with Overlap",
        input_types=dict(stitched=False),
        executable="tasks/stitch_fovs_with_overlap.py",
        output_types=dict(stitched=True),
        meta={"cpus_per_task": 1, "mem": 30000},
    ),
    NonParallelTask(
        name="Multiplexed Pixel Clustering",
        executable="tasks/multiplexed_pixel_clustering.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Correct Chromatic Shift",
        input_types=dict(chromatic_shift_corrected=False),
        executable_init="tasks/init_correct_chromatic_shift.py",
        executable="tasks/correct_chromatic_shift.py",
        output_types=dict(chromatic_shift_corrected=True),
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Convert Channel to Label",
        executable_init="tasks/init_convert_channel_to_label.py",
        executable="tasks/convert_channel_to_label.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Detect Blob Centroids",
        executable_init="tasks/init_detect_blob_centroids.py",
        executable="tasks/detect_blob_centroids.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Ashlar Stitching and Registration",
        input_types=dict(stitched=False),
        executable_init="tasks/init_ashlar_stitching_and_registration.py",
        executable="tasks/ashlar_stitching_and_registration.py",
        output_types=dict(stitched=True),
        meta={"cpus_per_task": 1, "mem": 15000},
    ),
    ParallelTask(
            name="Merge Plate Metadata",
            executable="tasks/merge_plate_metadata.py",
            meta={"cpus_per_task": 1, "mem": 3750},
        ),
]
