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
        name="Segment Secondary Objects",
        executable_init="tasks/init_segment_secondary_objects.py",
        executable="tasks/segment_secondary_objects.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    CompoundTask(
        name="Convert IC600 to OME-Zarr",
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
        name="Calculate BaSiCPy Illumination Models",
        executable_init="tasks/init_calculate_basicpy_illumination_models.py",
        executable="tasks/calculate_basicpy_illumination_models.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
    ParallelTask(
        name="Apply BaSiCPy Illumination Models",
        executable="tasks/apply_basicpy_illumination_models.py",
        meta={"cpus_per_task": 1, "mem": 3750},
    ),
]
