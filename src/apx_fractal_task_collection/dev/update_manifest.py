"""Generate JSON schemas for tasks and write them to the Fractal manifest."""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "apx_fractal_task_collection"
    create_manifest(package=PACKAGE)