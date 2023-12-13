"""
Script to generate JSON schemas for task arguments afresh, and write them
to the package manifest.

This script is a copy of [the original one in
fractal-tasks-core](https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/dev/update_manifest.py),
with minor changes.
"""
import json
import logging
from importlib import import_module
from pathlib import Path

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)
from fractal_tasks_core.dev.lib_task_docs import create_docs_info


PACKAGE = "custom_fractal_task_collection"


if __name__ == "__main__":

    # Read manifest
    imported_package = import_module(PACKAGE)
    manifest_path = (
        Path(imported_package.__file__).parent / "__FRACTAL_MANIFEST__.json"
    )
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    # Set global properties of manifest
    manifest["has_args_schemas"] = True
    manifest["args_schema_version"] = "pydantic_v1"

    # Loop over tasks
    task_list = manifest["task_list"]
    for ind, task in enumerate(task_list):
        executable = task["executable"]
        logging.info(f"[{executable}] START")

        # Create new JSON Schema for task arguments
        schema = create_schema_for_single_task(executable, package=PACKAGE)
        manifest["task_list"][ind]["args_schema"] = schema

        # Update docs_info, based on task-function description
        docs_info = create_docs_info(executable, package=PACKAGE)
        docs_link = ""
        if docs_info:
            manifest["task_list"][ind]["docs_info"] = docs_info
        if docs_link:
            manifest["task_list"][ind]["docs_link"] = docs_link

        logging.info(f"[{executable}] END (new schema/description/link)")
        print()

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    logging.info(f"Up-to-date manifest stored in {manifest_path.as_posix()}")
