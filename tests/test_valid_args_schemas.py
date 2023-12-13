import json
import sys
import typing
from pathlib import Path
from typing import Optional
from typing import Union

import pytest
from devtools import debug
from jsonschema.validators import Draft201909Validator
from jsonschema.validators import Draft202012Validator
from jsonschema.validators import Draft7Validator

from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)
from fractal_tasks_core.dev.lib_signature_constraints import _extract_function
from fractal_tasks_core.dev.lib_signature_constraints import (
    _validate_function_signature,
)

from . import MANIFEST
from . import TASK_LIST



def test_task_functions_have_valid_signatures():
    """
    Test that task functions have valid signatures.
    """
    for ind_task, task in enumerate(TASK_LIST):
        function_name = Path(task["executable"]).with_suffix("").name
        task_function = _extract_function(
                task["executable"], function_name, package_name="custom_fractal_task_collection"
                )
        _validate_function_signature(task_function)


def test_args_schemas_are_up_to_date():
    """
    Test that args_schema attributes in the manifest are up-to-date
    """
    for ind_task, task in enumerate(TASK_LIST):
        print(f"Now handling {task['executable']}")
        old_schema = TASK_LIST[ind_task]["args_schema"]
        new_schema = create_schema_for_single_task(
                task["executable"], package="custom_fractal_task_collection"
                )
        # The following step is required because some arguments may have a
        # default which has a non-JSON type (e.g. a tuple), which we need to
        # convert to JSON type (i.e. an array) before comparison.
        new_schema = json.loads(json.dumps(new_schema))
        assert new_schema == old_schema


@pytest.mark.parametrize(
    "jsonschema_validator",
    [Draft7Validator, Draft201909Validator, Draft202012Validator],
)
def test_args_schema_comply_with_jsonschema_specs(jsonschema_validator):
    """
    This test is actually useful, see
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/564.
    """
    for ind_task, task in enumerate(TASK_LIST):
        schema = TASK_LIST[ind_task]["args_schema"]
        my_validator = jsonschema_validator(schema=schema)
        my_validator.check_schema(my_validator.schema)
        print(
            f"Schema for task {task['executable']} is valid for "
            f"{jsonschema_validator}."
        )
