import apx_fractal_task_collection
import json
from pathlib import Path


PACKAGE = "apx_fractal_task_collection"
PACKAGE_DIR = Path(apx_fractal_task_collection.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
    TASK_LIST = MANIFEST["task_list"]
