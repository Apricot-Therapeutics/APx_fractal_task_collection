import json
import requests
from jsonschema import validate
from devtools import debug

from . import MANIFEST


def test_valid_manifest(tmp_path):
    """
    NOTE: to avoid adding a fractal-server dependency, we simply download the
    relevant file.
    """
    # Download JSON Schema for ManifestV1
    url = (
        "https://raw.githubusercontent.com/fractal-analytics-platform/"
        "fractal-server/main/"
        "fractal_server/app/schemas/json_schemas/manifest.json"
    )
    r = requests.get(url)
    with (tmp_path / "manifest_schema.json").open("wb") as f:
        f.write(r.content)
    with (tmp_path / "manifest_schema.json").open("r") as f:
        manifest_schema = json.load(f)

    debug(MANIFEST)
    validate(instance=MANIFEST, schema=manifest_schema)
