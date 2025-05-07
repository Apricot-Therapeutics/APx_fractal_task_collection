"""Generate JSON schemas for tasks and write them to the Fractal manifest."""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "apx_fractal_task_collection"
    AUTHOR = "Adrian Tschan"
    docs_link = "https://github.com/Apricot-Therapeutics/APx_fractal_task_collection"
    create_manifest(package=PACKAGE,
                    authors=AUTHOR,
                    docs_link=docs_link,
                    custom_pydantic_models=[
                        ("apx_fractal_task_collection", "utils.py", "TextureFeatures"),
                        ("apx_fractal_task_collection", "utils.py", "BaSiCPyModelParams"),
                    ])