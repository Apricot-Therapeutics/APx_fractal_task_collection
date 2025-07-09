# docs/gen_ref_pages.py
"""Generate the code reference pages."""

from pathlib import Path
import mkdocs_gen_files
import os

PACKAGE = "apx_fractal_task_collection"
SOURCE_ROOT = Path(f"src/{PACKAGE}")

def write_combined_subpackage_page(subfolder: str):
    nav = mkdocs_gen_files.Nav()
    folder_path = SOURCE_ROOT / subfolder

    for py_file in sorted(folder_path.glob("*.py")):
        doc_path = Path("..", subfolder, f"{py_file.stem}")
        write_path = Path("reference", subfolder, f"{py_file.stem}.md")
        if py_file.name == "__init__.py":
            continue  # skip empty/undocumented __init__.py
        else:
            with mkdocs_gen_files.open(write_path, "w") as f:
                mod_name = py_file.stem
                full_mod = f"{PACKAGE}.{subfolder}.{mod_name}"
                f.write(f"## `{full_mod}`\n\n")
                f.write(f"::: {full_mod}\n\n")

        nav[py_file.stem] = doc_path

    # Write the nav summary (for literate-nav)
    with mkdocs_gen_files.open(f"reference/{subfolder}_summary.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())

# Write init_utils as a standalone page
def write_standalone_page(module_name: str):
    doc_path = Path("reference", f"{module_name}.md")
    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(f"# `{PACKAGE}.{module_name}`\n\n")
        f.write(f"::: {PACKAGE}.{module_name}\n\n")

standalone_pages = ['init_utils', 'io_models', 'utils']
for module in standalone_pages:
    write_standalone_page(module)

# Generate tasks.md and features.md
write_combined_subpackage_page("tasks")
write_combined_subpackage_page("features")
