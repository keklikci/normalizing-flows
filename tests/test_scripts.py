import ast
from pathlib import Path


def test_every_archived_notebook_has_a_script() -> None:
    notebooks = sorted(Path("notebooks").rglob("*.ipynb"))
    scripts = {
        path.relative_to("scripts").with_suffix(".ipynb")
        for path in Path("scripts").rglob("*.py")
    }
    assert notebooks
    assert {path.relative_to("notebooks") for path in notebooks} == scripts


def test_every_script_compiles() -> None:
    for script in Path("scripts").rglob("*.py"):
        ast.parse(script.read_text(), filename=str(script))
