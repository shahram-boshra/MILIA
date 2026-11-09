"""Every third-party package that first-party code imports unconditionally is a declared dependency.

A package that is only installed as a dependency of a dependency can disappear when that dependency
changes (deptry rule DEP003, "transitive dependency"). Imports inside a ``try`` that handles
``ImportError`` (optional integrations) and under ``if TYPE_CHECKING:`` are exempt; the extras count as
declared, since code that needs an extra guards on it or is only reached with it installed.

The declared requirements are read from the installed ``milia-py`` metadata (``Requires-Dist``, built
from ``pyproject.toml``), and an import is mapped to its distribution through the installed packages; a
module that no installed distribution provides cannot be mapped and is not judged here (the test image
installs every extra).
"""

from __future__ import annotations

import ast
import sys
from importlib.metadata import packages_distributions, requires
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "milia_pipeline"
OPTIONAL_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _declared_distributions() -> set[str]:
    """Every requirement of milia-py, core and extras (markers ignored)."""
    return {canonicalize_name(Requirement(r).name) for r in requires("milia-py") or []}


def _handles_import_errors(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:
            return True
        types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        if any(isinstance(t, ast.Name) and t.id in OPTIONAL_IMPORT_ERRORS for t in types):
            return True
    return False


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _unconditional_imports(tree: ast.AST) -> set[str]:
    """Top-level module names imported outside optional-import guards."""
    found: set[str] = set()

    def visit(node: ast.AST, guarded: bool) -> None:
        if isinstance(node, ast.Import) and not guarded:
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not guarded and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
        for field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if not isinstance(child, ast.AST):
                    continue
                child_guarded = guarded
                if isinstance(node, ast.Try) and field == "body" and _handles_import_errors(node):
                    child_guarded = True
                if isinstance(node, ast.If) and field == "body" and _is_type_checking(node):
                    child_guarded = True
                visit(child, child_guarded)

    visit(tree, False)
    return found


def test_unconditional_third_party_imports_are_declared():
    declared = _declared_distributions()
    distributions = packages_distributions()
    undeclared: dict[str, set[str]] = {}

    for path in sorted(PACKAGE.rglob("*.py")):
        for module in _unconditional_imports(ast.parse(path.read_text(), filename=str(path))):
            if module in sys.stdlib_module_names or module in {"milia_pipeline", "__future__"}:
                continue
            providers = {canonicalize_name(d) for d in distributions.get(module, [])}
            if providers and not providers & declared:
                undeclared.setdefault(module, set()).add(str(path.relative_to(REPO)))

    assert undeclared == {}, (
        "Imported unconditionally but not declared in pyproject.toml (core or extra): "
        + "; ".join(f"{m} ({', '.join(sorted(files))})" for m, files in sorted(undeclared.items()))
    )


def test_the_guard_recognises_optional_imports():
    tree = ast.parse(
        "import needed\n"
        "try:\n    import optional\nexcept ImportError:\n    optional = None\n"
        "if TYPE_CHECKING:\n    import typing_only\n"
        "def f():\n    from local_need import x\n"
    )
    assert _unconditional_imports(tree) == {"needed", "local_need"}
