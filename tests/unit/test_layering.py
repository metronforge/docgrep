"""Guard the repository's source dependency boundaries."""

import ast
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "docgrep"
LAYERS = {"domain", "usecases", "adapters", "api"}


def _imported_modules(tree: ast.AST, package: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = parts[: len(parts) - node.level + 1]
                module = ".".join([*base, *(node.module or "").split(".")])
            else:
                module = node.module or ""
            modules.add(module.rstrip("."))
            if module == "docgrep":
                modules.update(f"docgrep.{alias.name}" for alias in node.names)
    return modules


def _violations(root: Path) -> list[str]:
    violations: list[str] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        layer = relative.parts[0]
        if layer not in LAYERS:
            continue
        package_parts = ["docgrep", *relative.parts[:-1]]
        package = ".".join(package_parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_modules(tree, package):
            parts = module.split(".")
            root_name = parts[0]
            target = parts[1] if root_name == "docgrep" and len(parts) > 1 else ""
            allowed = True
            if layer == "domain":
                allowed = (root_name in sys.stdlib_module_names) or (
                    root_name == "docgrep" and target == "domain"
                )
            elif layer == "usecases":
                allowed = (root_name in sys.stdlib_module_names) or (
                    root_name == "docgrep" and target in {"domain", "usecases"}
                )
            elif layer == "adapters":
                allowed = target not in {"usecases", "api"}
            if not allowed:
                violations.append(f"{relative}: {module}")
    return violations


def test_source_layering() -> None:
    assert _violations(SOURCE_ROOT) == []


def test_layering_guard_rejects_bad_import(tmp_path: Path) -> None:
    bad_module = tmp_path / "domain" / "bad.py"
    bad_module.parent.mkdir()
    bad_module.write_text("from docgrep.adapters import models\n", encoding="utf-8")

    assert _violations(tmp_path) == ["domain/bad.py: docgrep.adapters"]
