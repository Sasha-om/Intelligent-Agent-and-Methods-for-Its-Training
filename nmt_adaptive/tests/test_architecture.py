"""Архітектурні правила: ядро (core/, sim/) і config.py не знають про streamlit.

Навіщо: тоді ядро можна тестувати й запускати в симуляторі без браузера,
а інтерфейс можна замінити, не чіпаючи алгоритм.
"""
import ast
from pathlib import Path

import config

CORE_FOLDERS = ("core", "sim")


def _imported_modules(path: Path) -> set[str]:
    """Повертає назви модулів верхнього рівня, які імпортує файл (без запуску коду)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _core_files() -> list[Path]:
    """Усі .py-файли ядра разом із config.py."""
    files = [config.BASE_DIR / "config.py"]
    for folder in CORE_FOLDERS:
        files.extend(sorted((config.BASE_DIR / folder).glob("*.py")))
    return files


def test_core_never_imports_streamlit() -> None:
    """Жоден файл ядра не імпортує streamlit."""
    offenders = [p.name for p in _core_files() if "streamlit" in _imported_modules(p)]
    assert offenders == []


def test_core_files_are_found() -> None:
    """Тест вище справді щось перевіряє (а не порожній список файлів)."""
    assert len(_core_files()) >= 3
