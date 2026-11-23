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


def test_streamlit_only_in_app_pages() -> None:
    """Streamlit імпортують лише сторінки app_*.py; сховище й логіка сесії (app/storage.py, app/session.py) — ні."""
    app_files = sorted((config.BASE_DIR / "app").glob("*.py"))
    with_streamlit = {p.name for p in app_files if "streamlit" in _imported_modules(p)}
    assert with_streamlit == {"app_student.py", "app_teacher.py"}


def test_tests_do_not_touch_real_student_db() -> None:
    """Тести не відкривають справжню базу data/students/ (лише tmp_path)."""
    offenders = [p.name for p in (config.BASE_DIR / "tests").glob("test_*.py")
                 if p.name != "test_architecture.py" and "storage.connect()" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_streamlit_privacy_settings() -> None:
    """.streamlit/config.toml: без статистики в інтернет і лише localhost."""
    import tomllib
    settings = tomllib.loads((config.BASE_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert settings["browser"]["gatherUsageStats"] is False and settings["server"]["address"] == "localhost"
    assert settings["server"]["showEmailPrompt"] is False


def test_launchers_for_simple_start() -> None:
    """Подвійний клік: start_test / start_teacher / start_class → launch.bat (venv, requirements, streamlit)."""
    base = config.BASE_DIR
    launch = (base / "launch.bat").read_bytes().decode("utf-8")
    assert "requirements.txt" in launch and "\r\n" in launch
    assert "app\\app_student.py" in launch and "app\\app_teacher.py" in launch
    for name, mode in (("start_test.bat", "student"), ("start_teacher.bat", "teacher"), ("start_class.bat", "class")):
        assert f"launch.bat\" {mode}" in (base / name).read_bytes().decode("utf-8")


def test_launcher_ports_match_config() -> None:
    """Порти в launch.bat збігаються з config; у режимі класу тест відкритий у мережу, сторінка вчителя — ні."""
    launch = (config.BASE_DIR / "launch.bat").read_bytes().decode("utf-8")
    class_part = launch.split(":class")[-1]
    assert f"--server.address 0.0.0.0 --server.port {config.CLASS_PORT}" in class_part
    teacher_line = next(line for line in class_part.splitlines() if "app_teacher.py" in line)
    assert f"--server.port {config.TEACHER_PORT}" in teacher_line and "0.0.0.0" not in teacher_line


def test_pilot_results_not_in_git() -> None:
    """results_pilot/ (похідні від даних учнів) — у кореневому .gitignore; модулі пілоту — без streamlit."""
    gitignore = (config.BASE_DIR.parent / ".gitignore").read_bytes().decode("utf-8-sig")
    assert "nmt_adaptive/results_pilot/" in gitignore and config.PILOT_RESULTS_DIR.name == "results_pilot"
    for name in ("pilot.py", "pilot_storage.py", "pilot_report.py", "strategies.py"):
        assert "streamlit" not in _imported_modules(config.BASE_DIR / "app" / name)


def test_sim_does_not_import_app() -> None:
    """Шари: sim/ (симуляція) не залежить від app/ (застосунку); спільна статистика — у core/stats.py."""
    offenders = [p.name for p in (config.BASE_DIR / "sim").glob("*.py")
                 if any(m == "app" or m.startswith("app.") for m in _imported_modules(p))]
    assert offenders == []
