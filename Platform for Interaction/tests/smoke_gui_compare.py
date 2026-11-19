"""GUI smoke-тест вікна «Порівняння результатів» (етап 8): справжнє вікно tkinter (приховане).

Маркер gui: без tkinter/дисплея — пропуск. У sandbox — під Xvfb (INS-014).
Дані — фікстури tests/fixture_run_tabular.json і tests/fixture_run_dqn.json (tests/make_fixture_runs.py),
скопійовані в тимчасову папку, тож справжня results/ не чіпається.
"""
import json
import shutil

import pytest

import helpers

pytestmark = pytest.mark.gui
FIXTURES = helpers.TRAINER_PATH.parent / "tests"


@pytest.fixture
def root():
    import tkinter as tk
    r = tk.Tk()
    r.withdraw()
    r.errors = []
    r.report_callback_exception = lambda exc, val, tb: r.errors.append(val)
    yield r
    r.destroy()


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    for name in ("fixture_run_tabular.json", "fixture_run_dqn.json"):
        shutil.copy(FIXTURES / name, d / name)
    return d


def test_compare_window_flow(trainer, root, folder, tmp_path):
    (folder / "broken.json").write_text("{ не json", encoding="utf-8")
    win = trainer.CompareWindow(root, folder=folder)
    root.update()
    assert len(win.runs) == 2                                    # пошкоджений файл пропущено…
    assert "broken.json" in win.status.cget("text")              # …і про нього сказано
    win.select_all()
    root.update()
    assert len(win.table_rows) == 2 and win.banner.cget("text") == ""
    assert len(win.chart.find_withtag("series")) == 2
    assert len(win.chart.find_withtag("legend")) >= 2

    for metric in ("steps", "success", "optimal", "reward"):
        win.set_metric(metric)
        assert win.metric == metric and win.metric_btns[metric].active
        assert len(win.chart.find_withtag("series")) == 2
    win.smooth_var.set(10)
    win.redraw()
    win.group_var.set(True)                                      # різні агенти — дві групи по одному запуску
    win.redraw()
    assert len(win.chart.find_withtag("series")) == 2

    out = win.export_csv(tmp_path / "порівняння.csv")
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                       # BOM для Excel
    text = raw.decode("utf-8-sig")
    assert text.splitlines()[0].startswith("run_id;agent;seed") and len(text.splitlines()) == 3

    win.close()
    root.update()
    assert not win.winfo_exists()
    assert root.errors == []


def test_add_file_other_map_shows_banner(trainer, root, folder, tmp_path):
    data = json.loads((folder / "fixture_run_tabular.json").read_text(encoding="utf-8"))
    data["run_id"] = "other_map"
    data["env"]["fingerprint"] = "інша-карта"
    data["env"]["map"]["cols"] = 10
    other = tmp_path / "other.json"
    other.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    win = trainer.CompareWindow(root, folder=folder)
    win.add_file(other)
    assert len(win.runs) == 3
    win.select_all()
    root.update()
    assert "різних картах" in win.banner.cget("text")
    win.refresh()                                                # «Оновити» не губить доданий файл
    assert len(win.runs) == 3
    win.close()
    assert root.errors == []


def test_group_by_seed_draws_band(trainer, root, folder):
    data = json.loads((folder / "fixture_run_tabular.json").read_text(encoding="utf-8"))
    data["run_id"], data["seed"] = "fixture_tabular_seed2", 2
    data["metrics"]["reward_per_episode"] = [r + 5 for r in data["metrics"]["reward_per_episode"]]
    (folder / "seed2.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    win = trainer.CompareWindow(root, folder=folder)
    win.select_all()
    win.group_var.set(True)
    win.redraw()
    root.update()
    assert len(win.chart.find_withtag("series")) == 2            # таблиця (2 seed) + DQN
    assert len(win.chart.find_withtag("band")) == 1              # смуга ±σ лише там, де ≥ 2 запуски
    win.close()
    assert root.errors == []


def test_empty_folder(trainer, root, tmp_path):
    win = trainer.CompareWindow(root, folder=tmp_path / "немає")
    root.update()
    assert win.runs == [] and "немає збережених запусків" in win.status.cget("text").lower()
    win.select_all()
    win.redraw()
    assert win.chart.find_withtag("series") == ()
    win.close()
    assert root.errors == []


def test_open_from_main_window(trainer, root, folder):
    app = trainer.App(root)
    app.results_dir = folder
    app.open_compare()
    first = app.compare_win
    assert first.winfo_exists() and len(first.runs) == 2
    app.open_compare()                                           # друге натискання — те саме вікно
    assert app.compare_win is first
    first.close()
    app.open_compare()
    assert app.compare_win is not first
    app.compare_win.close()
    app._cancel()
    assert root.errors == []


def test_group_mode_table_colors_match_lines(trainer, root, folder):
    """У режимі «Групувати за seed» квадратики в таблиці — кольору лінії своєї групи."""
    data = json.loads((folder / "fixture_run_tabular.json").read_text(encoding="utf-8"))
    data["run_id"], data["seed"] = "fixture_tabular_seed2", 2
    (folder / "seed2.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    win = trainer.CompareWindow(root, folder=folder)
    win.select_all()
    ids = [r["run_id"] for r in win.selected_runs()]
    assert len({win.colors[i] for i in ids}) == 3
    win.group_var.set(True)
    win.redraw()
    tab = [i for i in ids if "tabular" in i]
    assert win.colors[tab[0]] == win.colors[tab[1]]
    win.close()
