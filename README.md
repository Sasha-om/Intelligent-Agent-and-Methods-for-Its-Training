# Q-lerning and AI

Навчальний проєкт до науково-дослідницької роботи МАН «Інтелектуальний агент та методи його навчання»:
тренажер навчання агента на сітці-графі двома методами — **табличне Q-навчання** і **глибоке Q-навчання (DQN)** —
із записом і порівнянням результатів.

## Склад

| Папка / файл | Що там |
|---|---|
| `Platform for Interaction/q_learning_trainer.py` | тренажер (tkinter): анімація навчання, редактор середовища, таблиця Q, перемикач «Таблиця / Нейромережа», збереження й порівняння запусків |
| `Platform for Interaction/qlearn/` | логіка без інтерфейсу: середовище `GridEnv`, агенти `TabularAgent` і `DQNAgent`, запис метрик, порівняння, пакетний запуск |
| `Platform for Interaction/tests/` | тести pytest: юніт, smoke (зокрема справжнє вікно), регресія |
| `Platform for Interaction/results/` | збережені запуски (JSON) і серії експериментів `exp-<дата>/` з CSV для Excel |
| `Platform for Interaction/maps/` | карти для експериментів (формат експорту тренажера) |
| `Platform for Interaction/Тренажер Q-навчання.html` | вебверсія (лише табличний режим) |
| `Pytorch_test/` | віртуальне середовище з PyTorch і перші досліди |

## Запуск тренажера (Windows, PowerShell)

```powershell
cd "Platform for Interaction"
python q_learning_trainer.py                                   # лише таблиця (стандартний Python)
& "..\Pytorch_test\.venv\Scripts\python.exe" q_learning_trainer.py   # таблиця + нейромережа (PyTorch)
```

У вікні: «Таблиця / Нейромережа» — вибір методу; «Старт», «Крок», «Швидке навчання»; поле «Seed» робить запуск
відтворюваним; «Зберегти результат…» пише запуск у `results/`; «Порівняти результати…» відкриває вікно порівняння
(таблиця метрик, криві навчання, групування за seed, експорт CSV).

## Пакетні експерименти (без вікна)

```powershell
cd "Platform for Interaction"
$py = "..\Pytorch_test\.venv\Scripts\python.exe"
& $py -m qlearn.runner --agent both --episodes 600 --seeds 0 1 2 3 4 --map default --out results\exp-demo\K1
& $py -m qlearn.runner --help        # усі параметри: --map wall_gap|rough|файл.json, --gamma, --epsilon, --set hidden=32 …
```

Кожен запуск — JSON у папці `--out`, плюс `comparison-*.csv` (усі запуски) і `summary-*.csv` (середнє ± σ по seed).

## Тести

```powershell
cd "Platform for Interaction"
& "..\Pytorch_test\.venv\Scripts\python.exe" -m pytest            # швидкі
& "..\Pytorch_test\.venv\Scripts\python.exe" -m pytest --runslow  # усі, включно з повільними регресіями
```

## Примітка

Службові Markdown-файли та налаштування Claude Code (`AGENTS.md`, `CLAUDE.md`, `Insight.md`, `Handoff.md`, `docs/`, `.claude/`)
зберігаються лише локально й свідомо виключені з git (див. `.gitignore`). Єдиний відстежуваний Markdown — цей `README.md`.
