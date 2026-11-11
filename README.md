# Q-lerning and AI

Навчальний проєкт до науково-дослідницької роботи «Інтелектуальний агент та методи його навчання»:
тренажер Q-навчання на сітці з перешкодами та (плановий) режим з нейромережею (DQN).

## Склад

| Папка | Що там |
|---|---|
| `Platform for Interaction/` | `q_learning_trainer.py` — тренажер на tkinter; `Тренажер Q-навчання.html` — вебверсія |
| `Pytorch_test/` | експерименти з PyTorch (прототип DQN) |

## Запуск тренажера (Windows, PowerShell)

```powershell
python "Platform for Interaction\q_learning_trainer.py"
```

Табличний режим потребує лише стандартного Python 3 з tkinter.
Режим нейромережі (після реалізації) потребує PyTorch.

## Примітка

Службові Markdown-файли та налаштування Claude Code (`AGENTS.md`, `CLAUDE.md`, `docs/`, `.claude/`)
зберігаються лише локально й свідомо виключені з git (див. `.gitignore`).
