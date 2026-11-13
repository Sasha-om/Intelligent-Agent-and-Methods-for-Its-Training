"""Генерує tests/golden_env.json — відбиток графа середовища для кожної тестової карти.

Знято зі СТАРОГО QModel 2026-11-13 перед рефакторингом етапу 2. Тест test_environment.py
перевіряє, що qlearn.GridEnv будує точно такий самий граф. Перегенеровувати — лише за погодженням.
"""
import hashlib
import json
import sys

import helpers


def graph_fingerprint(model):
    """Хеш графа: для кожного стану — кортежі (сусід, ключ Q, вартість, напрям)."""
    text = repr((model.cols, model.rows, list(model.free), [tuple(n) for n in model.nb],
                 model.start_i, model.finish_i))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    trainer = helpers.load_trainer()
    out = {}
    for name, env in sorted(helpers.build_maps().items()):
        m = trainer.QModel()
        m.load_env(env)
        out[name] = {"graph_sha256": graph_fingerprint(m), "free": len(m.free),
                     "step_cap": m.step_cap(), "reachable": m.finish_reachable()}
    path = helpers.TESTS_DIR / "golden_env.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Записано: {path}")


if __name__ == "__main__":
    main()
