"""Ознаки стану (вхід нейромережі) і маски дозволених дій — без torch і без NumPy.

Два варіанти входу (рішення Р3 у docs/PLAN.md):
    onehot  — вектор довжини cols*rows з однією одиницею на місці клітинки.
              Мережа «бачить» кожну клітинку окремо — найближче до таблиці, найнадійніше.
    compact — 8 чисел: x, y клітинки; dx, dy до фінішу; прохідність 4 сусідів (1 — звичайна,
              1/3 — ділянка 3×, 0 — стіна чи край). Розмір не залежить від сітки, мережа може узагальнювати.
"""
from .environment import DIRS, ROUGH_COST

KINDS = ("onehot", "compact")
ONEHOT_MAX_CELLS = 900   # 30×30: більші сітки дають надто велику матрицю ознак (100×100 → ≈400 МБ)


def feature_size(env, kind):
    if kind == "onehot":
        n = env.cols * env.rows
        if n > ONEHOT_MAX_CELLS:
            raise ValueError(f"вхід onehot підходить для сіток до {ONEHOT_MAX_CELLS} клітинок, "
                             f"а тут {env.cols}×{env.rows}; оберіть compact")
        return n
    if kind == "compact":
        return 8
    raise ValueError(f"невідомий вид ознак: {kind!r} (можливі: {', '.join(KINDS)})")


def onehot(env, i):
    v = [0.0] * (env.cols * env.rows)
    v[i] = 1.0
    return v


def compact(env, i):
    c, r = env.cell_of(i)
    fc, fr = env.finish
    w, h = max(env.cols - 1, 1), max(env.rows - 1, 1)
    v = [c / w, r / h, (fc - c) / w, (fr - r) / h]
    for dc, dr, _label in DIRS:
        nc, nr = c + dc, r + dr
        if not env.in_bounds(nc, nr) or (nc, nr) in env.obstacles:
            v.append(0.0)
        elif (nc, nr) in env.rough:
            v.append(1.0 / ROUGH_COST)
        else:
            v.append(1.0)
    return v


def features(env, i, kind):
    feature_size(env, kind)               # перевірка назви
    return onehot(env, i) if kind == "onehot" else compact(env, i)


def all_features(env, kind):
    """Ознаки для всіх клітинок (рядок i — стан i; у перешкод — нулі, вони не використовуються)."""
    size = feature_size(env, kind)
    free = set(env.free)
    return [features(env, i, kind) if i in free else [0.0] * size for i in range(env.cols * env.rows)]


def action_mask(env, i):
    """[True/False] × 4 за порядком DIRS: чи дозволена дія в стані i."""
    slots = {key - i * 4 for _to, key, _cost, _label in env.nb[i]}
    return [s in slots for s in range(4)]
