"""Юніт-тести DQN-агента qlearn.dqn.DQNAgent (етап 3). Потребують PyTorch (маркер torch).

Ключові формули:
    ціль        y = r + γ·(1 − done)·max_{a′ дозволені} Q_target(s′, a′)
    втрата      Huber(Q(s,a), y)
Неможливі дії (стіна, край) не беруть участі ні у виборі дії, ні в max для s′.
"""
import math
import random

import pytest

torch = pytest.importorskip("torch")

from qlearn.dqn import DQNAgent, masked_argmax          # noqa: E402
from qlearn.environment import GridEnv                  # noqa: E402
from test_agent_contract import AgentContract           # noqa: E402

pytestmark = pytest.mark.torch


def make(env_dict, epsilon=None, **params):
    agent = DQNAgent(GridEnv(env_dict), **params)
    if epsilon is not None:
        agent.epsilon = epsilon
    return agent


class TestDQNContract(AgentContract):
    @pytest.fixture
    def make_agent(self):
        def _make(env_dict, epsilon=None):
            random.seed(0)
            return make(env_dict, epsilon=epsilon)
        return _make


# ---------------------------------------------------------------- будова
def test_defaults_and_params(maps):
    random.seed(0)
    agent = make(maps["K1_default"])
    p = agent.params()
    assert agent.name == "dqn"
    assert (p["gamma"], p["epsilon"]) == (0.9, 0.12)
    for key in ("hidden", "lr", "batch", "buffer", "warmup", "target_sync", "features", "reward_scale"):
        assert key in p
    assert p["features"] == "compact"        # рішення Р3 (за вимірами збіжності етапу 3)


def test_unknown_param_rejected(maps):
    with pytest.raises(ValueError):
        make(maps["K1_default"], learning_rate=0.1)


def test_net_output_shape(maps):
    random.seed(0)
    agent = make(maps["K1_default"])
    out = agent.net(agent.X[:5])
    assert tuple(out.shape) == (5, 4)


@pytest.mark.parametrize("kind, size", [("onehot", 42), ("compact", 8)])
def test_feature_kinds(maps, kind, size):
    random.seed(0)
    agent = make(maps["K1_default"], features=kind)
    assert tuple(agent.X.shape) == (42, size)


# ------------------------------------------------------------ маскування
def test_masked_argmax_never_forbidden():
    q = torch.tensor([10.0, 1.0, 20.0, 2.0])
    mask = torch.tensor([False, True, False, True])
    assert masked_argmax(q, mask) == 3


def test_greedy_action_never_forbidden(maps):
    random.seed(1)
    agent = make(maps["K2_wall_gap"], epsilon=0.0)
    for i in agent.env.free:
        a = agent.choose_action(i)
        assert a in agent.env.nb[i]


# -------------------------------------------------------- ціль і втрата
def test_target_handcrafted(maps):
    """Стан 0 (кут (0,0) сітки 3×3): дозволені лише ВНИЗ(1) і ВПРАВО(3).
    Заглушка цільової мережі дає [100, 1, 50, 2] → max серед дозволених = 2.
    y = −0.1 + 0.9·2 = 1.7; для done=1 y = r = 0.49."""
    random.seed(0)
    agent = make(maps["S_3x3"])
    agent.target = lambda x: torch.tensor([[100.0, 1.0, 50.0, 2.0]]).repeat(x.shape[0], 1)
    y = agent.compute_targets(torch.tensor([-0.1, 0.49]), torch.tensor([0, 0]), torch.tensor([0.0, 1.0]))
    assert y.tolist() == pytest.approx([1.7, 0.49])


def test_truncation_not_terminal(maps):
    """Кінець епізоду через ліміт кроків — НЕ термінальний стан: у буфері done=False."""
    random.seed(2)
    agent = make(maps["K1_default"], epsilon=1.0)
    agent.env.step_cap = lambda: 3           # штучно короткий ліміт
    out = agent.run_episode()
    assert out["steps"] == 3 and out["goal"] is False
    assert all(t[4] is False for t in agent.buffer.items())


def test_goal_transition_is_terminal():
    env = {"cols": 3, "rows": 3, "obstacles": [], "rough": [], "start": (2, 1), "finish": (2, 2)}
    random.seed(3)
    agent = make(env, epsilon=1.0)
    for _ in range(30):
        agent.run_episode()
    goal_i = agent.env.finish_i
    for s, _slot, _r, s2, done in agent.buffer.items():
        assert done is (s2 == goal_i)


def test_reward_scaled_in_buffer(maps):
    random.seed(4)
    agent = make(maps["S_3x3"], epsilon=1.0, reward_scale=0.1)
    agent.learn_step(agent.env.start_i)
    _s, _a, r, _s2, _d = agent.buffer.items()[0]
    assert r == pytest.approx(-0.1)


def test_loss_decreases_on_fixed_batch(maps):
    random.seed(5)
    agent = make(maps["K1_default"], epsilon=1.0, target_sync=10**9)
    for _ in range(5):
        agent.run_episode()
    batch = agent.buffer.sample(32)
    first = agent.optimize(batch)
    for _ in range(50):
        last = agent.optimize(batch)
    assert last <= 0.5 * first


def test_target_sync_period(maps):
    random.seed(6)
    agent = make(maps["K1_default"], epsilon=1.0, target_sync=5, warmup=32, batch=32)

    def same():
        return all(torch.equal(a, b) for a, b in zip(agent.net.state_dict().values(),
                                                     agent.target.state_dict().values()))
    s = agent.env.start_i
    while agent.learn_steps < 5:
        res = agent.learn_step(s)
        s = agent.env.start_i if res["goal"] else res["to"]
    assert agent.learn_steps == 5 and same()
    while agent.learn_steps < 6:
        res = agent.learn_step(s)
        s = agent.env.start_i if res["goal"] else res["to"]
    assert not same()


# ------------------------------------------------------- знімок Q і стійкість
def test_q_snapshot_only_allowed_pairs(maps):
    random.seed(7)
    agent = make(maps["K2_wall_gap"])
    for _ in range(5):
        agent.run_episode()
    q = agent.q_snapshot()
    allowed = {key for i in agent.env.free for _to, key, _c, _l in agent.env.nb[i]}
    for key, v in enumerate(q):
        if key not in allowed:
            assert v == 0.0
        else:
            assert math.isfinite(v)


def test_q_snapshot_in_reward_units(maps):
    """Знімок Q — у тих самих одиницях, що й таблиця (поділено на reward_scale)."""
    random.seed(8)
    agent = make(maps["S_3x3"], reward_scale=0.1)
    i = agent.env.start_i
    with torch.no_grad():
        raw = agent.net(agent.X[i:i + 1])[0].tolist()
    key = agent.env.nb[i][0][1]
    assert agent.q_snapshot()[key] == pytest.approx(raw[key - 4 * i] / 0.1)


def test_learn_step_reports_target_and_loss(maps):
    random.seed(9)
    agent = make(maps["K1_default"], warmup=1, batch=1)
    res = agent.learn_step(agent.env.start_i)
    assert {"target", "loss"} <= set(res)
    assert math.isfinite(res["target"])


def test_same_seed_same_snapshot(maps):
    def run():
        random.seed(10)
        agent = make(maps["K1_default"])
        for _ in range(15):
            agent.run_episode()
        return agent.q_snapshot()
    assert run() == run()


def test_no_nan_after_200_episodes(maps):
    random.seed(11)
    agent = make(maps["K1_default"])
    for _ in range(200):
        agent.run_episode()
    assert all(math.isfinite(v) for v in agent.q_snapshot())


def test_learn_step_and_run_episode_equivalent(maps):
    """Покроковий режим (анімація) і швидкий цикл навчають мережу однаково (як і в таблиці)."""
    random.seed(12)
    fast = make(maps["K1_default"])
    for _ in range(8):
        fast.run_episode()
    random.seed(12)
    slow = make(maps["K1_default"])
    for _ in range(8):
        s, steps = slow.env.start_i, 0
        while True:
            res = slow.learn_step(s)
            steps += 1
            s = res["to"]
            if res["goal"] or steps >= slow.env.step_cap():
                break
    assert slow.q_snapshot() == fast.q_snapshot()
    assert slow.learn_steps == fast.learn_steps
