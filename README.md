# RL Bench Pro — Deterministic & Policy-Gradient Agents (Gymnasium + SB3)

Portfolio-ready RL suite with unified training/eval CLIs, video recording, and metrics logs.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .[dev]
pre-commit install
```
### Train
```bash
python scripts/train.py --algo dqn --env CartPole-v1 --timesteps 200000 --log runs/cartpole --video cartpole_videos
```
### Evaluate
```bash
python scripts/eval.py --model runs/cartpole/model.zip --env CartPole-v1 --episodes 50 --video eval_videos/cartpole
```

MIT © 2025 Dhamodar Burla
