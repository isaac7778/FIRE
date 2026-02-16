# Reinforcement Learning (Atari)

Source code for DQN Atari plasticity experiment (Fig. 4(a))

<p align="center">
  <img src="assets/dqn_results.png" width="600">
</p>

### 🎮 Game Demos: Vanilla (No Reset) vs FIRE

<table align="center">
  <tr>
    <th></th>
    <th align="center">Vanilla</th>
    <th align="center">FIRE (Ours)</th>
  </tr>
  <tr>
    <td><b>Asterix</b></td>
    <td align="center">
      <video src="assets/Asterix_Vanilla.mp4" width="280"></video>
    </td>
    <td align="center">
      <video src="assets/Asterix_FIRE.mp4" width="280"></video>
    </td>
  </tr>
  <tr>
    <td><b>BeamRider</b></td>
    <td align="center">
      <video src="assets/BeamRider_Vanilla.mp4" width="280"></video>
    </td>
    <td align="center">
      <video src="assets/BeamRider_FIRE.mp4" width="280"></video>
    </td>
  </tr>
  <tr>
    <td><b>DemonAttack</b></td>
    <td align="center">
      <video src="assets/DemonAttack_Vanilla.mp4" width="280"></video>
    </td>
    <td align="center">
      <video src="assets/DemonAttack_FIRE.mp4" width="280"></video>
    </td>
  </tr>
</table>

This project is built upon the [CleanRL](https://github.com/vwxyzjn/cleanrl) [1] codebase. It implements a two-stage DQN training pipeline for Atari games, designed to study various neural network plasticity and reset techniques applied at the midpoint of training.

> **Note:** Due to computational constraints, this study uses a vectorized DQN (`num_envs=4`) instead of the original DQN (`num_envs=1`). In our preliminary experiments, the performance difference between the two configurations was negligible.

## 📁 Code Structure

```
cleanrl-fire/
│
├── cleanrl/
│   ├── dqn_atari.py              # Stage 1: Train DQN to midpoint → save checkpoint
│   ├── dqn_atari_resume.py       # Stage 2: Resume from checkpoint with reset techniques
│   ├── buffers.py                # Replay buffer implementation
│   └── envs.py                   # Atari environment preprocessing wrappers
│
├── cleanrl_utils/
│   └── evals/
│       └── dqn_eval.py           # Evaluation utilities & human-normalized scoring
│
└── requirements/
    ├── requirements.txt           # Core dependencies
    └── requirements-atari.txt     # Atari-specific dependencies
```



## 📦 Installation

1. Create a virtual environment with Python 3.10:
```
conda create -n cleanrl-fire python=3.10 -y
conda activate cleanrl-fire
```

2. Install dependencies:
```
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-atari.txt
```

3. Install PyTorch matching your system (CUDA version, OS, etc.):

Visit https://pytorch.org/get-started/locally/ and follow the instructions for your environment.

## ▶️ How to Run

This project uses a **two-stage** training process.

### Stage 1: Initial Training

`dqn_atari.py` trains a DQN agent for the first half of the total training steps, then automatically saves a checkpoint and exits.

```
python -m cleanrl.dqn_atari --env-id {env_id} --seed {seed}
```

### Stage 2: Resume with Reset Techniques

`dqn_atari_resume.py` loads the saved checkpoint and continues training for the remaining half. At the start of resumed training, you can apply one of the following techniques (as studied in the referenced papers):

| Flag | Technique | Description |
|------|-----------|-------------|
| `--full_reset 1` | Reset [2] | Fully resets all network weights to their initial values |
| `--sr 1` | SR-DQN (Shrink & Perturb) [3] | Resets linear layers fully; interpolates conv layers (20% old + 80% init) |
| `--pi 1` | Plasticity Injection [4] | Freezes current weights and adds a new trainable residual branch, preserving outputs |
| `--fire 1` | FIRE (Ours) | Projects weights onto the closest orthogonal matrix (isotropy manifold) using Newton-Schulz iteration while resetting linear layers |
| *(none)* | Vanilla | Resumes training without any modification |

**Only one technique can be applied at a time.**

```
python -m cleanrl.dqn_atari_resume --env-id {env_id} --seed {seed} --full_reset {0,1} --sr {0,1} --pi {0,1} --fire {0,1}
```

### Example

```
# Stage 1
python -m cleanrl.dqn_atari --env-id BreakoutNoFrameskip-v4 --seed 1

# Stage 2 (with FIRE)
python -m cleanrl.dqn_atari_resume --env-id BreakoutNoFrameskip-v4 --seed 1 --fire 1
```

Run with `--help` to see full configs.
```
python -m cleanrl.dqn_atari --help
python -m cleanrl.dqn_atari_resume --help
```

## 📚 References

- [1] Huang, Shengyi, et al. "Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms." Journal of Machine Learning Research 23.274 (2022): 1-18.
- [2] Nikishin, Evgenii, et al. "The primacy bias in deep reinforcement learning." International conference on machine learning. PMLR, 2022.
- [3] D'Oro, Pierluca, et al. "Sample-efficient reinforcement learning by breaking the replay ratio barrier." Deep Reinforcement Learning Workshop NeurIPS 2022. 2022.
- [4] Nikishin, Evgenii, et al. "Deep reinforcement learning with plasticity injection." Advances in Neural Information Processing Systems 36 (2023): 37142-37159.
