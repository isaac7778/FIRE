import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    MaxAndSkipEnv,
    StickyActionEnv
)
from cleanrl.buffers import ReplayBuffer



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "Plasticity"  # None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 2500001
    """total timesteps of the experiments"""
    learning_rate: float = 0.0000625
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 2000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 20000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""

    eval_frequency: int = 250000
    """the frequency of evaluation. Set to 0 to disable"""


def make_env(env_id, seed, idx, capture_video, run_name):
    if 'NoFrameskip-v4' not in env_id:
        raise ValueError("only support NoFrameskip-v4 environments")

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(
                env_id,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = MaxAndSkipEnv(env, skip=4)
        env = ClipRewardEnv(env)
        env = StickyActionEnv(env, 0.25)

        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=100000)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, env.single_action_space.n)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()

        self.initialize_weights()

    def initialize_weights(self):  # since redo use xavier_uniform, we follow the same
        """Xavier uniform initialization for all conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, include_encoder_output=False):

        preact1 = self.conv1(x / 255.0)
        x = self.act1(preact1)

        preact2 = self.conv2(x)
        x = self.act2(preact2)

        preact3 = self.conv3(x)
        enc_out = self.act3(preact3)

        x = enc_out.view(x.size(0), -1)

        preact4 = self.fc1(x)
        feat = self.act4(preact4)

        # Output
        q = self.fc2(feat)

        if include_encoder_output:
            return enc_out, q
        else:
            return q


def linear_schedule(start_e: float, end_e: float, duration: int, t: int, bonus: int = 0):
    """
    Computes a linearly decayed value with a warmup period.

    Args:
        start_e: float, the starting value (e.g., epsilon starts at 1.0).
        end_e: float, the final value after decay (e.g., epsilon ends at 0.01).
        duration: int, the number of steps over which to decay the value.
        t: int, the current step.
        bonus: int, the number of steps to maintain the start_e value (warmup period).

    Returns:
        float, the current value adjusted by the linear schedule.
    """
    if t < bonus:
        # During the warmup period, return the starting value
        return start_e
    else:
        # After the warmup period, apply linear decay
        slope = (end_e - start_e) / duration
        value = slope * (t - bonus) + start_e
        # Ensure the value does not go below end_e
        return max(value, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    """
    # Note: The original DQN is typically trained with num_envs = 1. 
    # However, in this study we use num_envs = 4 (i.e., a vectorized DQN) due to limited training time. 
    # Accordingly, we adjust train_frequency to keep the effective update rate comparable.

    # For example, with train_frequency = 4 (RR = 0.25, the default DQN setting), 
    # when num_envs = 1 (original DQN setting) we perform one update every 4 environment steps. 
    # In contrast, when num_envs = 4, a single step yields 4 transitions (s, a, r, s'), so we perform one update per environment step.

    """
    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    assert args.target_network_frequency % args.num_envs == 0, "target network frequency should be divisible by the number of envs"
    assert args.train_frequency % args.num_envs == 0 or args.num_envs % args.train_frequency == 0
    if args.train_frequency % args.num_envs == 0:
        repeat_update = 1
    elif args.num_envs % args.train_frequency == 0:
        repeat_update = args.num_envs // args.train_frequency

    # env_id example: "BreakoutNoFrameskip-v4" -> game: "Breakout"
    game = args.env_id.replace("NoFrameskip-v4", "")
    run_name = f"{game}_{args.seed}"

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs)
    target_network = QNetwork(envs)

    q_network.to(device)
    target_network.to(device)

    target_network.load_state_dict(q_network.state_dict())

    # for reset
    from copy import deepcopy
    init_q_network = deepcopy(q_network)
    init_q_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(
        q_network.parameters(),
        lr=args.learning_rate,
        eps=1.5e-4,
    )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(0, args.total_timesteps, args.num_envs):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
            bonus=args.learning_starts
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_network.eval()
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            q_network.train()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    if args.track:
                        import wandb
                        wandb.log({
                            "episodic_return": info["episode"]["r"],
                            "episodic_length": info["episode"]["l"],
                            "global_step": global_step,
                            "epsilon": epsilon,
                        })

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                for n_update in range(repeat_update):
                    data = rb.sample(args.batch_size)

                    observations = data.observations
                    next_observations = data.next_observations

                    with torch.no_grad():
                        target_max, _ = target_network(next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    enc_out, qs = q_network(observations, include_encoder_output=True)

                    old_val = qs.gather(1, data.actions).squeeze()
                    loss = F.smooth_l1_loss(td_target, old_val)

                    if global_step % 160 == 0 and n_update == 0 and args.track:
                        import wandb
                        wandb.log({
                            "losses/td_loss": loss,
                            "losses/q_values": old_val.mean().item(),
                            "charts/SPS": int(global_step / (time.time() - start_time)),
                            "global_step": global_step,
                        })

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        if global_step % args.eval_frequency == 0:
            import wandb
            from cleanrl_utils.evals.dqn_eval import evaluate_steps

            model_path = f"runs/{run_name}/{global_step}"
            torch.save(q_network.state_dict(), model_path)
            episodic_returns = evaluate_steps(
                model_path,
                make_env,
                args.env_id,
                eval_steps=125000,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                device=device,
                epsilon=0.001,
                num_envs=args.num_envs,
                capture_video=False
            )
            mean_return = np.mean(episodic_returns).astype(float) if len(episodic_returns) > 0 else 0
            max_return = np.max(episodic_returns).astype(float) if len(episodic_returns) > 0 else 0
            min_return = np.min(episodic_returns).astype(float) if len(episodic_returns) > 0 else 0

            wandb.log({
                "eval/mean_episodic_return": mean_return,
                "eval/max_episodic_return": max_return,
                "eval/min_episodic_return": min_return,
                "global_step": global_step,
            })

            for idx, episodic_return in enumerate(episodic_returns):
                wandb.log({
                    "eval/episodic_return": episodic_return
                })

            # -------------------------------------------------
            # at the midpoint (global_step == total_timesteps // 2),
            # save the model + optimizer + replay buffer checkpoint after evaluation
            # After the training stops, we can continue training with the checkpoint by dqn_atari_resume.py
            # -------------------------------------------------
            midpoint_step = args.total_timesteps // 2
            if global_step == midpoint_step:
                checkpoint = {
                    "q_network": q_network.state_dict(),
                    "target_network": target_network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "init_q_network": init_q_network.state_dict(),
                    "replay_buffer": rb,
                    "global_step": global_step,
                }
                ckpt_path = f"runs/{run_name}/midpoint_ckpt.pt"
                # save the replay buffer together, so that it can be serialized even if it is larger than 4GiB
                # Be patient here, because it takes a long time to save the huge replay buffer.
                # According to our experience, this file is about 26GB.
                torch.save(checkpoint, ckpt_path, pickle_protocol=4)
                print(f"Saved midpoint checkpoint to {ckpt_path}")

                # save the midpoint checkpoint and exit immediately
                envs.close()
                if args.track:
                    wandb.finish()
                import sys
                sys.exit(0)



