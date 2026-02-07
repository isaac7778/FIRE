import os
import random
import time
from dataclasses import dataclass
import copy
import wandb

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

    # ckpt path
    resume_checkpoint_path: str = ""
    """Path to checkpoint (.pt) saved by dqn_atari_revision.py"""

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

    # options to apply only once at the start of resume training (0 or 1)
    fire: int = 0
    full_reset: int = 0
    sr: int = 0  # SR-DQN (S&P) to apply only once at the start of resume training
    pi: int = 0  # Plasticity Injection


def make_env(env_id, seed, idx, capture_video, run_name):
    if 'NoFrameskip-v4' not in env_id:
        raise ValueError("only support NoFrameskip-v4 environments")

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
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

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, include_encoder_output: bool = False):
        preact1 = self.conv1(x / 255.0)
        x = self.act1(preact1)

        preact2 = self.conv2(x)
        x = self.act2(preact2)

        preact3 = self.conv3(x)
        enc_out = self.act3(preact3)

        x = enc_out.view(x.size(0), -1)

        preact4 = self.fc1(x)
        feat = self.act4(preact4)

        q = self.fc2(feat)

        if include_encoder_output:
            return enc_out, q
        else:
            return q

class PlasticityInjectedQNetwork(nn.Module):
    def __init__(self, env, old_q_network=None, init_q_network=None, device=None):
        super().__init__()
        self.device = device

        # Helper to copy head
        def get_head_layers(network):
            head = nn.ModuleDict()
            head['fc1'] = network.fc1
            head['act4'] = network.act4
            head['fc2'] = network.fc2
            return head

        # Helper to make head struct
        def make_head_struct(env):
            head = nn.ModuleDict()
            head['fc1'] = nn.Linear(3136, 512)
            head['act4'] = nn.ReLU()
            head['fc2'] = nn.Linear(512, env.single_action_space.n)

            return head

        # Shared / Copied from old trained network
        if old_q_network is not None:
            self.conv1 = old_q_network.conv1
            self.conv2 = old_q_network.conv2
            self.conv3 = old_q_network.conv3
            self.act1 = old_q_network.act1
            self.act2 = old_q_network.act2
            self.act3 = old_q_network.act3
        else:
            # Fresh init for structural compatibility
            self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()

        # 2. Head Setup
        if old_q_network is not None:
            # (1). Frozen Old Head
            self.old_head = get_head_layers(old_q_network)
            # (2). Trainable New Head (from init)
            self.new_train_head = get_head_layers(copy.deepcopy(init_q_network))
            # (3). Frozen New Head (from init)
            self.new_frozen_head = get_head_layers(copy.deepcopy(init_q_network)) 
        else:
            # Structural Init
            self.old_head = make_head_struct(env)
            self.new_train_head = make_head_struct(env)
            self.new_frozen_head = make_head_struct(env)

        # Freeze
        for p in self.old_head.parameters(): p.requires_grad = False
        for p in self.new_frozen_head.parameters(): p.requires_grad = False

    def forward_encoder(self, enc, x):
        preact1 = enc['conv1'](x / 255.0)
        x = enc['act1'](preact1)

        preact2 = enc['conv2'](x)
        x = enc['act2'](preact2)

        preact3 = enc['conv3'](x)
        enc_out = enc['act3'](preact3)
        
        return enc_out

    def forward_head(self, head, x):
        preact4 = head['fc1'](x)
        x_feat = head['act4'](preact4)
        q = head['fc2'](x_feat)
        return q

    def forward(self, x):
        # Partial PI: Shared Encoder
        preact1 = self.conv1(x / 255.0)
        x = self.act1(preact1)

        preact2 = self.conv2(x)
        x = self.act2(preact2)

        preact3 = self.conv3(x)
        enc_out = self.act3(preact3)

        x_flat = enc_out.view(x.size(0), -1)

        # Heads
        q_old = self.forward_head(self.old_head, x_flat)
        q_new_train = self.forward_head(self.new_train_head, x_flat)
        q_new_frozen = self.forward_head(self.new_frozen_head, x_flat)

        q = q_old + q_new_train - q_new_frozen
        
        return q

def linear_schedule(start_e: float, end_e: float, duration: int, t: int, bonus: int = 0):
    if t < bonus:
        return start_e
    slope = (end_e - start_e) / duration
    value = slope * (t - bonus) + start_e
    return max(value, end_e)


def apply_fire(q_network, target_network, init_q_network, args):
    def newton_schulz(matrix, num_iters=10):
        a, b = (1.5, -0.5)
        assert matrix.ndim == 2
        if num_iters is None:
            num_iters = 10
        do_transpose = matrix.size(1) > matrix.size(0)

        X = matrix
        if do_transpose:
            X = X.T

        X = X / X.norm()
        for _ in range(num_iters):
            A = X.T @ X
            X = a * X + b * X @ A

        if do_transpose:
            X = X.T
        return X

    print("***********FIRE at start************")
    with torch.no_grad():
        for m_q, m_target, m_init in zip(q_network.modules(), target_network.modules(), init_q_network.modules()):
            if isinstance(m_q, nn.Linear):
                m_q.weight.copy_(m_init.weight)
                m_target.weight.copy_(m_init.weight)
                if m_q.bias is not None:
                    m_q.bias.copy_(m_init.bias)
                    m_target.bias.copy_(m_init.bias)

            elif isinstance(m_q, nn.Conv2d):
                param = m_q.weight
                weight_matrix = param.data.detach().clone()
                assert weight_matrix.ndim == 4
                ortho_weight_matrix = torch.zeros_like(weight_matrix)
                for i in range(weight_matrix.shape[2]):
                    for j in range(weight_matrix.shape[3]):
                        ortho_weight_matrix[:, :, i, j] = newton_schulz(weight_matrix[:, :, i, j],
                                                                        num_iters=10)

                kernel_size = weight_matrix.shape[2] * weight_matrix.shape[3]
                scale = np.sqrt(weight_matrix.shape[0] / weight_matrix.shape[1]) / kernel_size
                ortho_weight_matrix *= scale

                param.data = ortho_weight_matrix
                m_target.weight.copy_(m_q.weight)


def apply_full_reset(q_network, target_network, init_q_network):
    print("***********FULL RESET at start************")
    with torch.no_grad():
        for m_q, m_target, m_init in zip(q_network.modules(), target_network.modules(), init_q_network.modules()):
            if isinstance(m_q, (nn.Linear, nn.Conv2d)):
                m_q.weight.copy_(m_init.weight)
                m_target.weight.copy_(m_init.weight)
                if m_q.bias is not None:
                    m_q.bias.copy_(m_init.bias)
                    m_target.bias.copy_(m_init.bias)


def apply_srdqn(q_network, target_network, init_q_network):
    """
    SR-DQN(S&P) block (Linear reset + Conv2d S&P) to apply only once at the start of resume training.
    """
    print("***********Before SPR************")
    for n, m_q in q_network.named_modules():
        if isinstance(m_q, (nn.Linear, nn.Conv2d)):
            print(n, m_q.weight.norm())
    print()

    with torch.no_grad():
        for m_q, m_target, m_init in zip(q_network.modules(), target_network.modules(), init_q_network.modules()):
            if isinstance(m_q, nn.Linear):
                m_q.weight.copy_(m_init.weight)
                m_target.weight.copy_(m_init.weight)
                if m_q.bias is not None:
                    m_q.bias.copy_(m_init.bias)
                    m_target.bias.copy_(m_init.bias)

            elif isinstance(m_q, nn.Conv2d):
                m_q.weight.data.mul_(0.2).add_(m_init.weight.data, alpha=0.8)
                m_target.weight.data.mul_(0.2).add_(m_init.weight.data, alpha=0.8)
                if m_q.bias is not None:
                    m_q.bias.data.mul_(0.2).add_(m_init.bias.data, alpha=0.8)
                    m_target.bias.data.mul_(0.2).add_(m_init.bias.data, alpha=0.8)

    print("***********After SPR************")
    for n, m_q in q_network.named_modules():
        if isinstance(m_q, (nn.Linear, nn.Conv2d)):
            print(n, m_q.weight.norm())
    print()


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

    # fire, full_reset, sr, pi should be set to 1 at most once
    active_options = sum([args.fire == 1, args.full_reset == 1, args.sr == 1, args.pi == 1])
    assert active_options <= 1, "fire, full_reset, sr, pi can only be set to 1 at most once"

    game = args.env_id.replace("NoFrameskip-v4", "")
    if args.resume_checkpoint_path == "":
        args.resume_checkpoint_path = f"runs/{game}_{args.seed}/midpoint_ckpt.pt"
    print(f"Loading checkpoint from {args.resume_checkpoint_path}")
    
    # assert args.num_envs == 1
    assert args.target_network_frequency % args.num_envs == 0, "target network frequency should be divisible by the number of envs"
    assert args.train_frequency % args.num_envs == 0 or args.num_envs % args.train_frequency == 0
    if args.train_frequency % args.num_envs == 0:
        repeat_update = 1
    elif args.num_envs % args.train_frequency == 0:
        repeat_update = args.num_envs // args.train_frequency

    # run_name is constructed from env_id and the option (fire/full_reset/sr/pi/vanilla), and seed
    # ex: BreakoutNoFrameskip-v4_fire_1, BreakoutNoFrameskip-v4_full_reset_1, BreakoutNoFrameskip-v4_sr_1, BreakoutNoFrameskip-v4_pi_1, BreakoutNoFrameskip-v4_vanilla_1
    if args.fire == 1:
        run_name = f"{game}_fire_{args.seed}"
    elif args.full_reset == 1:
        run_name = f"{game}_full_reset_{args.seed}"
    elif args.sr == 1:
        run_name = f"{game}_sr_{args.seed}"
    elif args.pi == 1:
        run_name = f"{game}_pi_{args.seed}"
    else:
        run_name = f"{game}_vanilla_{args.seed}"

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs)
    target_network = QNetwork(envs)

    if torch.cuda.device_count() > 1:
        q_network = nn.DataParallel(q_network)
        target_network = nn.DataParallel(target_network)
    q_network.to(device)
    target_network.to(device)

    target_network.load_state_dict(q_network.state_dict())

    from copy import deepcopy
    init_q_network = deepcopy(q_network)
    init_q_network.load_state_dict(q_network.state_dict())


    optimizer = optim.Adam(
        q_network.parameters(),
        lr=args.learning_rate,
        eps=1.5e-4,
    )

    # load checkpoint
    print(f"Loading checkpoint from {args.resume_checkpoint_path}")
    checkpoint = torch.load(args.resume_checkpoint_path, map_location=device)
    q_network.load_state_dict(checkpoint["q_network"])
    target_network.load_state_dict(checkpoint["target_network"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_global_step = int(checkpoint.get("global_step", 0))

    if "init_q_network" in checkpoint:
        init_q_network.load_state_dict(checkpoint["init_q_network"])

    # apply fire / full_reset / sr_dqn(S&P) / PI only once at the start of resume training
    if args.fire == 1:
        apply_fire(q_network, target_network, init_q_network, args)
    if args.full_reset == 1:
        apply_full_reset(q_network, target_network, init_q_network)
    if args.sr == 1:
        apply_srdqn(q_network, target_network, init_q_network)
    
    if args.pi == 1:
        print(f"***********Plasticity Injection at start ************")
        
        # Unwrap if DataParallel
        current_q = q_network.module if isinstance(q_network, nn.DataParallel) else q_network
        current_target = target_network.module if isinstance(target_network, nn.DataParallel) else target_network
        current_init = init_q_network.module if isinstance(init_q_network, nn.DataParallel) else init_q_network
        
        # Create PI Networks
        pi_q_network = PlasticityInjectedQNetwork(envs, old_q_network=current_q, init_q_network=current_init, device=device)
        pi_target_network = PlasticityInjectedQNetwork(envs, old_q_network=current_target, init_q_network=current_init, device=device)
        
        q_network = pi_q_network
        target_network = pi_target_network
            
        q_network.to(device)
        target_network.to(device)
        
        # Re-initialize Optimizer
        print("Re-initializing Optimizer for PI")
        optimizer = optim.Adam(
            q_network.parameters(),
            lr=args.learning_rate,
            eps=1.5e-4,
        )

    # load replay buffer together with the checkpoint
    assert "replay_buffer" in checkpoint, "Checkpoint does not contain replay_buffer"
    rb: ReplayBuffer = checkpoint["replay_buffer"]

    # resume training from the next step
    start_global_step = start_global_step + args.num_envs
    print(f"Resuming training from global_step={start_global_step}")
    start_time = time.time()

    obs, _ = envs.reset()
    for global_step in range(start_global_step, args.total_timesteps, args.num_envs):
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
                    qs = q_network(observations)

                    old_val = qs.gather(1, data.actions).squeeze()
                    loss = F.smooth_l1_loss(td_target, old_val)

                    if global_step % 160 == 0 and n_update == 0 and args.track:
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
            from cleanrl_utils.evals.dqn_eval import evaluate_steps

            if args.pi == 1:
                EvalModel = PlasticityInjectedQNetwork
            else:
                EvalModel = QNetwork

            model_path = f"runs/{run_name}/{global_step}"
            torch.save(q_network.state_dict(), model_path)
            episodic_returns = evaluate_steps(
                model_path,
                make_env,
                args.env_id,
                eval_steps=125000,
                run_name=f"{run_name}-eval",
                Model=EvalModel,
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

    envs.close()
    if args.track:
        wandb.finish()
