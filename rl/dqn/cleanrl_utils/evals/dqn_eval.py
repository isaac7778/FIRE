import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


# def evaluate(
#     model_path: str,
#     make_env: Callable,
#     env_id: str,
#     eval_episodes: int,
#     run_name: str,
#     Model: torch.nn.Module,
#     device: torch.device = torch.device("cpu"),
#     epsilon: float = 0.05,
#     capture_video: bool = True,
# ):
#     envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
#     model = Model(envs).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     obs, _ = envs.reset()
#     episodic_returns = []
#     while len(episodic_returns) < eval_episodes:
#         if random.random() < epsilon:
#             actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
#         else:
#             q_values = model(torch.Tensor(obs).to(device))
#             actions = torch.argmax(q_values, dim=1).cpu().numpy()
#         next_obs, _, _, _, infos = envs.step(actions)
#         if "final_info" in infos:
#             for info in infos["final_info"]:
#                 if "episode" not in info:
#                     continue
#                 print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
#                 episodic_returns += [info["episode"]["r"]]
#         obs = next_obs
#
#     return episodic_returns


# edit evaluate function with steps
def evaluate_steps(
        model_path: str,
        make_env: Callable,
        env_id: str,
        eval_steps: int,
        run_name: str,
        Model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        epsilon: float = 0.05,
        capture_video: bool = True,
        num_envs: int = 1,
):
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_id, 46782132+i, i, capture_video, run_name) for i in range(num_envs)]
    )
    model = Model(envs)
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(model)


    obs, _ = envs.reset()
    episodic_returns = []
    total_steps = 0
    while total_steps < eval_steps:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)

        # Count steps and handle final information
        total_steps += envs.num_envs
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None or "episode" not in info:
                    continue
                print(f"total_steps={total_steps}, episodic_return={info['episode']['r']}")
                episodic_returns.append(info["episode"]["r"])
        obs = next_obs

    # episodic_returns = np.mean(episodic_returns).astype(float)
    return episodic_returns


ATARI_HUMAN_SCORES = {
    'alien': 7127.7,
    'amidar': 1719.5,
    'assault': 742.0,
    'asterix': 8503.3,
    'asteroids': 47388.7,
    'atlantis': 29028.1,
    'bankheist': 753.1,
    'battlezone': 37187.5,
    'beamrider': 16926.5,
    'berzerk': 2630.4,
    'bowling': 160.7,
    'boxing': 12.1,
    'breakout': 30.5,
    'centipede': 12017.0,
    'choppercommand': 7387.8,
    'crazyclimber': 35829.4,
    'demonattack': 1971.0,
    'doubledunk': -16.4,
    'enduro': 860.5,
    'fishingderby': -38.7,
    'freeway': 29.6,
    'frostbite': 4334.7,
    'gopher': 2412.5,
    'gravitar': 3351.4,
    'hero': 30826.4,
    'icehockey': 0.9,
    'jamesbond': 302.8,
    'kangaroo': 3035.0,
    'krull': 2665.5,
    'kungfumaster': 22736.3,
    'montezumarevenge': 4753.3,
    'mspacman': 6951.6,
    'namethisgame': 8049.0,
    'phoenix': 7242.6,
    'pitfall': 6463.7,
    'pong': 14.6,
    'privateeye': 69571.3,
    'qbert': 13455.0,
    'riverraid': 17118.0,
    'roadrunner': 7845.0,
    'robotank': 11.9,
    'seaquest': 42054.7,
    'skiing': -4336.9,
    'solaris': 12326.7,
    'spaceinvaders': 1668.7,
    'stargunner': 10250.0,
    'tennis': -8.3,
    'timepilot': 5229.2,
    'tutankham': 167.6,
    'upndown': 11693.2,
    'venture': 1187.5,
    'videopinball': 17667.9,
    'wizardofwor': 4756.5,
    'yarsrevenge': 54576.9,
    'zaxxon': 9173.3,
}

ATARI_RANDOM_SCORES = {
    'alien': 227.8,
    'amidar': 5.8,
    'assault': 222.4,
    'asterix': 210.0,
    'asteroids': 719.1,
    'atlantis': 12850.0,
    'bankheist': 14.2,
    'battlezone': 2360.0,
    'beamrider': 363.9,
    'berzerk': 123.7,
    'bowling': 23.1,
    'boxing': 0.1,
    'breakout': 1.7,
    'centipede': 2090.9,
    'choppercommand': 811.0,
    'crazyclimber': 10780.5,
    'defender': 2874.5,
    'demonattack': 152.1,
    'doubledunk': -18.6,
    'enduro': 0.0,
    'fishingderby': -91.7,
    'freeway': 0.0,
    'frostbite': 65.2,
    'gopher': 257.6,
    'gravitar': 173.0,
    'hero': 1027.0,
    'icehockey': -11.2,
    'jamesbond': 29.0,
    'kangaroo': 52.0,
    'krull': 1598.0,
    'kungfumaster': 258.5,
    'montezumarevenge': 0.0,
    'mspacman': 307.3,
    'namethisgame': 2292.3,
    'phoenix': 761.4,
    'pitfall': -229.4,
    'pong': -20.7,
    'privateeye': 24.9,
    'qbert': 163.9,
    'riverraid': 1338.5,
    'roadrunner': 11.5,
    'robotank': 2.2,
    'seaquest': 68.4,
    'skiing': -17098.1,
    'solaris': 1236.3,
    'spaceinvaders': 148.0,
    'stargunner': 664.0,
    'surround': -10.0,
    'tennis': -23.8,
    'timepilot': 3568.0,
    'tutankham': 11.4,
    'upndown': 533.4,
    'venture': 0.0,
    'videopinball': 0.0,
    'wizardofwor': 563.5,
    'yarsrevenge': 3092.9,
    'zaxxon': 32.5,
}


def normalize_score(ret, game):
    ret = np.array(ret)
    return (ret - ATARI_RANDOM_SCORES[game]) / (
        ATARI_HUMAN_SCORES[game] - ATARI_RANDOM_SCORES[game]
    )


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.dqn import QNetwork, make_env

    model_path = hf_hub_download(repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="q_network.pth")
    # evaluate(
    #     model_path,
    #     make_env,
    #     "CartPole-v1",
    #     eval_episodes=10,
    #     run_name=f"eval",
    #     Model=QNetwork,
    #     device="cpu",
    #     capture_video=False,
    # )
