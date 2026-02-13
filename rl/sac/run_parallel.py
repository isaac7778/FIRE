import argparse
import copy
import multiprocessing as mp
import os
import time

import numpy as np


def get_egl_device_id(server, cuda_id):
    if server == "kaist":
        cuda_id_to_egl_id = lambda i: [3,2,1,0,7,6,5,4][i]
    elif server == "kaist4":
        cuda_id_to_egl_id = lambda i: [1,0,3,2][i]
    elif server == "kaist7":
        cuda_id_to_egl_id = lambda i: [2,1,0,6,5,4,3][i]
    elif server == "kaist3":
        cuda_id_to_egl_id = lambda i: [0,2,1][i]
    elif server == "kaist139":
        cuda_id_to_egl_id = lambda i: [1,0,2][i]
    elif server == "kaist148":
        cuda_id_to_egl_id = lambda i: [3,2,1,0,6,5,4][i]
    elif server == "kaist67":
        cuda_id_to_egl_id = lambda i: [0,1,2,3,4,5,6,7][i]
    elif server == "local":
        cuda_id_to_egl_id = lambda i: 0
    else:
        raise ValueError

    return str(cuda_id_to_egl_id(int(cuda_id)))

def run_with_device(server, device_id, config_path, config_name, overrides):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = get_egl_device_id(server, device_id)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    # Now import the main script
    if config_name == "online_rl":
        from run_online import run
    elif config_name == "offline_rl":
        from run_offline import run
    else:
        raise NotImplementedError

    args = {
        "config_path": config_path,
        "config_name": config_name,
        "overrides": overrides,
    }
    run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="online_rl")
    parser.add_argument("--agent_config", type=str, default="hyper_simba")
    parser.add_argument("--env_type", type=str, default="dmc_hard")
    parser.add_argument("--device_ids", default=[0], nargs="+")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--num_exp_per_device", type=int, default=1)
    parser.add_argument("--server", type=str, default="local")
    parser.add_argument("--group_name", type=str, default="test")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--overrides", action="append", default=[])

    args = vars(parser.parse_args())
    seeds = (np.arange(args.pop("num_seeds")) * 1000).tolist()
    device_ids = args.pop("device_ids")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

    num_devices = len(device_ids)
    num_exp_per_device = args.pop("num_exp_per_device")
    pool_size = num_devices * num_exp_per_device

    # create configurations for child run
    experiments = []
    config_path = args.pop("config_path")
    config_name = args.pop("config_name")
    server = args.pop("server")
    group_name = args.pop("group_name")
    exp_name = args.pop("exp_name")
    agent_config = args.pop("agent_config")

    # import library after CUDA_VISIBLE_DEVICES operation
    # from scale_rl.envs.d4rl import D4RL_MUJOCO
    from scale_rl.envs.dmc import DMC_EASY_MEDIUM, DMC_HARD
    from scale_rl.envs.humanoid_bench import HB_LOCOMOTION_NOHAND
    from scale_rl.envs.mujoco import MUJOCO_ALL
    from scale_rl.envs.myosuite import MYOSUITE_TASKS

    env_type = args.pop("env_type")

    ###################
    # offline
    if env_type == "d4rl_mujoco":
        envs = D4RL_MUJOCO
        env_configs = ["d4rl"] * len(envs)

    ###################
    # online
    elif env_type == "hb_utd_scaling":
        envs = [
            "h1-balance_simple-v0",
            "h1-walk-v0",
            "h1-run-v0",
        ]
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "mujoco":
        envs = MUJOCO_ALL
        env_configs = [env_type] * len(envs)

    elif env_type == "dmc_em":
        envs = DMC_EASY_MEDIUM
        env_configs = ["dmc"] * len(envs)

    elif env_type == "dmc_hard":
        envs = DMC_HARD
        env_configs = ["dmc"] * len(envs)

    elif env_type == "myosuite":
        envs = MYOSUITE_TASKS
        env_configs = [env_type] * len(envs)

    elif env_type == "hb_locomotion":
        envs = HB_LOCOMOTION_NOHAND
        env_configs = [env_type] * len(envs)

    elif env_type == "hb_scaling":
        envs = [
            "h1-balance_simple-v0",
            "h1-run-v0",
            "h1-stair-v0",
            "h1-walk-v0",
            "h1-sit_hard-v0",
        ]
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "hb_scaling_4":
        envs = [
            "h1-stand-v0",
            "h1-reach-v0",
            "h1-hurdle-v0",
            "h1-crawl-v0",
        ]
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "hb_scaling_5":
        envs = [
            "h1-maze-v0",
            "h1-sit_simple-v0",
            "h1-balance_hard-v0",
            "h1-slide-v0",
            "h1-pole-v0",
        ]
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "hb_scaling_9":
        envs = [
            "h1-stand-v0",
            "h1-reach-v0",
            "h1-hurdle-v0",
            "h1-crawl-v0",
            "h1-maze-v0",
            "h1-sit_simple-v0",
            "h1-balance_hard-v0",
            "h1-slide-v0",
            "h1-pole-v0",
        ]
        env_configs = ["hb_locomotion"] * len(envs)

    elif env_type == "dmc_em_0_5":
        envs = DMC_EASY_MEDIUM[:5]
        env_configs = ["dmc"] * len(envs)

    elif env_type == "dmc_em_5_10":
        envs = DMC_EASY_MEDIUM[5:10]
        env_configs = ["dmc"] * len(envs)

    elif env_type == "dmc_em_10_15":
        envs = DMC_EASY_MEDIUM[10:15]
        env_configs = ["dmc"] * len(envs)

    elif env_type == "dmc_em_15_21":
        envs = DMC_EASY_MEDIUM[15:]
        env_configs = ["dmc"] * len(envs)

    elif env_type == "dmc_em_0_10":
        envs = DMC_EASY_MEDIUM[:10]
        env_configs = ["dmc"] * len(envs)

    elif env_type == "dmc_em_10_21":
        envs = DMC_EASY_MEDIUM[10:]
        env_configs = ["dmc"] * len(envs)

    elif env_type == "all":
        envs = (
            MUJOCO_ALL
            + DMC_EASY_MEDIUM
            + DMC_HARD
            + MYOSUITE_TASKS
            + HB_LOCOMOTION_NOHAND
        )
        env_configs = (
            ["mujoco"] * len(MUJOCO_ALL)
            + ["dmc"] * len(DMC_EASY_MEDIUM)
            + ["dmc"] * len(DMC_HARD)
            + ["myosuite"] * len(MYOSUITE_TASKS)
            + ["hb_locomotion"] * len(HB_LOCOMOTION_NOHAND)
        )

    elif env_type == "mini_benchmark":
        mujoco_envs = [
            "HalfCheetah-v4",
            "Humanoid-v4",
        ]

        dmc_em_envs = [
            "acrobot-swingup",
            "cartpole-swingup_sparse",
            "cheetah-run",
            "finger-spin",
        ]

        dmc_hard_envs = [
            "dog-run",
            "dog-trot",
            "humanoid-stand",
            "humanoid-walk",
        ]

        myo_envs = [
            "myo-key-turn-hard",
            "myo-pen-twirl-hard",
        ]

        hb_envs = [
            "h1-balance_simple-v0",
            "h1-run-v0",
            "h1-stair-v0",
            "h1-stand-v0",
        ]

        envs = mujoco_envs + dmc_em_envs + dmc_hard_envs + myo_envs + hb_envs
        env_configs = (
            ["mujoco"] * len(mujoco_envs)
            + ["dmc"] * len(dmc_em_envs)
            + ["dmc"] * len(dmc_hard_envs)
            + ["myosuite"] * len(myo_envs)
            + ["hb_locomotion"] * len(hb_envs)
        )

    else:
        raise NotImplementedError

    for seed in seeds:
        for idx, env_name in enumerate(envs):
            exp = copy.deepcopy(args)  # copy overriding arguments
            exp["config_path"] = config_path
            exp["config_name"] = config_name

            exp["overrides"].append("agent=" + agent_config)
            exp["overrides"].append("env=" + env_configs[idx])
            exp["overrides"].append("env.env_name=" + env_name)

            exp["overrides"].append("server=" + server)
            exp["overrides"].append("group_name=" + group_name)
            exp["overrides"].append("exp_name=" + exp_name)
            exp["overrides"].append("seed=" + str(seed))

            experiments.append(exp)
            print(exp)

    # run parallel experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method("spawn")
    available_gpus = device_ids
    process_dict = {gpu_id: [] for gpu_id in device_ids}

    for exp in experiments:
        wait = True
        # wait until there exists a finished process
        while wait:
            # Find all finished processes and register available GPU
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)

            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False
                    gpu_id, processes = min(
                        process_dict.items(), key=lambda x: len(x[1])
                    )
                    break

            time.sleep(10)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        exp["device_id"] = str(gpu_id)
        process = mp.Process(
            target=run_with_device,
            args=(
                server,
                exp["device_id"],
                exp["config_path"],
                exp["config_name"],
                exp["overrides"],
            ),
        )
        process.start()
        processes.append(process)
        print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)