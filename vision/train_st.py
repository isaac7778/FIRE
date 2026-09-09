import gc
import math
import sys
import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import numpy as np

from models import get_resnet18_CIFAR10, get_TinyViT_CIFAR100, get_VGG16_TinyImageNet
from task import TASKS
from dst_log_utils import ITOPTracker, get_sparsity_stats, get_current_pruning_ratio

# Add the bundled sparsimony repo to sys.path once at import time.
_sparsimony_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparsimony')
if os.path.isdir(_sparsimony_path) and _sparsimony_path not in sys.path:
    sys.path.insert(0, _sparsimony_path)


# ---------------------------------------------------------------------------
# Model / task / optimizer helpers
# ---------------------------------------------------------------------------

def get_optimizer(model, cfg):
    OPTIMIZERS = {'adam': torch.optim.Adam}
    if cfg.optimizer in OPTIMIZERS:
        return OPTIMIZERS[cfg.optimizer](model.parameters(), lr=cfg.lr)
    print(f"Optimizer '{cfg.optimizer}' not supported. Falling back to adam.")
    return torch.optim.Adam(model.parameters(), lr=cfg.lr)


def build_model(cfg):
    if cfg.model == 'RESNET18':
        assert cfg.task == 'CIFAR10'
        return get_resnet18_CIFAR10()
    elif cfg.model == 'TinyViT':
        assert cfg.task == 'CIFAR100'
        return get_TinyViT_CIFAR100()
    elif cfg.model == 'VGG16':
        assert cfg.task == 'TinyImageNet'
        return get_VGG16_TinyImageNet()
    raise ValueError(f"Invalid model: {cfg.model}")


def get_task(cfg):
    if cfg.benchmark == 'class_incremental':
        assert cfg.model in ('TinyViT', 'VGG16')
    benchmark_settings = {
        'warm_start':        {'n_epochs': 100, 'n_chunks': 2,  'mode': 'sample'},
        'continual':         {'n_epochs': 100, 'n_chunks': 10, 'mode': 'sample'},
        'class_incremental': {'n_epochs': 100, 'n_chunks': 20, 'mode': 'class'},
    }
    s = benchmark_settings[cfg.benchmark]
    task = TASKS[cfg.task](
        mode=s['mode'],
        n_chunks=s['n_chunks'],
        make_test_loader=True,
        access='full',
        test_access='same',
        seed=cfg.seed,
        warm_start_subset_ratio=cfg.warm_start_subset_ratio,
    )
    cfg.n_epochs = s['n_epochs']
    return task


# ---------------------------------------------------------------------------
# Gradient-step counter
# ---------------------------------------------------------------------------

def compute_total_gradient_steps(cfg, task):
    """Return the total number of optimizer.step() calls for the full run.

    Uses actual chunk sizes from the task object so the count is exact
    regardless of benchmark / dataset / batch_size.  This value is used to
    derive sparsifier hyperparameters (t_end, delta_t) so they scale
    correctly without manual tuning.
    """
    total = 0
    for i_iter in range(task.n_chunks):
        if cfg.benchmark == 'warm_start' and i_iter == 0:
            log_every = 100 // cfg.warm_start_subset_ratio
        else:
            log_every = cfg.log_every
        real_epochs = cfg.n_epochs * log_every
        chunk_size = len(task._train_datasets[i_iter])
        steps_per_epoch = math.ceil(chunk_size / cfg.batch_size)
        total += real_epochs * steps_per_epoch
    return total


# ---------------------------------------------------------------------------
# Sparsifier factory
# ---------------------------------------------------------------------------

def build_sparsifier(cfg, model, optimizer, total_steps):
    """Create, configure, and prepare the requested sparsifier.

    Derived hyperparameters (printed for reproducibility):
        t_end   = cfg.t_end_ratio  * total_steps          (all sparse methods)
        delta_t = total_steps // cfg.num_mask_updates      (all sparse methods)
        t_accel = cfg.t_accel_ratio * total_steps          (GMP only)

    Returns None when cfg.sparsifier == 'dense'.

    Supported values of cfg.sparsifier:
        'dense'  – no sparsification (baseline)
        'rigl'   – Rig-L  (ERK distribution, cosine decay schedule)
        'set'    – SET    (uniform distribution, constant schedule)
        'gmp'    – GMP*   (uniform distribution, accelerated cubic schedule)
        'static' – Static magnitude pruning (one-shot, no regrowth)
    """
    if cfg.sparsifier == 'dense':
        return None

    from sparsimony import rigl, gmp, static
    from sparsimony import set as sp_set  # avoid shadowing Python's built-in

    t_end = int(cfg.t_end_ratio * total_steps)

    if cfg.sparsifier == 'rigl':
        delta_t = max(1, t_end // cfg.num_mask_updates)
        sparsifier = rigl(
            optimizer,
            sparsity=cfg.sparsity,
            t_end=t_end,
            delta_t=delta_t,
            pruning_ratio=cfg.pruning_ratio,
        )

    elif cfg.sparsifier == 'set':
        delta_t = max(1, t_end // cfg.num_mask_updates)
        sparsifier = sp_set(
            optimizer,
            sparsity=cfg.sparsity,
            t_end=t_end,
            delta_t=delta_t,
            pruning_ratio=cfg.pruning_ratio,
        )

    elif cfg.sparsifier == 'gmp':
        t_accel = int(cfg.t_accel_ratio * total_steps)
        delta_t = max(1, (t_end - t_accel) // cfg.num_mask_updates)
        sparsifier = gmp(
            optimizer,
            t_accel=t_accel,
            t_end=t_end,
            delta_t=delta_t,
            initial_sparsity=cfg.initial_sparsity,
            final_sparsity=cfg.sparsity,
        )

    elif cfg.sparsifier == 'static':
        sparsifier = static(
            optimizer,
            sparsity=cfg.sparsity,
        )

    else:
        raise ValueError(
            f"Unknown sparsifier '{cfg.sparsifier}'. "
            "Choose from: dense, rigl, set, gmp, static"
        )

    # Prepare: reparametrize all Conv2d and Linear weight tensors
    sparse_config = [
        {"tensor_fqn": f"{fqn}.weight"}
        for fqn, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d)) and fqn != 'head.fc.0'
    ]
    sparsifier.prepare(model, sparse_config)

    print(
        f"[Sparsifier] {cfg.sparsifier} | sparsity={cfg.sparsity} | "
        f"total_steps={total_steps} | t_end={t_end} | delta_t={delta_t}"
    )
    return sparsifier


# ---------------------------------------------------------------------------
# W&B run-name builder
# ---------------------------------------------------------------------------

def build_run_name(cfg, sparsifier) -> str:
    base = f"{cfg.model}_{cfg.task}"
    if cfg.sparsifier == 'dense':
        return f"{base}_dense"
    # delta_t is stored on the scheduler for all non-static methods
    dt = getattr(getattr(sparsifier, 'scheduler', None), 'delta_t', None)
    if cfg.sparsifier in ('rigl', 'set'):
        return (
            f"{base}_dst"
            f"_sparsity_{cfg.sparsity}"
            f"_pruning_ratio_{cfg.pruning_ratio}"
            f"_delta_t_{dt}"
        )
    if cfg.sparsifier == 'gmp':
        return (
            f"{base}_gmp"
            f"_accel_sparsity_{cfg.initial_sparsity}"
            f"_final_sparsity_{cfg.sparsity}"
            f"_delta_t_{dt}"
        )
    if cfg.sparsifier == 'static':
        return f"{base}_static_sparsity_{cfg.sparsity}"
    return f"{base}_{cfg.sparsifier}"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(cfg):
    cfg.print()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build task first so cfg.n_epochs is populated before step counting
    task = get_task(cfg)
    model = build_model(cfg).to(device)
    optimizer = get_optimizer(model, cfg)

    total_steps = compute_total_gradient_steps(cfg, task)
    # init DDP AFTER reparametrization if using distributed training
    sparsifier = build_sparsifier(cfg, model, optimizer, total_steps)
    itop_tracker = ITOPTracker(sparsifier) if sparsifier is not None else None

    criterion = nn.CrossEntropyLoss()
    initial_lr = 0.0
    warmup_rate = 0.1

    wandb_project = cfg.wandb_project or f"{cfg.benchmark}_{cfg.task}_{cfg.model}"
    wandb.init(
        entity="ucalgary",
        project=wandb_project,
        name=build_run_name(cfg, sparsifier),
        config=cfg.__dict__,
        mode="disabled" if cfg.disable_wandb else "online",
    )

    global_epoch = 0
    global_step = 0

    for i_iter in range(task.n_chunks):
        if cfg.benchmark == 'warm_start' and i_iter == 0:
            log_every = 100 // cfg.warm_start_subset_ratio
        else:
            log_every = cfg.log_every

        trainloader = task.set_level(i_iter, batch_size=cfg.batch_size)
        real_epochs = cfg.n_epochs * log_every
        pbar = tqdm(range(real_epochs), leave=True)

        # Reset optimizer at every iteration; hand the new instance to sparsifier
        optimizer = get_optimizer(model, cfg)
        target_lr = [pg['lr'] for pg in optimizer.param_groups]

        if sparsifier is not None:
            sparsifier.optimizer = optimizer
            if hasattr(sparsifier, 'zero_inactive_param_momentum_buffers'):
                sparsifier.zero_inactive_param_momentum_buffers()

        for epoch in pbar:
            pbar.set_description(f'Iter {i_iter} | Epoch {epoch}')
            do_logging = global_epoch % log_every == 0

            # Warmup LR scheduling (from https://arxiv.org/abs/2406.02596)
            ls = global_step % cfg.n_epochs
            we = cfg.n_epochs * warmup_rate
            remain = (epoch + 1) / cfg.n_epochs - int((epoch + 1) / cfg.n_epochs)
            for i, pg in enumerate(optimizer.param_groups):
                if ls < we:
                    current_lr = initial_lr + (target_lr[i] - initial_lr) * remain * (10 // log_every)
                else:
                    current_lr = target_lr[i]
                pg['lr'] = current_lr

            total = correct = 0
            for inputs, labels, _orig_idx, _chunk_idx in trainloader:
                model.train()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                if cfg.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

                optimizer.step()
                if sparsifier is not None:
                    if sparsifier.step():
                        itop_tracker.update()

            train_acc = correct / total

            if do_logging:
                global_step += 1
                log_dict = {
                    'train/acc': train_acc,
                    'train/lr': current_lr,
                    'level': i_iter,
                    'global_step': global_step,
                    'global_epoch': global_epoch,
                    'iter': i_iter,
                }
                p_fix = {'acc': train_acc, 'lr': current_lr}

                test_acc, _ = task.test(model, device)
                log_dict['test/acc'] = test_acc
                p_fix['test_acc'] = test_acc

                if cfg.benchmark == 'class_incremental':
                    acc_full, _ = task.test(model, device, full=True)
                    log_dict['test/acc_full'] = acc_full
                    p_fix['test_acc_full'] = acc_full

                if sparsifier is not None:
                    sparsity_stats = get_sparsity_stats(model)
                    log_dict['dst/mask_sparsity']  = sparsity_stats['mask_sparsity']
                    log_dict['dst/weight_sparsity'] = sparsity_stats['weight_sparsity']
                    log_dict['dst/itop_rate']       = itop_tracker.compute()
                    pruning_ratio = get_current_pruning_ratio(sparsifier)
                    if pruning_ratio is not None:
                        log_dict['dst/pruning_ratio'] = pruning_ratio

                wandb.log(log_dict, step=global_step)
                pbar.set_postfix(**p_fix)

            global_epoch += 1

        del trainloader
        torch.cuda.empty_cache()
        gc.collect()

    wandb.finish()


if __name__ == "__main__":
    from config_st import get_config
    cfg = get_config()
    main(cfg)
