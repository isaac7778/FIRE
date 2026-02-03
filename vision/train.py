import gc
import copy
import torch
import wandb
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from models import get_resnet18_CIFAR10, get_TinyViT_CIFAR100, get_VGG16_TinyImageNet
from task import TASKS

from interventions.node_reset.redo import ReDO
from interventions.node_reset.cbp import CBP
from interventions.node_reset.snr import SNR
from interventions.shrink_and_perturb import shrink_and_perturb
from interventions.dash import dash
from interventions.parseval import parseval_loss
from interventions.fire import fire
from interventions.muon import SingleDeviceMuonWithAuxAdam

def get_optimizer(model, cfg):
    OPTIMIZERS = {
        'adam': torch.optim.Adam,
        'muon': SingleDeviceMuonWithAuxAdam,
    }

    if cfg.optimizer == 'muon':
        adam_lr_ratio = 0.1
        if cfg.model == 'RESNET18':
            filter_params = [p for p in model.backbone.parameters() if len(p.shape) == 4 and p.requires_grad]
            hidden_gains_biases = [p for p in model.backbone.parameters() if p.ndim < 2 and p.requires_grad]
            head_weights = [p for p in model.head.parameters() if p.ndim >= 2 and p.requires_grad]
            head_biases = [p for p in model.head.parameters() if p.ndim < 2 and p.requires_grad]
            muon_params = filter_params
            adam_params = hidden_gains_biases + head_weights + head_biases
        elif cfg.model == 'TinyViT':
            hidden_weights = [p for p in model.backbone.parameters() if p.ndim >= 2 and p.requires_grad]
            hidden_gains_biases = [p for p in model.backbone.parameters() if p.ndim < 2 and p.requires_grad]
            head_weights = [p for p in model.head.parameters() if p.ndim >= 2 and p.requires_grad]
            head_gains_biases = [p for p in model.head.parameters() if p.ndim < 2 and p.requires_grad]
            muon_params = hidden_weights
            adam_params = hidden_gains_biases + head_weights + head_gains_biases
        elif cfg.model == 'VGG16':
            hidden_weights = [p for p in model.backbone.parameters() if p.ndim >= 2 and p.requires_grad]
            hidden_gains_biases = [p for p in model.backbone.parameters() if p.ndim < 2 and p.requires_grad]
            head_weights = [p for p in model.head.parameters() if p.ndim >= 2 and p.requires_grad]
            head_biases = [p for p in model.head.parameters() if p.ndim < 2 and p.requires_grad]
            muon_params = hidden_weights
            adam_params = hidden_gains_biases + head_weights + head_biases
        else:
            raise NotImplementedError(f"Model {cfg.model} not supported for muon optimizer.")
        param_groups = [
            dict(params=muon_params, use_muon=True,
                 lr=cfg.lr / adam_lr_ratio),
            dict(params=adam_params, use_muon=False,
                 lr=cfg.lr, betas=(0.9, 0.95)),
        ]
        optimizer = OPTIMIZERS[cfg.optimizer](param_groups)
    else:
        optimizer = OPTIMIZERS[cfg.optimizer](model.parameters(), lr=cfg.lr)
    return optimizer

def build_model(cfg):
    if cfg.model == 'RESNET18':
        assert cfg.task == 'CIFAR10'
        model = get_resnet18_CIFAR10()

    elif cfg.model == 'TinyViT':
        assert cfg.task == 'CIFAR100'
        model = get_TinyViT_CIFAR100()

    elif cfg.model == 'VGG16':
        assert cfg.task == 'TinyImageNet'
        model = get_VGG16_TinyImageNet()
    else:
        raise ValueError(f"Invalid model_name: {cfg.model}")

    return model

def get_task(cfg):
    if cfg.benchmark=='class_incremental':
        assert cfg.model == 'TinyViT' or cfg.model == 'VGG16'
    benchmark_settings = {
        'warm_start': {'n_epochs': 100, 'n_chunks': 2, 'mode': 'sample'},
        'continual': {'n_epochs': 100, 'n_chunks': 10, 'mode': 'sample'},
        'class_incremental': {'n_epochs': 100, 'n_chunks': 20, 'mode': 'class'},
    }
    task_setting = benchmark_settings[cfg.benchmark]

    task = TASKS[cfg.task](
        mode=task_setting['mode'],
        n_chunks=task_setting['n_chunks'],
        make_test_loader=True,
        access='full',
        test_access='same',
        seed=cfg.seed,
        warm_start_subset_ratio=cfg.warm_start_subset_ratio,
    )
    cfg.n_epochs = task_setting['n_epochs']
    return task

def main(cfg):

    cfg.print()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Initialize task
    task = get_task(cfg)

    # Initialize model
    model = build_model(cfg).to(device)

    # for interventions
    init_model = deepcopy(model)

    if cfg.redo['enable']:
        redo_handle = ReDO(model, period=cfg.redo['period'], threshold=cfg.redo['threshold'])

    if cfg.cbp['enable']:
        cbp_handle = CBP(model, period=1, replacement_rate=cfg.cbp['replacement_rate'], maturity_threshold=cfg.cbp['maturity_threshold'])

    if cfg.snr['enable']:
        snr_handle = SNR(model, period=1, threshold_reset_freq=cfg.snr['tau_reset_freq'], threshold_percentile=cfg.snr['tau_percentile'])

    # Initialize optimizer
    optimizer = get_optimizer(model, cfg)
    initial_lr = 0.0
    target_lr = [param_group['lr'] for param_group in optimizer.param_groups]
    warmup_rate = 0.1

    # Initialize loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    wandb.init(
        project=f"{cfg.benchmark}_{cfg.task}_{cfg.model}",
        name="run1",
        config=cfg,
        mode="disabled" if cfg.disable_wandb else "online",
    )

    global_epoch = 0
    global_step = 0

    # main loop
    for i_iter in range(task.n_chunks):
        # loggings and steps
        if cfg.benchmark=='warm_start' and i_iter==0:
            log_every = 100 // cfg.warm_start_subset_ratio
        else:
            log_every = cfg.log_every

        # Set level
        trainloader = task.set_level(i_iter, batch_size=cfg.batch_size)
        real_epochs = cfg.n_epochs * log_every
        pbar = tqdm(range(real_epochs), leave=True)

        optimizer = get_optimizer(model, cfg) # reset optimizer at every iteration
        
        # reinitialization-based interventions
        if i_iter > 0:
            # shrink and perturb
            if cfg.snp['coef'] > 0:
                shrink_and_perturb(model, init_model, cfg.snp['coef'])

            # full reset
            if cfg.full_reset['enable']:
                ref_model = copy.deepcopy(model)

                for (hname, hparam), (tname, tparam) in zip(model.named_parameters(),
                                                            init_model.named_parameters()):
                    assert hname == tname
                    hparam.data = tparam.data.clone()

                for (hname, hbuffer), (tname, tbuffer) in zip(model.named_buffers(),
                                                            init_model.named_buffers()):
                    assert hname == tname
                    hbuffer.data = tbuffer.data.clone()

            # dash
            if cfg.dash['alpha'] > 0 or cfg.dash['lambda'] > 0:
                dataloader = deepcopy(trainloader)
                dash(model, cfg.dash['alpha'], cfg.dash['lambda'], dataloader, criterion, device)
            
            # fire
            if cfg.fire['enable']:
                fire(model, iteration=cfg.fire['iter_num'], is_vit=cfg.model=="TinyViT")
        
        # muon optimizer
        if cfg.optimizer == "muon" and cfg.benchmark == "warm_start":
            if i_iter == 0:
                print("[Muon] Use adam optimizer at iteration 0")
                dummy_cfg = copy.deepcopy(cfg)
                dummy_cfg.optimizer = "adam"
                optimizer = get_optimizer(model, dummy_cfg)
                print("[Muon] Optimizer: {}".format(type(optimizer)))
            else:
                print("[Muon] Use muon optimizer at iteration 1")
                optimizer = get_optimizer(model, cfg)
                print("[Muon] Optimizer: {}".format(type(optimizer)))
            target_lr = [param_group['lr'] for param_group in optimizer.param_groups]  # 목표 학습률 (10 에포크 이후에 유지할 값)

        # Train model
        for epoch in pbar:
            pbar.set_description(f'Iter - {i_iter}, Epoch - {epoch}')

            do_logging = global_epoch % log_every == 0
            
            # warmup learning rate scheduling: from https://arxiv.org/abs/2406.02596
            ls = global_step % cfg.n_epochs
            we = cfg.n_epochs * warmup_rate
            remain = (epoch+1) / cfg.n_epochs - int((epoch+1) / cfg.n_epochs)
            for i, param_group in enumerate(optimizer.param_groups):
                if ls < we:
                    current_lr = initial_lr + (target_lr[i] - initial_lr) * remain * (10 // log_every)

                else:
                    # 10 에포크 이후에는 target_lr 유지
                    current_lr = target_lr[i]
                param_group['lr'] = current_lr

            # epoch loop
            current_lr = cfg.lr
            total = 0
            correct = 0
            for i_step, (inputs, labels, original_indices, chunk_indices) in enumerate(trainloader, 0):
                model.train()

                inputs, labels = inputs.to(device), labels.to(device)

                # forward
                outputs = model(inputs)

                # calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                ce_loss = criterion(outputs, labels)#=========================CE loss

                # intervention: regularization-based 
                regen_loss = 0.0
                pars_loss = 0.0
                if not (cfg.benchmark == "warm_start" and i_iter == 0):

                    if cfg.regen['coef'] > 0:
                        for (name, param), (tar_name, tar_param) in zip(model.named_parameters(), init_model.named_parameters()):
                            assert name==tar_name
                            regen_loss += torch.sum((param - tar_param.data) ** 2)
                    
                    if cfg.parseval_reg['enable']:
                        pars_loss = parseval_loss(model, scale=1) * cfg.parseval_reg['coef']

                # compute loss
                loss = (
                    ce_loss
                    + cfg.regen['coef']/2.0 * regen_loss
                    + pars_loss
                )
                # backward
                optimizer.zero_grad()
                loss.backward()

                if cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

                optimizer.step()

                # intervention: cbp
                if cfg.cbp['enable'] and not (cfg.benchmark == 'warm_start' and i_iter==0):
                    cbp_handle.apply(trainloader)

                # intervention: snr
                if cfg.snr['enable'] and not (cfg.benchmark == 'warm_start' and i_iter==0):
                    snr_handle.apply(trainloader, optimizer)

            train_acc = correct / total

            if do_logging:
                global_step += 1

                log_dict = {
                    'train/acc': train_acc,
                    'train/lr': current_lr,
                    'level': i_iter,
                    'global_step': global_step, 'global_epoch': global_epoch, 'iter': i_iter,
                }
                p_fix_dict = {
                    "acc": train_acc,
                    "lr": current_lr,
                }

                test_acc, test_info = task.test(model, device)
                log_dict['test/acc'] = test_acc
                p_fix_dict["test_acc"] = test_acc
                if cfg.benchmark=='class_incremental':
                    acc_full, _ = task.test(model, device, full=True)
                    log_dict['test/acc_full'] = acc_full
                    p_fix_dict['test_acc_full'] = acc_full

                wandb.log(log_dict, step=global_step)                    
                pbar.set_postfix(**p_fix_dict)

            # intervention: redo
            if cfg.redo['enable'] and not (cfg.benchmark == 'warm_start' and i_iter == 0):
                redo_handle.apply(trainloader, optimizer)

            global_epoch += 1

        del trainloader
        torch.cuda.empty_cache()  # Free GPU memory
        gc.collect()  # Force garbage collection

    wandb.finish()

if __name__ == "__main__":
    from config import get_config
    cfg = get_config()
    main(cfg)
