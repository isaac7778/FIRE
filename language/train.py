"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train_warm_start.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train_warm_start.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_warm_start.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_warm_start.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from datetime import datetime

from interventions.snp import shrink_and_perturb
from interventions.fire import fire

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = 2000
log_interval = 1
eval_iters = 200
save_checkpoint = True # if True, always save a checkpoint after each chunk
save_checkpoint_periodically = True
# data
seed = 1337
gradient_accumulation_steps = 8 # used to simulate larger batch sizes
batch_size = 12*5 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_ratio = 0.1 # ratio of warmup period at each chunk
max_warmup_iters = 2000 # use smaller value between 10% of chunk_iters and max_warmup_iters
# chunk_lr_decay_iters = 20000
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'float16' # use float16 by default due to instability of bfloat16
compile = True # use PyTorch 2.0 to compile the model to be faster

# intervention configs
snp_shrink_coef = 0.8
snp_init_load_path = ''

fire_iteration = 5
# wandb logging
warm_start_load_path = ''
reset_optimizer = True
method_type = 'vanilla' # vanilla, full_reset, snp, fire
wandb_log = True
comment = ""

# continual pre-training configs
c0_dataset = 'openwebtext'
c0_subset_ratio = 0.1
c0_data_replay_ratio = 300

c1_dataset = 'openwebtext'
c1_subset_ratio = 1.0
c1_data_replay_ratio = 2
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
chunk_config = [
    {'ratio': c0_subset_ratio, 'data_replay_ratio': c0_data_replay_ratio, 'dataset': c0_dataset},
    {'ratio': c1_subset_ratio, 'data_replay_ratio': c1_data_replay_ratio, 'dataset': c1_dataset},
]

# get run name
wandb_run_name = method_type
if method_type == 'snp':
    wandb_run_name += str(snp_shrink_coef)
if method_type == 'fire':
    wandb_run_name += str(fire_iteration)

wandb_run_name += "_seed"+str(seed)

if not reset_optimizer:
    wandb_run_name += "_keep_optim"

wandb_run_name += comment

if method_type == 'full_reset':
    wandb_run_name += f"_{chunk_config[1]['dataset']}_{chunk_config[1]['ratio']}_{chunk_config[1]['data_replay_ratio']}"
else:
    wandb_run_name += f"_{chunk_config[0]['dataset']}_{chunk_config[0]['ratio']}_{chunk_config[0]['data_replay_ratio']}_{chunk_config[1]['dataset']}_{chunk_config[1]['ratio']}_{chunk_config[1]['data_replay_ratio']}"
out_dir = os.path.join("output", wandb_run_name)

if method_type not in ("vanilla", "full_reset", "snp", "fire"):
    raise ValueError("wrong method_type")
# -----------------------------------------------------------------------------


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
def get_batch(dataset_name, split, ratio=None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if dataset_name == "wiki_owt":
        if random.random() < 0.1/9.1:
            dataset_name = "wikitext"
        else:
            dataset_name = "openwebtext"

    data_dir = os.path.join('data', dataset_name)
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    if ratio is not None: # use subset of the dataset
        end_index = int(len(data) * ratio) - block_size
    else:
        end_index = len(data) - block_size
    ix = torch.randint(end_index, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def get_dataset_len(dataset_name):
    if dataset_name == "wiki_owt":
        owt_len = len(np.memmap(os.path.join("data", "openwebtext", 'train.bin'), dtype=np.uint16, mode='r'))
        wiki_len = len(np.memmap(os.path.join("data", "wikitext", 'train.bin'), dtype=np.uint16, mode='r'))
        return owt_len + wiki_len
    else:
        data_dir = os.path.join('data', dataset_name)
        return len(np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r'))


iter_num = 0

# load_model
def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model, checkpoint

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

if method_type not in ('vanilla', 'full_reset'):
    assert len(warm_start_load_path) > 0, "interventions should start from pretrained weights"

if len(warm_start_load_path) > 0:
    print(f"Resuming training from {warm_start_load_path}")
    # resume training from a checkpoint.
    model, w_ckpt = load_model(warm_start_load_path)

    if method_type == 'snp':
        init_model, _ = load_model(snp_init_load_path)
        shrink_and_perturb(model, init_model, shrink_coef=snp_shrink_coef)
    elif method_type == 'fire':
        fire(model, iteration=fire_iteration)
else:
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    # if meta_vocab_size is None:
    #     print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    # model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    model_args['vocab_size'] = 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if len(warm_start_load_path) > 0 and not reset_optimizer:
    print("Load optimizer from checkpoint")
    optimizer.load_state_dict(w_ckpt['optimizer'])
else:
    w_ckpt = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(dataset_name, train_ratio=None):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(dataset_name, split, ratio=train_ratio if (split == 'train' and train_ratio is not None) else None)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, wi, ldi):
    # 1) linear warmup for warmup_iters steps
    if it < wi:
        return learning_rate * (it + 1) / (wi + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > ldi:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - wi) / (ldi - wi)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=f"warm_start_nanoGPT", name=wandb_run_name, config=config)

print("chunk_config: ", chunk_config)
global_iter_num = 0
global_logging_step = 0
global_learned_token = 0
running_mfu = -1.0
raw_model = model.module if ddp else model # unwrap DDP container if needed
skip_first_chunk = (method_type == 'full_reset' or len(warm_start_load_path) > 0)

if save_checkpoint:
    # save initial model for snp
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'config': config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, f'init_ckpt.pt'))

for chunk_index, chunk_info in enumerate(chunk_config):
    chunk_dataset_name = chunk_info['dataset']
    train_data_len = get_dataset_len(chunk_dataset_name)
    print(f"Current dataset: {chunk_dataset_name}, total length: {train_data_len}")

    chunk_ratio = chunk_info['ratio']
    chunk_num_tokens = int(chunk_info['data_replay_ratio'] * train_data_len * chunk_ratio)
    if chunk_num_tokens <= 0:
        continue
    chunk_num_iters = chunk_num_tokens // tokens_per_iter
    chunk_warmup_iters = min(int(warmup_ratio * chunk_num_iters), max_warmup_iters)
    chunk_lr_decay_iters = chunk_num_iters
    chunk_eval_interval = min(eval_interval, chunk_num_iters//20)
    # apply interventions
    skip_this_chunk = chunk_index == 0 and skip_first_chunk

    # reset optimizer
    if reset_optimizer:
        optimizer.state.clear()

    # training loop
    X, Y = get_batch(chunk_dataset_name, 'train', ratio=chunk_ratio) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    pbar = tqdm(total=chunk_num_iters)
    best_val_loss = 1e9
    while True:
        if not skip_this_chunk:
            # determine and set the learning rate for this iteration
            lr = get_lr(local_iter_num, wi=chunk_warmup_iters, ldi=chunk_lr_decay_iters) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if local_iter_num % chunk_eval_interval == 0 and master_process:
                losses = estimate_loss(chunk_dataset_name, train_ratio=chunk_ratio)
                # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if wandb_log:
                    wandb.log({
                        "global_learned_token": global_learned_token,
                        "chunk_index": chunk_index,
                        "global_iter": global_iter_num,
                        "local_iter": local_iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    }, step=global_logging_step)
                    print(f"logged to wandb, val loss: {losses['val']}")

                if save_checkpoint_periodically:
                    # save checkpoint every *5 interval
                    if local_iter_num % (chunk_eval_interval*5) == 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            "chunk_index": chunk_index,
                            "global_iter": global_iter_num,
                            "local_iter": local_iter_num,
                            "val/loss": losses['val'],
                            'config': config,
                        }
                        print(f"saving checkpoint to {out_dir}")
                        torch.save(checkpoint, os.path.join(out_dir, f'chunk{chunk_index}_iter{global_iter_num}_ckpt.pt'))

                if best_val_loss > losses['val'] and save_checkpoint: # save best ckpt
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        "chunk_index": chunk_index,
                        "global_iter": global_iter_num,
                        "local_iter": local_iter_num,
                        "val/loss": losses['val'],
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, f'best_chunk{chunk_index}_ckpt.pt'))
                    best_val_loss = losses['val']

                global_logging_step += 1

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch(chunk_dataset_name, 'train', ratio=chunk_ratio)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if global_iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if global_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                pbar.set_description(f"chunk {chunk_index}/{len(chunk_config)-1} iter {local_iter_num}/{chunk_num_iters-1}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        global_iter_num += 1
        global_learned_token += tokens_per_iter
        local_iter_num += 1
        pbar.update(1)

        # termination conditions
        if local_iter_num > chunk_num_iters:
            break

    if save_checkpoint:
        if global_iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                "chunk_index": chunk_index,
                "global_iter": global_iter_num,
                "local_iter": local_iter_num,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'chunk{chunk_index}_ckpt.pt'))
    pbar.close()

if ddp:
    destroy_process_group()
