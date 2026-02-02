import copy
# from einops import rearrange
import torch
from torch import nn
from torch.utils.data import DataLoader


def dash(
    model: nn.Module, dash_alpha: float, dash_lambda: float,
    dataloader: DataLoader, criterion: nn.Module, device: str
):
    # Initialize
    temp_model = copy.deepcopy(model)
    all_grad_lst = []
    all_grad_bias_lst = []
    for m in temp_model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            all_grad_lst.append(torch.zeros_like(m.weight))
            if m.bias != None and m.bias.requires_grad:
                all_grad_bias_lst.append(torch.zeros_like(m.bias))

    # Gather gradient
    for i, batches in enumerate(dataloader):
        x_batch, y_batch = batches[0], batches[1]
        temp_model.zero_grad()
        y_hat = temp_model(x_batch.to(device))
        loss = criterion(y_hat, y_batch.to(device))
        loss.backward()

        step = 0
        bias_step = 0
        for m in temp_model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                all_grad_lst[step] = (1 - dash_alpha) * all_grad_lst[step] + dash_alpha * (
                    -m.weight.grad.detach())
                if m.bias != None and m.bias.grad is not None:
                    all_grad_bias_lst[bias_step] = (1 - dash_alpha) * all_grad_bias_lst[
                        bias_step] + dash_alpha * (-m.bias.grad.detach())
                    bias_step += 1
                step += 1
        torch.cuda.empty_cache()

    # Shrink
    with torch.no_grad():
        step = 0
        bias_step = 0

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # grad = rearrange(all_grad_lst[step], 'o i h w -> o i (h w)')
                # param_r = rearrange(m.weight, 'o i h w -> o i (h w)')
                o, i, h, w = all_grad_lst[step].size()
                grad = all_grad_lst[step].contiguous().view(o, i, -1)
                o, i, h, w = m.weight.size()
                param_r = m.weight.contiguous().view(o, i, -1)

                cos_sim = torch.cosine_similarity(grad, param_r, dim=-1)
                scale = torch.clamp(cos_sim, min=dash_lambda, max=1)
                param_r.mul_(scale[:, :, None])
                step += 1

                if m.bias is not None and m.bias.requires_grad:
                    cos_sim_bias = torch.cosine_similarity(all_grad_bias_lst[bias_step].reshape(1, -1),
                                                           m.bias.reshape(1, -1))
                    scale_bias = torch.clamp(cos_sim_bias, min=dash_lambda, max=1)
                    m.bias.mul_(scale_bias)
                    bias_step += 1

            elif isinstance(m, nn.Linear):
                cos_sim = torch.cosine_similarity(all_grad_lst[step], m.weight, dim=-1)
                scale = torch.clamp(cos_sim, min=dash_lambda, max=1)
                m.weight.mul_(scale[:, None])
                step += 1
                if m.bias is not None:
                    cos_sim_bias = torch.cosine_similarity(all_grad_bias_lst[bias_step].reshape(1, -1),
                                                           m.bias.reshape(1, -1))
                    scale_bias = torch.clamp(cos_sim_bias, min=dash_lambda, max=1)
                    m.bias.mul_(scale_bias)
                    bias_step += 1
