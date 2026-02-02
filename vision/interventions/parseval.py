def parseval_loss(model, scale, effective=False):
    total_loss = 0
    for name, param in model.named_parameters():
        if model.is_norm_layer_parameter(name):
            if 'weight' in name:
                if effective: scaler_vector = param/torch.norm(param)
                else: scaler_vector = param
                total_loss += parseval_loss_scaler(param, scale)
        else:
            if 'weight' in name:
                weight_matrix = param.reshape(param.shape[0], -1)
                if effective: weight_matrix = weight_matrix/torch.norm(weight_matrix, dim=1, keepdim=True)
                total_loss += parseval_loss_matrix(weight_matrix, scale)
    return total_loss

def parseval_loss_matrix(matrix, scale):
    return torch.norm(
        torch.matmul(matrix, matrix.t()) - scale * torch.eye(matrix.shape[0], device=matrix.device),
        p='fro') ** 2

def parseval_loss_scaler(scaler, scale):
    return (scaler**2 - scale).norm() ** 2