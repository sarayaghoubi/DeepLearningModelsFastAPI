import torch
from torch.nn import MSELoss, HuberLoss
from torchmetrics import R2Score


def evaluate(net, dataloader, device):
    net.eval()
    score = 0
    num_val_batches = len(dataloader)
    loss_score = MSELoss()
    huberscore = HuberLoss()
    # iterate over the validation set
    for v, l in dataloader:
        v = v.to(device=device, dtype=torch.float32)
        l = l.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            p = net(v)
            score += loss_score(l, p).item() + huberscore(l, p).item()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return score
    print(f'validation loss: {score / num_val_batches}')
    return score / num_val_batches


def test(net, dataloader, device):
    net.eval()
    score = 0
    num_val_batches = len(dataloader)
    r2score = R2Score()
    # iterate over the validation set
    for v, l in dataloader:
        v = v.to(device=device, dtype=torch.float32)
        l = l.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            p = net(v)
            sc = r2score(p, l)
            score += sc
            print(sc)
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return score
    print(f'validation loss: {score / num_val_batches}')
    return score / num_val_batches
