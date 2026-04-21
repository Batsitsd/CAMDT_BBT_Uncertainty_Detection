import torch

def mc_dropout_inference(model, x, T=20):
    model.train()  # keep dropout active

    outputs = []
    for _ in range(T):
        outputs.append(model(x))

    outputs = torch.stack(outputs)

    mean = outputs.mean(dim=0)
    variance = outputs.var(dim=0)

    return mean, variance