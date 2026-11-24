"""Construct the CIFAR-paper candidate optimizer and epoch-step schedule."""
import torch


def build_optimizer(model, config):
    name = str(config.get("optimizer", "SGD")).lower()
    if name == "adadelta":
        return torch.optim.Adadelta(model.parameters(),
            lr=float(config["learning_rate"]),
            rho=float(config.get("rho", 0.9)), eps=float(config.get("eps", 1e-6)),
            weight_decay=float(config.get("weight_decay", 0.0)))
    if name != "sgd":
        raise ValueError(f"Unsupported optimizer candidate: {name}")
    return torch.optim.SGD(model.parameters(),
        lr=float(config["learning_rate"]), momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]))


def build_scheduler(optimizer, config):
    schedule = config["lr_schedule"]
    if schedule.get("enabled", True) is False:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(schedule["milestones"]), gamma=float(schedule["gamma"]))
