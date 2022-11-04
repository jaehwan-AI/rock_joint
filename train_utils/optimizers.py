import torch.optim as optim


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def create_optimizer(args, model):
    """
    define optimizer
    """
    param_groups = model.parameters()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=False,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    else:
        raise ValueError("Not a valid optimizer")
    
    return optimizer
