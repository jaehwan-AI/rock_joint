import argparse
import os
from importlib import import_module

import matplotlib
import matplotlib.pyplot as plt
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import ROCKDataset
from dataset.transforms import test_transforms, train_transforms
from train_utils.losses import create_criterion
from train_utils.optimizers import create_optimizer, get_lr
from train_utils.schedulers import create_scheduler
from train_utils.utils import grid_image
from trainer import (createDirectory, increment_path, seed_everything,
                     str2bool, train_step, two_loss_step)

matplotlib.use('Agg')


def main(args):
    seed_everything(args.random_seed)

    # save path
    save_dir = increment_path(os.path.join(args.save_path, args.model, args.optimizer, args.data_path.split('/')[-1]))
    createDirectory(save_dir)

    # device
    if args.device == 'gpu0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
    elif args.device == 'gpu1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    # dataset & dataloader
    train_transform = train_transforms()
    test_transform = test_transforms()

    train_dataset = ROCKDataset(
        args, root=args.data_path, mode='train', transforms=train_transform)
    val_dataset = ROCKDataset(
        args, root=args.data_path, mode='val', transforms=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # model
    model_module = getattr(import_module("models.model"), args.model)
    model = model_module(pretrained=args.pretrained)
    model = model.to(device)
    if args.wandb:
        wandb.watch(model)

    # loss & optimizer & scheduler
    criterion = create_criterion(args)
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer)

    # start training
    best_val_mIoU = 0
    step = 0
    for epoch in range(args.epochs):
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0
        total_miou, total_val_miou = 0, 0

        print('training...')
        training = True
        cnt = 0
        tqdm_dataset = tqdm(enumerate(train_loader))
        for batch, batch_item in tqdm_dataset:
            if args.criterion == 'two_loss':
                loss, acc, acc_cls, mIoU, fwavacc, IoU = two_loss_step(
                    training, batch_item, model, optimizer, device, args.beta)
            else:
                loss, acc, acc_cls, mIoU, fwavacc, IoU = train_step(
                    training, batch_item, model, optimizer, criterion, device, args.beta)

            total_loss += loss
            total_miou += mIoU
            total_acc += acc

            current_lr = get_lr(optimizer)
            tqdm_dataset.set_postfix({
                'Epoch': epoch+1,
                'lr': '{:06f}'.format(current_lr),
                'batch Loss': '{:06f}'.format(loss.item()),
                'Mean Loss': '{:06f}'.format(total_loss/(batch+1)),
                'Mean IoU': '{:06f}'.format(total_miou/(batch+1)),
                'Mean Acc': '{:06f}'.format(total_acc/(batch+1))
            })

            # debug mode
            cnt += 1
            if args.debug:
                if cnt % 10 == 0:
                    print("TRAINING DEBUG MODE, BREAK!!")
                    break

        # wandb log
        if args.wandb:
            wandb.log(
                {
                    "Train/Train loss": total_loss/(batch+1),
                    "Train/Train mIoU": total_miou/(batch+1),
                    "Train/Train Acc": total_acc/(batch+1),
                    "Train/Learning rate": current_lr
                },
                step=step
            )

        print('Validation...')
        training = False
        cnt_ = 0
        tqdm_dataset = tqdm(enumerate(val_loader))
        for batch, batch_item in tqdm_dataset:
            figrue = None
            if args.criterion == 'two_loss':
                imgs, masks, preds, loss, acc, acc_cls, mIoU, fwavacc, IoU = two_loss_step(
                    training, batch_item, model, optimizer, device, args.beta)
            else:
                imgs, masks, preds, loss, acc, acc_cls, mIoU, fwavacc, IoU = train_step(
                    training, batch_item, model, optimizer, criterion, device, args.beta)

            total_val_loss += loss
            total_val_miou += mIoU
            total_val_acc += acc

            tqdm_dataset.set_postfix({
                'Epoch': epoch+1,
                'batch Loss': '{:06f}'.format(loss.item()),
                'Mean Loss': '{:06f}'.format(total_val_loss/(batch+1)),
                'Mean IoU': '{:06f}'.format(total_val_miou/(batch+1)),
                'Mean Acc': '{:06f}'.format(total_val_acc/(batch+1))
            })

            if figrue is None:
                figure = grid_image(
                    imgs, masks, preds
                )

            # wandb log
            if args.wandb:
                wandb.log(
                    {
                        "Media/Predict Images": figure
                    },
                    step=step
                )
                step += 1
                plt.close(figure)

            # debug mode
            cnt_ += 1
            if args.debug:
                if cnt_ % 10 == 0:
                    print("VALIDATION DEBUG MODE, BREAK!!")
                    break

        # wandb log
        if args.wandb:
            wandb.log(
                {
                    "Valid/Valid loss": total_val_loss/(batch+1),
                    "Valid/Valid mIoU": total_val_miou/(batch+1),
                    "Valid/Valid Acc": total_val_acc/(batch+1),
                },
                step=step
            )

        # scheduler
        scheduler.step()

        if best_val_mIoU < (total_val_miou/(batch+1)):
            best_val_mIoU = total_val_miou/(batch+1)
            save_file = args.model+'_'+str(epoch)+'_'+str(best_val_mIoU)+'.pt'
            save_path = os.path.join(save_dir, save_file)
            torch.save(model.state_dict(), save_path)
            print('Saved model weight!! {}'.format(save_path))

    if args.wandb:
        wandb.alert(
            title="Finish!",
            text="The training is over~",
            level=wandb.AlertLevel.INFO
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_seed", type=int, default=2022, help="random seed (default: 2022)")
    parser.add_argument(
        "--device", type=str, default="gpu0", help="device type (default: gpu0)")

    # dataset
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train (default: 100)")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="input batch size for training (default: 4)")
    parser.add_argument(
        "--img_size", type=int, default=1024, help="input image & target size")

    # model
    parser.add_argument(
        "--model", type=str, default="DeepLabV3_Res101", help="model type")
    parser.add_argument(
        "--pretrained", type=str2bool, default="True", help="use pretrained weight")

    # criterion
    parser.add_argument(
        "--criterion", type=str, default="focal", help="criterion type")

    # optimizer
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer type (default: adam)")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="weight decay (default: 1e-5)")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum (default: 0.9)")
    parser.add_argument(
        "--amsgrad", action="store_true", help="amsgrad for adam")

    # scheduler
    parser.add_argument(
        "--scheduler", type=str, default="lambda", help="scheduler type (default: lambda)",)
    parser.add_argument(
        "--poly_exp", type=float, default=1.0, help="polynomial LR exponent (default: 1.0)",)
    parser.add_argument(
        "--max_lr", type=float, default=1e-3, help="OneCycleLR max lr (default: 1e-3)")
    parser.add_argument(
        "--t_up", type=int, default=5, help="warm up epoch (default: 5)")
    parser.add_argument(
        "--iter", type=int, default=10, help="iteration rate (default: 10)")

    # Container environment
    parser.add_argument(
        "--data_path", type=str, default=os.path.abspath('./data'), help="dataset path")
    parser.add_argument(
        "--save_path", type=str, default=os.path.abspath('./save_weight/'), help="model save dir path")

    # wandb
    parser.add_argument(
        "--wandb", action="store_true", help="wandb implement or not")
    parser.add_argument(
        "--entity", type=str, default="jaehwan", help="wandb entity name",)
    parser.add_argument(
        "--project", type=str, default="Joint", help="wandb project name")
    parser.add_argument(
        "--run_name", type=str, default="test", help="wandb run name")

    # debug
    parser.add_argument(
        "--debug", action="store_true", help="debug mode implement or not")

    args = parser.parse_args()
    print(args)

    # wandb init
    if args.wandb:
        wandb.init(entity=args.entity, project=args.project)
        wandb.run.name = args.run_name

    main(args)