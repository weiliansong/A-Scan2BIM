from turtle import pos
import torch
import torch.nn as nn
import os
import time
import numpy as np
import datetime
import argparse
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.building_order_enum import (
    BuildingCornerDataset,
    collate_fn_seq,
    get_pixel_features,
)
from models.corner_models import CornerEnum
from models.order_class_models import EdgeTransformer, EdgeTransformer2
from models.loss import CornerCriterion, EdgeCriterion
from models.corner_to_edge import prepare_edge_data
import utils.misc as utils
from utils.nn_utils import pos_encode_2d
import torch.nn.functional as F

# from infer_full import FloorHEAT

# for debugging NaNs
# torch.autograd.set_detect_anomaly(True)


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--lr_drop", default=50, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="./ckpts_order/debug",
        help="path where to save, empty for no saving",
    )

    # my own
    parser.add_argument("--test_idx", type=int, default=0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--rand_aug", action="store_true", default=True)
    parser.add_argument("--last_first", type=bool, default=False)
    parser.add_argument("--lock_pos", type=str, default="none")
    parser.add_argument("--normalize_by_seq", action="store_true")

    return parser


def train_one_epoch(
    image_size,
    edge_model,
    edge_criterion,
    data_loader,
    optimizer,
    writer,
    epoch,
    max_norm,
    args,
):
    # backbone.train()
    edge_model.train()
    edge_criterion.train()
    optimizer.zero_grad()

    acc_avg = 0
    loss_avg = 0

    pbar = tqdm(data_loader)
    for batch_i, data in enumerate(pbar):
        logits, loss, acc = run_model(data, edge_model, epoch, edge_criterion)

        loss = loss / args.grad_accum
        loss.backward()

        num_iter = epoch * len(data_loader) + batch_i
        writer.add_scalar("train/loss_mb", loss, num_iter)
        writer.add_scalar("train/acc_mb", acc, num_iter)
        pbar.set_description("Train Loss: %.3f Acc: %.3f" % (loss, acc))

        acc_avg += acc.item()
        loss_avg += loss.item()

        if ((batch_i + 1) % args.grad_accum == 0) or (
            (batch_i + 1) == len(data_loader)
        ):
            if max_norm > 0:
                # torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(edge_model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()

    acc_avg /= len(data_loader)
    loss_avg /= len(data_loader)
    writer.add_scalar("train/acc", acc_avg, epoch)
    writer.add_scalar("train/loss", loss_avg, epoch)

    print("Train loss: %.3f acc: %.3f" % (loss_avg, acc_avg))

    return -1


@torch.no_grad()
def evaluate(image_size, edge_model, edge_criterion, data_loader, writer, epoch, args):
    # backbone.train()
    edge_model.eval()
    edge_criterion.eval()

    loss_total = 0
    acc_total = 0

    pbar = tqdm(data_loader)
    for batch_i, data in enumerate(pbar):
        logits, loss, acc = run_model(data, edge_model, epoch, edge_criterion)
        pbar.set_description("Eval Loss: %.3f" % loss)

        loss_total += loss.item()
        acc_total += acc.item()

    loss_avg = loss_total / len(data_loader)
    acc_avg = acc_total / len(data_loader)

    print("Val loss: %.3f acc: %.3f\n" % (loss_avg, acc_avg))
    writer.add_scalar("eval/loss", loss_avg, epoch)
    writer.add_scalar("eval/acc", acc_avg, epoch)

    return loss_avg, acc_avg


def run_model(data, edge_model, epoch, edge_criterion):
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].cuda()

    # run the edge model
    assert (data["label"] >= 0).all() and (data["label"] <= 1).all()

    logits = edge_model(data)
    loss = edge_criterion(logits, data["label"])
    acc = (logits.argmax(-1) == data["label"]).float().mean()

    return logits, loss, acc


def main(args):
    DATAPATH = "./data/bim_dataset_big_v5/"
    REVIT_ROOT = "../../../revit_projects/"
    image_size = 512

    # prepare datasets
    train_dataset = BuildingCornerDataset(
        DATAPATH,
        REVIT_ROOT,
        phase="train",
        image_size=image_size,
        rand_aug=args.rand_aug,
        test_idx=args.test_idx,
        loss_type="class",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn_seq,
    )

    test_dataset = BuildingCornerDataset(
        DATAPATH,
        REVIT_ROOT,
        phase="valid",
        image_size=image_size,
        rand_aug=False,
        test_idx=args.test_idx,
        loss_type="class",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn_seq,
    )

    edge_model = EdgeTransformer(d_model=256)
    edge_model = edge_model.cuda()

    edge_criterion = nn.CrossEntropyLoss()

    edge_params = [p for p in edge_model.parameters()]

    all_params = edge_params  # + backbone_params
    optimizer = torch.optim.AdamW(
        all_params, lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume)
        edge_model.load_state_dict(ckpt["edge_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        lr_scheduler.step_size = args.lr_drop

        print(
            "Resume from ckpt file {}, starting from epoch {}".format(
                args.resume, ckpt["epoch"]
            )
        )
        start_epoch = ckpt["epoch"] + 1

    else:
        ckpt = torch.load("heat_checkpoints/%d/checkpoint_best.pth" % args.test_idx)

        replacements = [
            ["decoder_1", "module.transformer.relational_decoder"],
            ["decoder_2", "module.transformer.relational_decoder"],
        ]

        edge_model_dict = edge_model.state_dict()
        for key in edge_model_dict.keys():
            replaced = False
            for (old, new) in replacements:
                if old in key:
                    assert not replaced
                    new_key = key.replace(old, new)
                    edge_model_dict[key] = ckpt["edge_model"][new_key]
                    replaced = True
                    print(key)

        edge_model.load_state_dict(edge_model_dict)
        print("Resume from pre-trained checkpoints")

    n_edge_parameters = sum(p.numel() for p in edge_params if p.requires_grad)
    n_all_parameters = sum(p.numel() for p in all_params if p.requires_grad)
    print("number of trainable edge params:", n_edge_parameters)
    print("number of all trainable params:", n_all_parameters)

    print("Start training")
    start_time = time.time()

    output_dir = Path("%s/%d" % (args.output_dir, args.test_idx))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare summary writer
    writer = SummaryWriter(log_dir=output_dir)

    best_acc = 0
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: %d" % epoch)
        train_one_epoch(
            image_size,
            edge_model,
            edge_criterion,
            train_dataloader,
            optimizer,
            writer,
            epoch,
            args.clip_max_norm,
            args,
        )
        lr_scheduler.step()

        val_loss, val_acc = evaluate(
            image_size,
            edge_model,
            edge_criterion,
            test_dataloader,
            writer,
            epoch,
            args,
        )

        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc
        else:
            is_best = False

        if args.output_dir:
            checkpoint_paths = [output_dir / ("checkpoint_latest.pth")]
            checkpoint_paths.append(output_dir / ("checkpoint_%03d.pth" % epoch))
            if is_best:
                checkpoint_paths.append(output_dir / "checkpoint_best.pth")

            for checkpoint_path in checkpoint_paths:
                torch.save(
                    {
                        "edge_model": edge_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "val_acc": val_acc,
                    },
                    checkpoint_path,
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GeoVAE training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
