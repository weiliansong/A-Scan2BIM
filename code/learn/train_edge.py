import argparse
import datetime
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import my_utils
import utils.misc as utils
from datasets.building_corners_full import (
    BuildingCornerDataset,
    collate_fn_seq,
    get_pixel_features,
)
from models.corner_models import CornerEnum
from models.corner_to_edge import prepare_edge_data
from models.edge_full_models import EdgeEnum
from models.loss import CornerCriterion, EdgeCriterion
from models.unet import ResNetBackbone


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--lr_drop", default=50, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=100, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # * Loss coefficient/
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir",
        default="./ckpts_edge/debug",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--corner_model", default="unet", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    # my own
    parser.add_argument("--test_idx", type=int, default=0)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--deform_type", default="DETR_dense")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--pool_type", default="max")
    parser.add_argument("--revectorize", action="store_true")

    return parser


def train_one_epoch(
    backbone,
    edge_model,
    edge_criterion,
    data_loader,
    optimizer,
    epoch,
    max_norm,
    args,
):
    backbone.train()
    edge_model.train()
    edge_criterion.train()
    optimizer.zero_grad()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=100, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 40

    batch_i = 0
    for data in metric_logger.log_every(data_loader, print_freq, header):
        (
            s1_logits,
            s2_logits_hb,
            s2_logits_rel,
            s1_losses,
            s2_losses_e_hb,
            s2_losses_e_rel,
            s2_losses_w_hb,
            s2_losses_w_rel,
            s1_acc,
            s2_acc_e_hb,
            s2_acc_e_rel,
            s2_acc_w_hb,
            s2_acc_w_rel,
        ) = run_model(
            data,
            backbone,
            edge_model,
            epoch,
            edge_criterion,
            args,
        )

        # compute loss
        loss = (
            s1_losses
            + s2_losses_e_hb
            + s2_losses_e_rel
            + s2_losses_w_hb
            + s2_losses_w_rel
        )
        loss /= args.grad_accum
        loss.backward()

        # collect stats
        loss_dict = {
            "loss_e_s1": s1_losses,
            "loss_e_s2_hb": s2_losses_e_hb,
            "loss_e_s2_rel": s2_losses_e_rel,
            "loss_w_s2_hb": s2_losses_w_hb,
            "loss_w_s2_rel": s2_losses_w_rel,
            "edge_acc_s1": s1_acc,
            "edge_acc_e_s2_hb": s2_acc_e_hb,
            "edge_acc_e_s2_rel": s2_acc_e_rel,
            "edge_acc_w_s2_hb": s2_acc_w_hb,
            "edge_acc_w_s2_rel": s2_acc_w_rel,
        }
        loss_value = loss.item()
        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if ((batch_i + 1) % args.grad_accum == 0) or (
            (batch_i + 1) == len(data_loader)
        ):
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(edge_model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()

        batch_i += 1

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def run_model(
    data,
    backbone,
    edge_model,
    epoch,
    edge_criterion,
    args,
    per_class_acc=False,
):
    image = data["img"].cuda()
    image_feats, feat_mask, all_image_feats = backbone(image)

    edge_coords = data["edge_coords"].cuda()
    edge_mask = data["edge_coords_mask"].cuda()
    edge_lengths = data["edge_coords_lengths"].cuda()
    edge_labels = data["edge_labels"].cuda()
    width_labels = data["width_labels"].cuda()
    corner_nums = data["processed_corners_lengths"]

    # run the edge model
    max_candidates = torch.stack([corner_nums.max() * 3] * len(corner_nums), dim=0)

    (
        logits_s1,
        logits_edge_hb,
        logits_edge_rel,
        logits_width_hb,
        logits_width_rel,
        s2_ids,
        s2_edge_mask,
        s2_gt_values,
        ref_dict,
    ) = edge_model(
        image_feats,
        feat_mask,
        edge_coords,
        edge_mask,
        edge_labels,
        corner_nums,
        max_candidates,
    )

    # my_utils.vis_ref(image, edge_coords, ref_dict, s2_ids)

    (
        s1_losses,
        s1_acc,
        s2_losses_e_hb,
        s2_acc_e_hb,
        s2_losses_e_rel,
        s2_acc_e_rel,
        s2_losses_w_hb,
        s2_acc_w_hb,
        s2_losses_w_rel,
        s2_acc_w_rel,
    ) = edge_criterion(
        logits_s1,
        logits_edge_hb,
        logits_edge_rel,
        logits_width_hb,
        logits_width_rel,
        s2_ids,
        s2_edge_mask,
        edge_labels,
        width_labels,
        edge_lengths,
        edge_mask,
        s2_gt_values,
    )

    # s1_losses = edge_criterion(input=logits_s1, target=edge_labels)

    # if not per_class_acc:
    #     s1_acc = (logits_s1.argmax(1) == edge_labels).float().mean()
    # else:
    #     edge_preds = logits_s1.argmax(1)
    #     pos_acc = (edge_preds == edge_labels)[edge_labels == 1].float().mean()
    #     neg_acc = (edge_preds == edge_labels)[edge_labels == 0].float().mean()
    #     s1_acc = (pos_acc + neg_acc) / 2

    # s2_losses_hb = s2_losses_rel = 0
    # s2_acc_hb = s2_acc_rel = 0

    return (
        logits_s1,
        logits_edge_hb,
        logits_edge_rel,
        s1_losses,
        s2_losses_e_hb,
        s2_losses_e_rel,
        s2_losses_w_hb,
        s2_losses_w_rel,
        s1_acc,
        s2_acc_e_hb,
        s2_acc_e_rel,
        s2_acc_w_hb,
        s2_acc_w_rel,
    )


@torch.no_grad()
def evaluate(
    backbone,
    edge_model,
    edge_criterion,
    data_loader,
    epoch,
    args,
):
    backbone.eval()
    edge_model.eval()
    edge_criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    for data in metric_logger.log_every(data_loader, 10, header):
        (
            s1_logits,
            s2_logits_hb,
            s2_logits_rel,
            s1_losses,
            s2_losses_e_hb,
            s2_losses_e_rel,
            s2_losses_w_hb,
            s2_losses_w_rel,
            s1_acc,
            s2_acc_e_hb,
            s2_acc_e_rel,
            s2_acc_w_hb,
            s2_acc_w_rel,
        ) = run_model(
            data,
            backbone,
            edge_model,
            epoch,
            edge_criterion,
            args,
            per_class_acc=True,
        )

        loss_dict = {
            "loss_e_s1": s1_losses,
            "loss_e_s2_hb": s2_losses_e_hb,
            "loss_e_s2_rel": s2_losses_e_rel,
            "loss_w_s2_hb": s2_losses_w_hb,
            "loss_w_s2_rel": s2_losses_w_rel,
            "edge_acc_s1": s1_acc,
            "edge_acc_e_s2_hb": s2_acc_e_hb,
            "edge_acc_e_s2_rel": s2_acc_e_rel,
            "edge_acc_w_s2_hb": s2_acc_w_hb,
            "edge_acc_w_s2_rel": s2_acc_w_rel,
        }

        loss = (
            s1_losses
            + s2_losses_e_hb
            + s2_losses_e_rel
            + s2_losses_w_hb
            + s2_losses_w_rel
        )
        loss_value = loss.item()
        metric_logger.update(loss=loss_value, **loss_dict)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    DATAPATH = "./data/bim_dataset_big_v5"
    train_dataset = BuildingCornerDataset(
        DATAPATH,
        phase="train",
        rand_aug=True,
        test_idx=args.test_idx,
        multiplier=100,
        batch_size=args.batch_size,
        threshold=args.threshold,
        task="edge_class",
        revectorize=args.revectorize,
    )
    test_dataset = BuildingCornerDataset(
        DATAPATH,
        phase="valid",
        rand_aug=False,
        test_idx=args.test_idx,
        multiplier=1,
        batch_size=args.batch_size,
        threshold=args.threshold,
        task="edge_class",
        revectorize=args.revectorize,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn_seq,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn_seq,
    )

    backbone = ResNetBackbone()
    strides = backbone.strides
    num_channels = backbone.num_channels

    # backbone = nn.DataParallel(backbone)
    backbone = backbone.cuda()

    edge_model = EdgeEnum(
        input_dim=128,
        hidden_dim=256,
        num_feature_levels=4,
        backbone_strides=strides,
        backbone_num_channels=num_channels,
        deform_type=args.deform_type,
        num_samples=args.num_samples,
        pool_type=args.pool_type,
    )
    # edge_model = nn.DataParallel(edge_model)
    edge_model = edge_model.cuda()

    edge_criterion = EdgeCriterion()
    # edge_criterion = nn.CrossEntropyLoss()

    backbone_params = [p for p in backbone.parameters()]
    edge_params = [p for p in edge_model.parameters()]

    all_params = edge_params + backbone_params
    optimizer = torch.optim.AdamW(
        all_params, lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    start_epoch = args.start_epoch

    if args.resume:
        ckpt = torch.load(args.resume)
        backbone.load_state_dict(ckpt["backbone"])
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
        # NOTE initialize our pretrained corner model and also from Jiacheng's checkpoint
        # corner_ckpt = torch.load('checkpoints_bim_corners/checkpoint_best.pth')
        ckpt = torch.load("heat_checkpoints/%d/checkpoint_best.pth" % args.test_idx)

        backbone_ckpt = {}
        for (key, value) in ckpt["backbone"].items():
            key = key.replace("module.", "")
            backbone_ckpt[key] = value
        backbone.load_state_dict(backbone_ckpt)

        # edge_model_ckpt = {}
        # for (key, value) in ckpt["edge_model"].items():
        #     key = key.replace("module.", "")
        #     edge_model_ckpt[key] = value
        # edge_model.load_state_dict(edge_model_ckpt)

        print("Resume from pre-trained checkpoints")

    n_backbone_parameters = sum(p.numel() for p in backbone_params if p.requires_grad)
    n_edge_parameters = sum(p.numel() for p in edge_params if p.requires_grad)
    n_all_parameters = sum(p.numel() for p in all_params if p.requires_grad)
    print("number of trainable backbone params:", n_backbone_parameters)
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
        train_stats = train_one_epoch(
            backbone,
            edge_model,
            edge_criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            args,
        )
        lr_scheduler.step()

        val_stats = evaluate(
            backbone,
            edge_model,
            edge_criterion,
            test_dataloader,
            epoch,
            args,
        )

        # val_acc = val_stats['corner_recall']
        # val_acc = val_stats["edge_acc_s1"]
        val_acc = (
            val_stats["edge_acc_s1"]
            + val_stats["edge_acc_e_s2_hb"]
            + val_stats["edge_acc_w_s2_hb"]
        ) / 3

        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc
        else:
            is_best = False

        # write out stats
        for key, value in train_stats.items():
            writer.add_scalar("train/%s" % key, value, epoch)
        for key, value in val_stats.items():
            writer.add_scalar("val/%s" % key, value, epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            if is_best:
                checkpoint_paths.append(output_dir / "checkpoint_best.pth")

            for checkpoint_path in checkpoint_paths:
                torch.save(
                    {
                        "backbone": backbone.state_dict(),
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
