"""
train_cli.py  —  CLI wrapper for single-GPU Attention-UNet training.

Accepts all hyperparameters as command-line arguments and delegates to the
original training logic in train.py.  train.py itself is NOT modified.

Usage (inside container, cwd is /app/main):
    python train_cli.py --data /data --output /output [OPTIONS]
"""

import argparse
import os
import sys
import time
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ── local imports (all relative to /app/main) ────────────────────────────────
from train   import load_data, DATASET, train, evaluate
from utils   import seeding, create_dir, epoch_time, EarlyStopping, load_checkpoint
from metrics import DiceCELoss
from model   import build_unet


# ── argument parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Single-GPU Attention-UNet training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data",   required=True,
                   help="Dataset root. Must contain images/ and masks/ sub-dirs with *.png files.")
    p.add_argument("--output", default="/output",
                   help="Directory where checkpoint.pth will be saved.")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data to use for validation.")

    # ── model ────────────────────────────────────────────────────────────────
    p.add_argument("--num-classes", type=int, default=5,
                   help="Number of segmentation classes.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint if one exists in --output.")

    # ── image ────────────────────────────────────────────────────────────────
    p.add_argument("--img-w", type=int, default=512, help="Input image width.")
    p.add_argument("--img-h", type=int, default=512, help="Input image height.")

    # ── training ─────────────────────────────────────────────────────────────
    p.add_argument("--epochs",    type=int,   default=1000)
    p.add_argument("--batch",     type=int,   default=4)
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--wd",        type=float, default=1e-4,  help="AdamW weight decay.")
    p.add_argument("--patience",  type=int,   default=100,   help="Early-stopping patience.")
    p.add_argument("--workers",   type=int,   default=4,     help="DataLoader workers.")
    p.add_argument("--seed",      type=int,   default=42)

    # ── loss ─────────────────────────────────────────────────────────────────
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--ce-weight",   type=float, default=0.5)

    # ── colormap (flat list, 3 values per class) ──────────────────────────────
    p.add_argument(
        "--colormap",
        nargs="+", type=int,
        default=[0,0,0, 255,0,0, 0,0,255, 0,255,0, 255,255,255],
        help=(
            "Flat RGB colormap: 3 integers per class in class order. "
            "Default is 5-class AV: Background Artery Vein Junction Uncertain. "
            "Example for 3 classes: --colormap 0 0 0  255 0 0  0 0 255"
        ),
    )

    return p.parse_args()


def build_colormap(flat):
    if len(flat) % 3 != 0:
        raise ValueError(f"--colormap must have a multiple of 3 values, got {len(flat)}.")
    return [[flat[i], flat[i+1], flat[i+2]] for i in range(0, len(flat), 3)]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    seeding(args.seed)

    # Resolve paths
    dataset_path    = os.path.abspath(args.data)
    output_dir      = os.path.abspath(args.output)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    create_dir(output_dir)

    colormap    = build_colormap(args.colormap)
    num_classes = args.num_classes
    if len(colormap) != num_classes:
        print(f"[WARN] --colormap has {len(colormap)} entries but --num-classes={num_classes}. "
              f"Using colormap length as num_classes.")
        num_classes = len(colormap)

    size = (args.img_w, args.img_h)

    print("=" * 60)
    print("  Attention-UNet  |  Single-GPU Training")
    print("=" * 60)
    print(f"  Dataset     : {dataset_path}")
    print(f"  Output      : {output_dir}")
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Image size  : {size}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Num classes : {num_classes}")
    print(f"  Colormap    : {colormap}")
    print("=" * 60)

    # ── dataset ──────────────────────────────────────────────────────────────
    # Validate dataset directory structure before calling load_data
    imgs_dir  = os.path.join(dataset_path, "images")
    masks_dir = os.path.join(dataset_path, "masks")
    if not os.path.isdir(imgs_dir) or not os.path.isdir(masks_dir):
        print(f"\n[ERROR] Dataset directory structure is incorrect.")
        print(f"        Expected:")
        print(f"          {dataset_path}/")
        print(f"            images/   *.png")
        print(f"            masks/    *.png")
        print(f"        Found dirs: {os.listdir(dataset_path) if os.path.isdir(dataset_path) else 'PATH DOES NOT EXIST'}")
        sys.exit(1)

    from glob import glob
    n_imgs = len(glob(os.path.join(imgs_dir,  "*.png")))
    n_masks = len(glob(os.path.join(masks_dir, "*.png")))
    if n_imgs == 0:
        print(f"\n[ERROR] No *.png files found in {imgs_dir}")
        print(f"        Check that your images are PNG format and the bind mount is correct.")
        print(f"        Contents of {imgs_dir}: {os.listdir(imgs_dir)[:10]}")
        sys.exit(1)
    if n_imgs != n_masks:
        print(f"\n[ERROR] Image/mask count mismatch: {n_imgs} images vs {n_masks} masks.")
        sys.exit(1)
    if int(args.val_split * n_imgs) < 1:
        min_split = 1.0 / n_imgs
        print(f"\n[ERROR] --val-split {args.val_split} produces 0 validation samples for {n_imgs} images.")
        print(f"        Use --val-split >= {min_split:.3f}  (at least 1 sample)")
        sys.exit(1)

    (train_x, train_y), (valid_x, valid_y) = load_data(dataset_path, split=args.val_split)
    print(f"Dataset — Train: {len(train_x)}  |  Val: {len(valid_x)}")

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(p=0.1, max_holes=3, max_height=32, max_width=32),
    ], is_check_shapes=False)

    train_dataset = DATASET(train_x, train_y, size, colormap, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, colormap, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # ── model ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_unet(num_classes=num_classes).to(device)
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
                       optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
    early_stop   = EarlyStopping(patience=args.patience, verbose=True)

    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    loss_fn = DiceCELoss(num_classes=num_classes,
                         dice_weight=args.dice_weight,
                         ce_weight=args.ce_weight,
                         ignore_index=-1)

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        mins, secs = epoch_time(t0, time.time())

        print(f"[{epoch:04}/{args.epochs}] {mins}m{secs}s  "
              f"train={train_loss:.4f}  val={valid_loss:.4f}")

        scheduler.step(valid_loss)
        early_stop(valid_loss, model, optimizer, epoch, checkpoint_path)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nDone. Best checkpoint → {checkpoint_path}")


if __name__ == "__main__":
    main()
