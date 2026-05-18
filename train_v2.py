import argparse, sys, time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from unet_v2         import UNetV2, SegLossV2
from data_loader_v2  import (get_train_val_loaders,
                              VESSEL_TO_ID)


def dice_coeff(prob, target, smooth=1.0):
    pred  = (prob > 0.5).float().reshape(-1)
    t     = target.float().reshape(-1)
    inter = (pred * t).sum().item()
    return (2.0 * inter + smooth) / (pred.sum().item() + t.sum().item() + smooth)


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, v, n=1):
        self.val = v; self.sum += v * n; self.count += n
        self.avg = self.sum / self.count


def get_loss_keys(use_reid=False):
    keys = ["total", "bce_loss", "dice_loss", "cldice_loss", "dice_coeff"]
    if use_reid:
        keys.append("reid_loss")
    return keys

def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, epoch=0, total_epochs=0, accum_steps=1,
                    use_reid=False):
    model.train()

    loss_keys = get_loss_keys(use_reid)
    meters = {k: AverageMeter() for k in loss_keys}
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"  Train [{epoch:3d}/{total_epochs}]",
                leave=False, dynamic_ncols=True, unit="batch")

    for step, (imgs, masks, vids, _) in enumerate(pbar):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        vids  = vids.to(device, non_blocking=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                if use_reid:
                    seg_logits, reid_dict = model(imgs, vids, return_reid=True)
                    loss_dict = criterion(seg_logits, masks,
                                          reid_dict=reid_dict, vessel_ids=vids)
                else:
                    seg_logits = model(imgs, vids)
                    loss_dict  = criterion(seg_logits, masks)
        else:
            if use_reid:
                seg_logits, reid_dict = model(imgs, vids, return_reid=True)
                loss_dict = criterion(seg_logits, masks,
                                      reid_dict=reid_dict, vessel_ids=vids)
            else:
                seg_logits = model(imgs, vids)
                loss_dict  = criterion(seg_logits, masks)

        total_loss = loss_dict["total"] / accum_steps

        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        bs = imgs.size(0)
        with torch.no_grad():
            d = dice_coeff(torch.sigmoid(seg_logits.detach()[:, 1]), masks)

        meters["total"].update(loss_dict["total"].item(), bs)
        meters["bce_loss"].update(loss_dict["bce_loss"].item(), bs)
        meters["dice_loss"].update(loss_dict["dice_loss"].item(), bs)
        meters["cldice_loss"].update(loss_dict["cldice_loss"].item(), bs)
        meters["dice_coeff"].update(d, bs)
        if "reid_loss" in loss_dict and "reid_loss" in meters:
            meters["reid_loss"].update(loss_dict["reid_loss"].item(), bs)

        pbar.set_postfix({"loss": f"{meters['total'].avg:.4f}",
                          "dice": f"{meters['dice_coeff'].avg:.4f}"})

    pbar.close()
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate(model, loader, criterion, device,
             epoch=0, total_epochs=0, use_amp=False, use_reid=False):
    model.eval()
    loss_keys = get_loss_keys(use_reid)
    meters = {k: AverageMeter() for k in loss_keys}

    pbar = tqdm(loader, desc=f"  Val   [{epoch:3d}/{total_epochs}]",
                leave=False, dynamic_ncols=True, unit="batch")

    for imgs, masks, vids, _ in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        vids  = vids.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_reid:
                seg_logits, reid_dict = model(imgs, vids, return_reid=True)
                loss_dict = criterion(seg_logits, masks,
                                      reid_dict=reid_dict, vessel_ids=vids)
            else:
                seg_logits = model(imgs, vids)
                loss_dict  = criterion(seg_logits, masks)

        d  = dice_coeff(torch.sigmoid(seg_logits[:, 1].float()), masks)
        bs = imgs.size(0)
        meters["total"].update(loss_dict["total"].item(), bs)
        meters["bce_loss"].update(loss_dict["bce_loss"].item(), bs)
        meters["dice_loss"].update(loss_dict["dice_loss"].item(), bs)
        meters["cldice_loss"].update(loss_dict["cldice_loss"].item(), bs)
        meters["dice_coeff"].update(d, bs)
        if "reid_loss" in loss_dict and "reid_loss" in meters:
            meters["reid_loss"].update(loss_dict["reid_loss"].item(), bs)

        pbar.set_postfix(loss=f"{meters['total'].avg:.4f}",
                         dice=f"{meters['dice_coeff'].avg:.4f}")

    pbar.close()
    return {k: m.avg for k, m in meters.items()}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    vessels = args.vessels.split(",")

    train_loader, val_loader = get_train_val_loaders(
        args.data, vessels=vessels,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch, num_workers=args.workers,
        cache_ram=not args.no_cache,
        merge_split=args.merge_split,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        artifact_prob=args.artifact_prob,
    )
    print(f"Train samples   : {len(train_loader.dataset)}")
    print(f"Val   samples   : {len(val_loader.dataset)}")
    print(f"Artifact prob   : {args.artifact_prob:.0%}")
    if len(train_loader.dataset) == 0:
        print("[ERROR] No training data."); return

    frozen = not args.unfreeze
    model = UNetV2(
        checkpoint=args.checkpoint, freeze=frozen,
        n_classes=2, n_vessels=len(VESSEL_TO_ID),
        use_semantic_prompt=args.use_semantic_prompt,
        use_sparse_gat=args.use_sparse_gat,
        use_reid=args.use_reid,
        n_prompt_tokens=args.n_prompt_tokens,
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        k_neighbors=args.k_neighbors,
        max_nodes=args.max_nodes,
        node_threshold=args.node_threshold,
        gnn_iters=args.gnn_iters,
        reid_embed_dim=args.reid_embed_dim,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters      : {total_p:,} total, {train_p:,} trainable")

    if args.resume:
        st = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ("model", "state_dict"):
            if isinstance(st, dict) and key in st and isinstance(st[key], dict):
                st = st[key]; break
        m, u = model.load_state_dict(st, strict=False)
        print(f"[Resume] {args.resume}")
        if m: print(f"[Resume] Missing: {len(m)} keys (e.g. {m[:3]})")
        if u: print(f"[Resume] Unexpected: {len(u)} keys (e.g. {u[:3]})")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    effective_lr = args.lr * (args.accum_steps if args.scale_lr else 1)
    backbone_lr  = effective_lr * args.backbone_lr_scale

    param_groups = []
    bb = [p for p in model.backbone_parameters() if p.requires_grad]
    if bb:
        param_groups.append({"params": bb, "lr": backbone_lr, "name": "backbone"})
    decoder_params = list(model.decoder_parameters())
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": effective_lr, "name": "decoder"})
    if args.use_reid:
        reid_params = list(model.reid_parameters())
        if reid_params:
            param_groups.append({"params": reid_params,
                                 "lr": effective_lr * 2.0, "name": "reid"})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    warmup_ep = max(0, min(args.warmup_epochs, args.epochs - 1))
    if warmup_ep > 0:
        warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                             total_iters=warmup_ep)
        cosine    = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_ep,
                                      eta_min=effective_lr * 1e-2)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_ep])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                      eta_min=effective_lr * 1e-2)

    criterion = SegLossV2(
        lambda_cldice=args.lambda_cldice,
        lambda_reid  =args.lambda_reid if args.use_reid else 0.0,
        cldice_iter  =args.cldice_iter,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and args.amp else None

    print(f"\nLR              : backbone={backbone_lr:.2e}  decoder={effective_lr:.2e}")
    print(f"AMP             : {'ON' if scaler else 'OFF'}")
    print(f"Loss            : WeightedBCE + WeightedDice + clDice(λ={args.lambda_cldice}, "
          f"iter={args.cldice_iter}) + InfoNCE(λ={args.lambda_reid})")

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_dice, log_rows = 0.0, []

    loss_keys = get_loss_keys(args.use_reid)
    csv_parts = ["epoch"]
    for prefix in ["tr", "va"]:
        for k in loss_keys:
            csv_parts.append(f"{prefix}_{k}")
    csv_parts.append("lr")
    csv_header = ",".join(csv_parts)

    for epoch in (bar := tqdm(range(1, args.epochs + 1), desc="Epoch", unit="ep")):
        t0 = time.time()

        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, epoch, args.epochs, args.accum_steps,
            use_reid=args.use_reid,
        )
        va = validate(model, val_loader, criterion, device,
                      epoch, args.epochs,
                      use_amp=(scaler is not None),
                      use_reid=args.use_reid)
        scheduler.step()

        elapsed = time.time() - t0
        is_best = va["dice_coeff"] > best_dice
        lr_now  = scheduler.get_last_lr()[0]

        bar.set_postfix(va_dice=f"{va['dice_coeff']:.4f}",
                        best=f"{max(best_dice, va['dice_coeff']):.4f}")

        extra = ""
        if "reid_loss" in va: extra += f"  reid={va['reid_loss']:.4f}"
        tqdm.write(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"tr={tr['total']:.4f}/{tr['dice_coeff']:.4f}  "
            f"va={va['total']:.4f}/{va['dice_coeff']:.4f}"
            f"{extra}  lr={lr_now:.2e}  ({elapsed:.1f}s)"
            + ("  ★THIS SHIT GOOD!!!" if is_best else ""))

        row = [str(epoch)]
        for prefix, metrics in [("tr", tr), ("va", va)]:
            for k in loss_keys:
                row.append(f"{metrics.get(k, 0.0):.6f}")
        row.append(f"{lr_now:.6e}")
        log_rows.append(",".join(row))

        if is_best:
            best_dice = va["dice_coeff"]
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch:04d}.pth")

    with open(save_dir / "train_log.csv", "w") as f:
        f.write(csv_header + "\n")
        f.write("\n".join(log_rows))

    config = {
        "use_semantic_prompt": args.use_semantic_prompt,
        "use_sparse_gat":     args.use_sparse_gat,
        "use_reid":           args.use_reid,
        "gnn_iters":          args.gnn_iters,
        "gat_layers":         args.gat_layers,
        "gat_heads":          args.gat_heads,
        "k_neighbors":        args.k_neighbors,
        "max_nodes":          args.max_nodes,
        "node_threshold":     args.node_threshold,
        "n_prompt_tokens":    args.n_prompt_tokens,
        "reid_embed_dim":     args.reid_embed_dim,
    }
    torch.save(config, save_dir / "model_config.pth")

    print(f"\nDone. Best val Dice = {best_dice:.4f}  →  {save_dir}")


def _parse():
    p = argparse.ArgumentParser(
        description="SAM3 UNet V2 (SemanticPrompt + SparseGAT + ReID)")

    p.add_argument("--data",        type=str, required=True)
    p.add_argument("--vessels",     type=str, default="LAD,LCx,RCA")
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--no_cache",    action="store_true")
    p.add_argument("--merge_split", action="store_true")
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--split_seed",  type=int,   default=42)
    p.add_argument("--artifact_prob", type=float, default=0.35,
                   help="訓練資料加偽影的機率（0.35 ≈ 1/3）")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch",       type=int,   default=8)
    p.add_argument("--accum_steps", type=int,   default=1)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--scale_lr",    action="store_true")
    p.add_argument("--amp",         action="store_true")
    p.add_argument("--compile",     action="store_true")
    p.add_argument("--save_dir",    type=str,   default="./checkpoints")
    p.add_argument("--save_every",  type=int,   default=10)
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--checkpoint",        type=str,   default=None)
    p.add_argument("--unfreeze",          action="store_true")
    p.add_argument("--backbone_lr_scale", type=float, default=0.01)
    p.add_argument("--use_semantic_prompt", action="store_true", default=True)
    p.add_argument("--no_semantic_prompt",  action="store_true")
    p.add_argument("--use_sparse_gat",      action="store_true", default=True)
    p.add_argument("--use_dense_gnn",       action="store_true")
    p.add_argument("--use_reid",            action="store_true", default=True)
    p.add_argument("--no_reid",             action="store_true")
    p.add_argument("--n_prompt_tokens", type=int, default=8)
    p.add_argument("--gat_layers",     type=int,   default=2)
    p.add_argument("--gat_heads",      type=int,   default=4)
    p.add_argument("--k_neighbors",    type=int,   default=16)
    p.add_argument("--max_nodes",      type=int,   default=4096)
    p.add_argument("--node_threshold", type=float, default=0.3)
    p.add_argument("--gnn_iters",      type=int,   default=3)
    p.add_argument("--reid_embed_dim", type=int, default=128)
    p.add_argument("--lambda_reid",    type=float, default=0.1)
    p.add_argument("--lambda_cldice", type=float, default=0.5,
                   help="clDice 在總損失中的權重（reference 預設 0.5）")
    p.add_argument("--cldice_iter",   type=int,   default=10,
                   help="clDice 中 SoftSkeletonize 迭代次數（越大越慢但骨架更乾淨）")

    args = p.parse_args()
    if args.no_semantic_prompt: args.use_semantic_prompt = False
    if args.no_reid:            args.use_reid = False
    if args.use_dense_gnn:      args.use_sparse_gat = False
    return args


if __name__ == "__main__":
    train(_parse())