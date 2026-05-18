from __future__ import annotations
import time
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt, binary_erosion


def _binary_surface(mask: np.ndarray) -> np.ndarray:

    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return mask & (~eroded)


def compute_hd95(pred: np.ndarray, gt: np.ndarray,
                 spacing: float = 1.0,
                 percentile: float = 95.0,
                 empty_penalty: Optional[float] = None) -> float:

    pred = np.asarray(pred).astype(bool)
    gt   = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} vs gt {gt.shape} shape mismatch")

    p_any = pred.any()
    g_any = gt.any()
    if not p_any and not g_any:
        return 0.0
    if not p_any or not g_any:
        if empty_penalty is None:
            H, W = pred.shape
            return float(np.sqrt(H * H + W * W) * spacing)
        return float(empty_penalty)

    pred_surf = _binary_surface(pred)
    gt_surf   = _binary_surface(gt)

    if not pred_surf.any():
        pred_surf = pred
    if not gt_surf.any():
        gt_surf = gt

    dt_to_gt   = distance_transform_edt(~gt_surf)
    dt_to_pred = distance_transform_edt(~pred_surf)

    d_p2g = dt_to_gt[pred_surf]   * spacing
    d_g2p = dt_to_pred[gt_surf]   * spacing

    return float(max(np.percentile(d_p2g, percentile),
                     np.percentile(d_g2p, percentile)))


def compute_iou(pred: np.ndarray, gt: np.ndarray, smooth: float = 1.0) -> float:

    pred = np.asarray(pred).astype(bool)
    gt   = np.asarray(gt).astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float((inter + smooth) / (union + smooth))


def compute_dice(pred: np.ndarray, gt: np.ndarray, smooth: float = 1.0) -> float:

    pred = np.asarray(pred).astype(bool)
    gt   = np.asarray(gt).astype(bool)
    inter = (pred & gt).sum()
    return float((2 * inter + smooth) / (pred.sum() + gt.sum() + smooth))


def measure_inference_latency(
        model: nn.Module,
        input_shape: Sequence[int] = (1, 1, 512, 512),
        device: str = "cuda",
        vessel_id: int = 0,
        n_warmup: int = 20,
        n_runs: int = 100,
        use_amp: bool = False,
) -> Dict[str, float]:

    if input_shape[0] != 1:
        warnings.warn("input_shape[0] != 1; this measures *batched* latency, not per-image.")

    model.eval()
    device_t = torch.device(device)
    x = torch.randn(*input_shape, device=device_t)
    vids = torch.full((input_shape[0],), int(vessel_id),
                       device=device_t, dtype=torch.long)

    is_cuda = device_t.type == "cuda"

    def _forward():
        with torch.no_grad():
            if use_amp and is_cuda:
                with torch.amp.autocast("cuda"):
                    out = model(x, vids)
            else:
                out = model(x, vids)
        return out

    for _ in range(n_warmup):
        _forward()
    if is_cuda:
        torch.cuda.synchronize()

    times_ms: List[float] = []
    for _ in range(n_runs):
        if is_cuda:
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            _forward()
            end_ev.record()
            torch.cuda.synchronize()
            times_ms.append(start_ev.elapsed_time(end_ev))
        else:
            t0 = time.perf_counter()
            _forward()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms":   float(np.mean(arr)),
        "std_ms":    float(np.std(arr)),
        "p50_ms":    float(np.percentile(arr, 50)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "min_ms":    float(np.min(arr)),
        "max_ms":    float(np.max(arr)),
        "fps":       1000.0 / float(np.median(arr)),
        "n_warmup":  n_warmup,
        "n_runs":    n_runs,
        "batch":     int(input_shape[0]),
        "amp":       bool(use_amp),
        "device":    str(device_t),
    }

def count_flops(
        model: nn.Module,
        input_shape: Sequence[int] = (1, 1, 512, 512),
        device: str = "cuda",
        vessel_id: int = 0,
) -> Dict[str, Any]:

    model.eval()
    device_t = torch.device(device)
    x = torch.randn(*input_shape, device=device_t)
    vids = torch.full((input_shape[0],), int(vessel_id),
                       device=device_t, dtype=torch.long)

    try:
        from torch.utils.flop_counter import FlopCounterMode  # type: ignore
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            _ = model(x, vids)
        total = int(counter.get_total_flops())
        return {
            "flops":  total,
            "gflops": total / 1e9,
            "method": "torch.utils.flop_counter",
            "note":   "PyTorch 內建；計的是 FLOPs (1 MAC ≈ 2 FLOPs 但這套已換算)",
        }
    except (ImportError, AttributeError):
        pass

    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
        with torch.no_grad():
            fca = FlopCountAnalysis(model, (x, vids))
            fca = fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            total = int(fca.total())
        return {
            "flops":  total * 2,    # fvcore 算的是 MACs（multiply-accumulate）
            "gflops": (total * 2) / 1e9,
            "method": "fvcore",
            "note":   "fvcore 報的是 MACs；這裡 ×2 換算 FLOPs（1 MAC = 1 mul + 1 add）",
        }
    except ImportError:
        pass

    try:
        from thop import profile  # type: ignore
        with torch.no_grad():
            macs, _ = profile(model, inputs=(x, vids), verbose=False)
        return {
            "flops":  int(macs * 2),
            "gflops": (macs * 2) / 1e9,
            "method": "thop",
            "note":   "thop 報的是 MACs；這裡 ×2 換算 FLOPs",
        }
    except ImportError:
        pass

    return {
        "flops":  -1, "gflops": -1.0,
        "method": "none",
        "note":   "找不到 FLOP 計算工具。請安裝：torch>=2.1（內建）或 "
                  "pip install fvcore / pip install thop",
    }

def _cache_probs_and_gts(
        model: nn.Module,
        loader,
        device: str,
        use_amp: bool = False,
        prob_extractor: Optional[Callable] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:

    model.eval()
    device_t = torch.device(device)
    cached = []
    if prob_extractor is None:
        def prob_extractor(model, imgs, vids):
            out = model(imgs, vids)
            if isinstance(out, tuple):
                out = out[0]
            return torch.sigmoid(out[:, 1].float())

    with torch.no_grad():
        for batch in loader:
            imgs, masks, vids, _ = batch
            imgs = imgs.to(device_t, non_blocking=True)
            vids = vids.to(device_t, non_blocking=True)

            if use_amp and device_t.type == "cuda":
                with torch.amp.autocast("cuda"):
                    prob = prob_extractor(model, imgs, vids)
            else:
                prob = prob_extractor(model, imgs, vids)

            prob_np = prob.float().cpu().numpy()
            mask_np = masks.cpu().numpy()

            for b in range(prob_np.shape[0]):
                cached.append((prob_np[b], (mask_np[b] > 0.5).astype(np.uint8)))
    return cached


def sweep_threshold_per_image(
        model: nn.Module,
        loader,
        device: str,
        search_range: Tuple[float, float] = (0.30, 0.65),
        step: float = 0.025,
        set_name: str = "Val",
        use_amp: bool = False,
        prob_extractor: Optional[Callable] = None,
        post_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        verbose: bool = True,
) -> Dict[str, Any]:

    cached = _cache_probs_and_gts(model, loader, device, use_amp, prob_extractor)
    if not cached:
        if verbose:
            print(f"[{set_name}] empty loader; cannot sweep.")
        return {"best_thr": None, "best_dice": None, "results": []}

    thresholds = np.arange(search_range[0], search_range[1] + 1e-9, step)
    thresholds = [float(round(t, 4)) for t in thresholds]

    if verbose:
        print(f"[{set_name}] threshold sweep: {len(thresholds)} thresholds in "
              f"[{thresholds[0]:.3f}, {thresholds[-1]:.3f}], step={step}")
        print(f"[{set_name}] cached {len(cached)} samples")

    results = []
    best = {"thr": None, "mean_dice": -1.0, "std_dice": 0.0,
            "per_image": None}

    for thr in thresholds:
        dices = []
        for prob_np, gt_np in cached:
            pred_bin = (prob_np > thr).astype(np.uint8)
            if post_fn is not None:
                pred_bin = post_fn(pred_bin)
            dices.append(compute_dice(pred_bin, gt_np))
        m = float(np.mean(dices))
        s = float(np.std(dices))
        results.append({"thr": thr, "mean_dice": m, "std_dice": s})
        if m > best["mean_dice"]:
            best.update(thr=thr, mean_dice=m, std_dice=s,
                        per_image=dices)
        if verbose:
            print(f"  thr={thr:.3f}  mean_dice={m:.4f} ± {s:.4f}")

    if verbose:
        print(f"[{set_name}] BEST thr={best['thr']:.3f}  "
              f"mean per-image Dice={best['mean_dice']:.4f} ± {best['std_dice']:.4f}")

    return {
        "best_thr":           best["thr"],
        "best_dice":          best["mean_dice"],
        "best_dice_std":      best["std_dice"],
        "best_per_image":     best["per_image"],
        "results":            results,
        "n_samples":          len(cached),
        "set_name":           set_name,
    }



def evaluate_at_threshold_per_image(
        model: nn.Module,
        loader,
        device: str,
        threshold: float,
        use_amp: bool = False,
        compute_hd95_flag: bool = True,
        prob_extractor: Optional[Callable] = None,
        post_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        spacing: float = 1.0,
) -> Dict[str, Any]:

    model.eval()
    device_t = torch.device(device)
    if prob_extractor is None:
        def prob_extractor(model, imgs, vids):
            out = model(imgs, vids)
            if isinstance(out, tuple): out = out[0]
            return torch.sigmoid(out[:, 1].float())

    per_image: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            imgs, masks, vids, paths = batch
            imgs = imgs.to(device_t, non_blocking=True)
            vids = vids.to(device_t, non_blocking=True)
            if use_amp and device_t.type == "cuda":
                with torch.amp.autocast("cuda"):
                    prob = prob_extractor(model, imgs, vids)
            else:
                prob = prob_extractor(model, imgs, vids)
            prob_np = prob.float().cpu().numpy()
            mask_np = masks.cpu().numpy()

            for b in range(prob_np.shape[0]):
                pred_bin = (prob_np[b] > threshold).astype(np.uint8)
                if post_fn is not None:
                    pred_bin = post_fn(pred_bin)
                gt_bin = (mask_np[b] > 0.5).astype(np.uint8)
                rec: Dict[str, Any] = {
                    "filename": str(paths[b]) if isinstance(paths, (list, tuple)) else str(paths),
                    "dice":     compute_dice(pred_bin, gt_bin),
                    "iou":      compute_iou(pred_bin, gt_bin),
                }
                if compute_hd95_flag:
                    rec["hd95"] = compute_hd95(pred_bin, gt_bin, spacing=spacing)
                per_image.append(rec)

    def _ms(key: str) -> Tuple[float, float]:
        vals = np.asarray([r[key] for r in per_image], dtype=np.float64)
        return float(vals.mean()), float(vals.std())

    summary = {
        "dice_mean":  _ms("dice")[0],  "dice_std":  _ms("dice")[1],
        "iou_mean":   _ms("iou")[0],   "iou_std":   _ms("iou")[1],
        "n_samples":  len(per_image),
        "threshold":  float(threshold),
    }
    if compute_hd95_flag:
        summary["hd95_mean"], summary["hd95_std"] = _ms("hd95")
    return {"per_image": per_image, "summary": summary, "threshold": float(threshold)}
