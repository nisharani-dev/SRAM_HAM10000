# inference_engine.py
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

# ─────────────────────────────────────────────
# HARDCODED PATHS
# ─────────────────────────────────────────────
PTH_PATH      = "/mnt/c/Users/agarw/OneDrive/Desktop/OpenRAM SAAS/best_model.pth"
WEIGHTS_DIR   = "/mnt/c/Users/agarw/OneDrive/Desktop/OpenRAM SAAS/weights"
KAGGLE_BASE   = "/home/agarw/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2"
METADATA_PATH = "test.csv"
IMAGE_DIRS    = [
    os.path.join(KAGGLE_BASE, "HAM10000_images_part_1"),
    os.path.join(KAGGLE_BASE, "HAM10000_images_part_2"),
]
OUTPUT_DIR    = "/mnt/c/Users/agarw/OneDrive/Desktop/OpenRAM SAAS/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES    = 7
BATCH_SIZE     = 1
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORD_SIZE_BITS = 16

# OpenRAM TT .lib timing
CYCLE_TIME_NS  = 0.614
ACCESS_TIME_NS = 0.165
SETUP_TIME_NS  = 0.009
HOLD_TIME_NS   = 0.001
MAX_FREQ_GHZ   = 1.0 / CYCLE_TIME_NS

# Leakage
LEAKAGE_MACRO_MW    = 0.002314
CELLS_PER_MACRO     = 128 * 16
LEAKAGE_PER_CELL_NW = (LEAKAGE_MACRO_MW * 1e6) / CELLS_PER_MACRO

# SPICE extracted
SNM_MV       = 287.36
NOMINAL_BER  = 7.03e-31

# BER sweep
BER_VALUES = [7.03e-31, 1e-20, 1e-15, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2]

# HAM10000 classes
CLASS_NAMES = {
    0: "akiec", 1: "bcc",  2: "bkl",
    3: "df",    4: "mel",  5: "nv",  6: "vasc"
}

# ─────────────────────────────────────────────
# SRAM MODEL
# ─────────────────────────────────────────────
class SRAMModel:
    def __init__(self, ber):
        self.ber         = ber
        self.total_cells = 0
        self.total_flips = 0

    def store_and_retrieve(self, weights_f32):
        w_f16   = weights_f32.astype(np.float16)
        w_int16 = w_f16.view(np.uint16).copy()
        corrupted, n = self._inject(w_int16)
        self.total_cells += w_int16.size
        self.total_flips += n
        return corrupted.view(np.float16).astype(np.float32)

    def _inject(model, sram):
        with torch.no_grad():
            for name, param in model.named_parameters():
                w_np      = param.detach().cpu().numpy().astype(np.float32)
                corrupted = sram.store_and_retrieve(w_np)
                # Clamp to valid float16 range to prevent inf/nan from
                # bit-flipped exponent bits causing degenerate softmax outputs
                corrupted = np.nan_to_num(corrupted, nan=0.0,
                                        posinf=65504.0, neginf=-65504.0)
                corrupted = np.clip(corrupted, -65504.0, 65504.0)
                param.copy_(torch.from_numpy(corrupted))
        return model

    def hardware_latency_ms(self):
        return (self.total_cells * CYCLE_TIME_NS) / 1e6

    def leakage_mw(self):
        return self.total_cells * LEAKAGE_PER_CELL_NW / 1e6

# ─────────────────────────────────────────────
# MODEL  (timm — matches checkpoint key names)
# ─────────────────────────────────────────────
def build_model():
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    return model

def load_clean_model():
    model      = build_model()
    checkpoint = torch.load(PTH_PATH, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        sd = checkpoint.get("model_state_dict",
             checkpoint.get("state_dict", checkpoint))
    else:
        sd = checkpoint

    # Strip any 'backbone.' prefix saved during training
    new_sd = {}
    for k, v in sd.items():
        key = k[len("backbone."):] if k.startswith("backbone.") else k
        new_sd[key] = v

    model.load_state_dict(new_sd, strict=True)
    model.eval()
    return model

def inject_sram_weights(model, sram):
    with torch.no_grad():
        for name, param in model.named_parameters():
            w_np      = param.detach().cpu().numpy().astype(np.float32)
            corrupted = sram.store_and_retrieve(w_np)
            param.copy_(torch.from_numpy(corrupted))
    return model

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class HAM10000Dataset(Dataset):
    def __init__(self, transform=None):
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")

        self.df = pd.read_csv(METADATA_PATH)
        self.class_to_idx = {v: k for k, v in CLASS_NAMES.items()}
        self.df["label"] = self.df["dx"].map(self.class_to_idx)
        self.transform = transform

        # Warn about images that can't be found
        missing = [
            row["image_id"]
            for _, row in self.df.iterrows()
            if self._find_image(row["image_id"]) is None
        ]
        if missing:
            print(f"WARNING: {len(missing)}/{len(self.df)} images not found on disk.")

        print(f"Test set size : {len(self.df)} images")
        print(f"Class mapping : {self.class_to_idx}")

    def _find_image(self, image_id):
        for d in IMAGE_DIRS:
            p = os.path.join(d, f"{image_id}.jpg")
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = self._find_image(row["image_id"])
        label = int(row["label"])

        if path is None:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def _sanitize_probs(probs):
    """
    Guard against inf/nan logits produced by bit-corrupted weights.
    If a row is non-finite, replace with a uniform distribution so
    downstream metrics remain well-defined instead of producing NaN.
    """
    if np.isfinite(probs).all():
        return probs

    probs = np.where(np.isfinite(probs), probs, 0.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    # Rows where every logit was inf/nan become uniform
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return probs / row_sums


def run_inference(model, loader):
    model.eval()
    model.to(DEVICE)
    all_preds, all_labels, all_probs, all_times = [], [], [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            t0     = time.perf_counter()
            logits = model(images)
            t1     = time.perf_counter()

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs = _sanitize_probs(probs)   # fix inf/nan from corrupted weights
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_times.append((t1 - t0) * 1000)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(loader)} images...", flush=True)

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs),
            np.array(all_times))

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_ece(probs, labels, n_bins=10):
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs  = preds == labels
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    n     = len(labels)
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(accs[mask].mean() - confs[mask].mean())
    return float(ece)

def compute_metrics(preds, labels, probs, inf_times, sram, ber):
    correct  = preds == labels
    accuracy = float(correct.mean() * 100)

    max_probs = probs.max(axis=1)

    # Use nanmean so any residual NaN values don't poison the metric
    mean_conf_correct = float(np.nanmean(max_probs[correct]))   if correct.any()   else 0.0
    mean_conf_wrong   = float(np.nanmean(max_probs[~correct]))  if (~correct).any() else 0.0
    overall_conf      = float(np.nanmean(max_probs))

    ece              = compute_ece(probs, labels)
    sw_latency_ms    = float(inf_times.mean())
    hw_latency_ms    = float(sram.hardware_latency_ms())
    total_latency_ms = sw_latency_ms + hw_latency_ms
    total_params     = sram.total_cells

    return {
        "ber"               : ber,
        "accuracy"          : accuracy,
        "mean_conf_correct" : mean_conf_correct,
        "mean_conf_wrong"   : mean_conf_wrong,
        "overall_conf"      : overall_conf,
        "ece"               : ece,
        "sw_latency_ms"     : sw_latency_ms,
        "hw_latency_ms"     : hw_latency_ms,
        "total_latency_ms"  : total_latency_ms,
        "sw_fps"            : float(1000.0 / sw_latency_ms),
        "hw_fps"            : float(1000.0 / total_latency_ms),
        "total_params"      : total_params,
        "f32_mb"            : float(total_params * 4 / 1024**2),
        "f16_mb"            : float(total_params * 2 / 1024**2),
        "compression_ratio" : 2.0,
        "leakage_mw"        : float(sram.leakage_mw()),
        "total_flips"       : int(sram.total_flips),
    }

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_results(results, baseline_acc, baseline_conf):
    bers      = [r["ber"]               for r in results]
    accs      = [r["accuracy"]          for r in results]
    confs     = [r["overall_conf"]      for r in results]
    conf_corr = [r["mean_conf_correct"] for r in results]
    conf_wrng = [r["mean_conf_wrong"]   for r in results]
    eces      = [r["ece"]               for r in results]
    hw_fps    = [r["hw_fps"]            for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "SRAM-Aware EfficientNetB0 on HAM10000 | "
        "FreePDK45 6T | TT Corner | VDD=1.0V | SNM=287.36mV",
        fontsize=12, fontweight="bold"
    )

    # Accuracy vs BER
    ax = axes[0, 0]
    ax.semilogx(bers, accs, "b-o", linewidth=2, markersize=6)
    ax.axhline(y=baseline_acc, color="gray", linestyle="--",
               label=f"Float32 baseline ({baseline_acc:.1f}%)")
    ax.set_xlabel("BER")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs BER")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Confidence vs BER
    ax = axes[0, 1]
    ax.semilogx(bers, confs,     "g-o",  linewidth=2, label="Overall")
    ax.semilogx(bers, conf_corr, "b--s", linewidth=1.5, label="Correct preds")
    ax.semilogx(bers, conf_wrng, "r--^", linewidth=1.5, label="Wrong preds")
    ax.axhline(y=baseline_conf, color="gray", linestyle="--",
               label=f"Baseline ({baseline_conf:.3f})")
    ax.set_xlabel("BER")
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Confidence Drop vs BER")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # ECE vs BER
    ax = axes[0, 2]
    ax.semilogx(bers, eces, "r-o", linewidth=2, markersize=6)
    ax.set_xlabel("BER")
    ax.set_ylabel("ECE")
    ax.set_title("Calibration Error vs BER")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # FPS vs BER
    ax = axes[1, 0]
    ax.semilogx(bers, hw_fps, "m-o", linewidth=2, markersize=6)
    ax.set_xlabel("BER")
    ax.set_ylabel("FPS (hardware-aware)")
    ax.set_title("Throughput vs BER")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Accuracy vs Confidence scatter
    ax = axes[1, 1]
    sc = ax.scatter(confs, accs,
                    c=np.log10([max(b, 1e-35) for b in bers]),
                    cmap="coolwarm", s=80, zorder=5)
    plt.colorbar(sc, ax=ax, label="log10(BER)")
    ax.set_xlabel("Mean Confidence")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Confidence")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Summary table
    ax = axes[1, 2]
    ax.axis("off")
    r = results[0]
    rows = [
        ["SNM",              "287.36 mV"],
        ["Nominal BER",      f"{r['ber']:.2e}"],
        ["Accuracy",         f"{r['accuracy']:.2f}%"],
        ["Baseline acc",     f"{baseline_acc:.2f}%"],
        ["HW latency",       f"{r['hw_latency_ms']:.3f} ms"],
        ["SW latency",       f"{r['sw_latency_ms']:.3f} ms"],
        ["Total latency",    f"{r['total_latency_ms']:.3f} ms"],
        ["HW FPS",           f"{r['hw_fps']:.2f}"],
        ["Float32 size",     f"{r['f32_mb']:.2f} MB"],
        ["Float16 size",     f"{r['f16_mb']:.2f} MB"],
        ["Compression",      "2x"],
        ["Leakage",          f"{r['leakage_mw']:.4f} mW"],
        ["Max freq",         f"{MAX_FREQ_GHZ:.3f} GHz"],
    ]
    tbl = ax.table(cellText=rows,
                   colLabels=["Metric", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax.set_title("Summary (Nominal BER)", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "inference_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("SRAM-Aware Inference Engine")
    print("EfficientNetB0 + HAM10000 + FreePDK45 6T SRAM")
    print(f"Device: {DEVICE}")
    print("="*60 + "\n")

    # Sanity check paths before doing anything
    print("Checking paths...")
    print(f"  METADATA_PATH exists : {os.path.exists(METADATA_PATH)}")
    print(f"  IMAGE_DIR_1 exists   : {os.path.exists(IMAGE_DIRS[0])}")
    print(f"  IMAGE_DIR_2 exists   : {os.path.exists(IMAGE_DIRS[1])}")
    print(f"  PTH_PATH exists      : {os.path.exists(PTH_PATH)}")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Cannot find metadata CSV: {METADATA_PATH}")
    if not os.path.exists(PTH_PATH):
        raise FileNotFoundError(f"Cannot find model weights: {PTH_PATH}")

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = HAM10000Dataset(transform=transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

    # ── Float32 baseline ──
    print("\nRunning float32 baseline...")
    base_model = load_clean_model().to(DEVICE)
    b_preds, b_labels, b_probs, b_times = run_inference(base_model, loader)
    baseline_acc  = float((b_preds == b_labels).mean() * 100)
    baseline_conf = float(b_probs.max(axis=1).mean())
    baseline_fps  = float(1000.0 / b_times.mean())

    print(f"\nBaseline accuracy  : {baseline_acc:.2f}%")
    print(f"Baseline confidence: {baseline_conf:.4f}")
    print(f"Baseline SW FPS    : {baseline_fps:.2f}")
    print(f"Baseline SW latency: {b_times.mean():.3f} ms/image")

    # ── BER sweep ──
    all_results = []
    for ber in BER_VALUES:
        print(f"\n{'='*40}")
        print(f"BER = {ber:.2e}")
        print(f"{'='*40}")

        sram  = SRAMModel(ber=ber)
        model = load_clean_model()
        model = inject_sram_weights(model, sram)
        model = model.to(DEVICE)

        preds, labels, probs, inf_times = run_inference(model, loader)
        metrics = compute_metrics(preds, labels, probs, inf_times, sram, ber)
        all_results.append(metrics)

        acc_drop  = baseline_acc - metrics["accuracy"]
        conf_drop = baseline_conf - metrics["overall_conf"]

        print(f"  Accuracy          : {metrics['accuracy']:.2f}% "
              f"(drop: {acc_drop:.2f}%)")
        print(f"  Conf (overall)    : {metrics['overall_conf']:.4f} "
              f"(drop: {conf_drop:.4f})")
        print(f"  Conf (correct)    : {metrics['mean_conf_correct']:.4f}")
        print(f"  Conf (wrong)      : {metrics['mean_conf_wrong']:.4f}")
        print(f"  ECE               : {metrics['ece']:.4f}")
        print(f"  SW latency        : {metrics['sw_latency_ms']:.3f} ms")
        print(f"  HW latency        : {metrics['hw_latency_ms']:.3f} ms")
        print(f"  Total latency     : {metrics['total_latency_ms']:.3f} ms")
        print(f"  SW FPS            : {metrics['sw_fps']:.2f}")
        print(f"  HW FPS            : {metrics['hw_fps']:.2f}")
        print(f"  Bit flips         : {metrics['total_flips']:,}")
        print(f"  Leakage           : {metrics['leakage_mw']:.6f} mW")

    # ── Speed improvement ──
    speedup = b_times.mean() / all_results[0]["sw_latency_ms"]
    print(f"\nFloat16 speedup vs Float32 : {speedup:.3f}x")
    print(f"Float32 size               : {all_results[0]['f32_mb']:.2f} MB")
    print(f"Float16 size               : {all_results[0]['f16_mb']:.2f} MB")
    print(f"Memory compression         : 2x")
    print(f"Max SRAM frequency         : {MAX_FREQ_GHZ:.3f} GHz")

    # ── Save JSON ──
    out_json = os.path.join(OUTPUT_DIR, "inference_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "snm_mv"          : SNM_MV,
                "nominal_ber"     : NOMINAL_BER,
                "cycle_time_ns"   : CYCLE_TIME_NS,
                "access_time_ns"  : ACCESS_TIME_NS,
                "max_freq_ghz"    : MAX_FREQ_GHZ,
                "leakage_macro_mw": LEAKAGE_MACRO_MW,
                "word_size_bits"  : WORD_SIZE_BITS,
            },
            "baseline": {
                "accuracy"        : baseline_acc,
                "confidence"      : baseline_conf,
                "sw_fps"          : baseline_fps,
                "sw_latency_ms"   : float(b_times.mean()),
            },
            "speedup_f16_vs_f32" : float(speedup),
            "sweep_results"      : all_results
        }, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ── Plot ──
    plot_results(all_results, baseline_acc, baseline_conf)

    print("\n" + "="*60)
    print("All done!")
    print(f"Results in: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
