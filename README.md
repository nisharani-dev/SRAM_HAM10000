# SRAM-Aware Neural Network Inference Engine

A hardware-software co-simulation framework that evaluates the reliability of a deep learning model (EfficientNet-B0) when its weights are stored in a 6T SRAM macro fabricated in FreePDK45. The project bridges SPICE-level SRAM characterization with PyTorch inference to quantify how bit errors in physical memory affect classification accuracy.

---

## What This Project Does

1. **Designs and characterizes a 6T SRAM macro** using OpenRAM on FreePDK45 (45nm) technology — generating SPICE netlists, timing `.lib` files, GDS layout, and LEF abstracts.
2. **Extracts the Static Noise Margin (SNM)** from SPICE butterfly-curve simulations and converts it to a Bit Error Rate (BER) using the complementary error function.
3. **Trains / loads an EfficientNet-B0** classifier on the HAM10000 skin lesion dataset (7 classes, ~10k images).
4. **Simulates SRAM storage of model weights** — weights are quantized to float16, stored in the SRAM model, and random bit-flips are injected at a given BER before inference.
5. **Sweeps BER from nominal (7.03×10⁻³¹) to degraded (10⁻²)** and records accuracy, confidence, ECE, latency, FPS, leakage, and bit-flip counts.
6. **Exposes a FastAPI endpoint** so the pipeline can be triggered as a SaaS service.

---

## Project Structure

```
.
├── main.py                  # FastAPI app — POST /run triggers OpenRAM pipeline
├── run_openram.py           # Subprocess wrapper for the SRAM compiler
├── inference_engine.py      # Core: SRAM model, BER sweep, metrics, plotting
├── extract.py               # Extracts EfficientNet-B0 weights to .npy (f32 + f16)
├── kaggle_setup.py          # Downloads HAM10000 dataset via kagglehub
├── test.csv                 # Test-split metadata (image_id, dx label)
├── test_metadata.csv        # Extended metadata
├── best_model.pth           # Trained EfficientNet-B0 checkpoint
├── model_weight.pth         # Alternate checkpoint
│
├── weights/                 # Extracted weights — EfficientNet-B0 (f32 + f16 .npy)
├── weights_VER1/            # Weights from an earlier model version (MobileNetV2-style)
│
├── 6T_simulation/
│   ├── config.py            # OpenRAM config: 128×16-bit, FreePDK45, single bank
│   └── macro_6T/
│       ├── sram_6T.sp       # SPICE netlist of the macro
│       ├── sram_6T.lef      # LEF abstract for place-and-route
│       ├── sram_6T.gds      # GDS layout
│       ├── sram_6T.v        # Verilog behavioral model
│       ├── sram_6T_*.lib    # Liberty timing files (TT/FF/SS corners, temps, voltages)
│       ├── delay_meas.sp    # SPICE testbench — read/write timing
│       ├── snm_sim_inv1/2.sp# SPICE testbench — butterfly curve (SNM)
│       ├── vtc1.txt / vtc2.txt # Simulated VTC data
│       ├── extract_ber.py   # SNM → BER extraction script
│       ├── extracted_ber.txt# Output: SNM=287.36mV, BER=7.03e-31
│       └── snm_butterfly.png# Butterfly curve plot
│
└── results_exp1..5/
    ├── inference_results.json  # Full metrics for each BER point
    └── inference_results.png   # 6-panel summary plot
```

---

## SRAM Macro Specifications

| Parameter         | Value                  |
|-------------------|------------------------|
| Technology        | FreePDK45 (45nm)       |
| Cell type         | 6T SRAM                |
| Organization      | 128 words × 16 bits    |
| VDD               | 1.0 V                  |
| Corner (nominal)  | TT, 25°C               |
| SNM               | 287.36 mV              |
| Nominal BER       | 7.03 × 10⁻³¹           |
| Cycle time        | 0.614 ns               |
| Access time       | 0.165 ns               |
| Max frequency     | ~1.63 GHz              |
| Leakage (macro)   | 2.314 mW               |

---

## Model & Dataset

| Parameter      | Value                                      |
|----------------|--------------------------------------------|
| Model          | EfficientNet-B0 (timm)                     |
| Parameters     | 4,016,515                                  |
| Float32 size   | 15.32 MB                                   |
| Float16 size   | 7.66 MB (2× compression)                   |
| Dataset        | HAM10000 skin lesion (7 classes)           |
| Test set       | 1,502 images                               |
| Baseline acc   | 81.42% (float32)                           |
| Baseline conf  | 0.9091                                     |
| SW latency     | 20.33 ms                                   |
| SW FPS         | 49.19                                      |

---

## Key Results (BER Sweep)

| BER          | Accuracy | Bit Flips | Total Latency (ms) | HW FPS |
|--------------|-----------|-----------|--------------------|--------|
| 7.03e-31     | 81.49%    | 0         | 20.014             | 49.97  |
| 1e-20        | 81.49%    | 0         | 19.990             | 50.02  |
| 1e-15        | 81.49%    | 0         | 20.194             | 49.52  |
| 1e-10        | 81.49%    | 0         | 19.641             | 50.91  |
| 1e-7         | 81.49%    | 6         | 20.013             | 49.97  |
| 1e-5         | 66.84%    | 631       | 19.705             | 50.75  |
| 1e-4         | 3.46%     | 6,333     | 19.615             | 50.98  |
| 1e-3         | 3.46%     | 63,876    | 19.724             | 50.70  |
| 1e-2         | 3.46%     | 642,582   | 19.875             | 50.32  |

The model maintains stable accuracy under SRAM bit errors up to a BER of **10⁻⁵**, after which accuracy degrades sharply. At BER ≥ 10⁻⁴, the model collapses to **3.46%** — the majority-class prior — indicating complete inference failure. The high static noise margin (287.36 mV) of the FreePDK45 6T cell places the nominal operating BER at 7.03×10⁻³¹, far below the degradation threshold.

---

## Hardware-Aware Metrics at Nominal BER

| Metric               | Value           |
|----------------------|-----------------|
| SNM                  | 287.36 mV       |
| Nominal BER          | 7.03 × 10⁻³¹   |
| Float16 accuracy     | 81.49%          |
| Float32 baseline acc | 81.42%          |
| Accuracy drop (F16)  | +0.07% (gain)   |
| HW latency           | 2.466 ms        |
| SW latency           | 17.548 ms       |
| Total latency        | 20.014 ms       |
| Throughput (HW FPS)  | 49.97 FPS       |
| Float32 model size   | 15.32 MB        |
| Float16 model size   | 7.66 MB         |
| Compression ratio    | 2×              |
| Leakage power        | 4.538 mW        |
| Max frequency        | 1.629 GHz       |
| F16 vs F32 speedup   | 1.16×           |

---

## Setup

### Prerequisites

- Python 3.9+
- PyTorch (CUDA optional but recommended)
- `timm`, `numpy`, `pandas`, `scipy`, `matplotlib`, `Pillow`, `fastapi`, `uvicorn`
- OpenRAM (for re-running SRAM compilation)
- Ngspice or compatible SPICE simulator (for SNM extraction)

### Install dependencies

```bash
pip install torch torchvision timm numpy pandas scipy matplotlib pillow fastapi uvicorn kagglehub
```

### Download the dataset

```bash
python kaggle_setup.py
```

This downloads HAM10000 via `kagglehub` and prints the local cache path. Update `IMAGE_DIRS` in `inference_engine.py` to match.

---

## Usage

### 1. Extract model weights to `.npy`

```bash
python extract.py best_model.pth
```

Saves per-layer float32 and float16 `.npy` files to `weights/`.

### 2. Run the BER sweep inference

```bash
python inference_engine.py
```

Runs the float32 baseline, then sweeps BER values, printing metrics and saving results to `results/inference_results.json` and `results/inference_results.png`.

### 3. Extract SNM and BER from SPICE simulation

```bash
python 6T_simulation/macro_6T/extract_ber.py vtc1.txt vtc2.txt
```

Reads VTC curves from SPICE output, computes SNM via butterfly analysis, and writes `extracted_ber.txt`.

### 4. Start the FastAPI SaaS server

```bash
python -m uvicorn main:app --reload
```

Then POST an OpenRAM config file to trigger compilation:

```bash
curl -X POST http://localhost:8000/run -F "file=@6T_simulation/config.py"
```

---

## How the SRAM Fault Injection Works

1. Each model parameter tensor is cast to `float16` and reinterpreted as `uint16`.
2. Each bit is independently flipped with probability equal to the target BER.
3. The corrupted `uint16` array is reinterpreted back as `float16`, then cast to `float32`.
4. `nan`/`inf` values from corrupted exponent bits are clamped to the float16 range (±65504).
5. The patched weights are loaded back into the model and inference runs normally.

---

## Output Files

Each experiment folder (`results_exp1/` … `result_exp5/`) contains:

- `inference_results.json` — full config, baseline, and per-BER metrics
- `inference_results.png` — 6-panel plot: accuracy vs BER, confidence drop, ECE, FPS, accuracy-confidence scatter, and a summary table

---

## Liberty Timing Corners Available

| File                          | Corner | VDD   | Temp  |
|-------------------------------|--------|-------|-------|
| `sram_6T_TT_1p0V_25C.lib`    | TT     | 1.0 V | 25°C  |
| `sram_6T_TT_0p9V_25C.lib`    | TT     | 0.9 V | 25°C  |
| `sram_6T_TT_1p1V_25C.lib`    | TT     | 1.1 V | 25°C  |
| `sram_6T_TT_1p0V_0C.lib`     | TT     | 1.0 V | 0°C   |
| `sram_6T_TT_1p0V_100C.lib`   | TT     | 1.0 V | 100°C |
| `sram_6T_FF_1p0V_25C.lib`    | FF     | 1.0 V | 25°C  |
| `sram_6T_SS_1p0V_25C.lib`    | SS     | 1.0 V | 25°C  |
