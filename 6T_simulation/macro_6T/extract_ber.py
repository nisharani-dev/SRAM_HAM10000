# extract_ber.py
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
import sys

def load_vtc(filepath):
    data = np.loadtxt(filepath, skiprows=1)
    print(f"  Columns found: {data.shape[1]}")
    print(f"  Col0 range: {data[:,0].min():.3f} to {data[:,0].max():.3f}")
    print(f"  Last col range: {data[:,-1].min():.3f} to {data[:,-1].max():.3f}")
    vin  = data[:, 0]   # first column — swept voltage
    vout = data[:, -1]  # last column — output voltage
    return vin, vout

def calculate_snm(vin, vout):
    """
    Butterfly curve SNM:
    Curve 1: vout = f(vin)       — inverter 1 VTC
    Curve 2: vin  = f(vout)      — inverter 2 VTC (mirrored)
    SNM = side of largest square in the eye between the two curves.
    """
    # Interpolate curve 2 as mirror: for each vout value, find vin
    # Mirror: x-axis becomes vout, y-axis becomes vin
    from scipy.interpolate import interp1d

    # Curve 1: y1 = f(x)
    x = vin
    y1 = vout

    # Curve 2: mirrored — x2=vout, y2=vin → reinterpolate onto same x axis
    # We want y2(x) = inverse of VTC, i.e. for input x find what vin gives vout=x
    try:
        inv_interp = interp1d(vout, vin, kind="linear",
                              bounds_error=False, fill_value="extrapolate")
        y2 = inv_interp(x)
    except Exception:
        y2 = x  # fallback

    # SNM = max square fitting in eye
    # Eye opening at each point = |y1 - y2| / sqrt(2)
    eye = np.abs(y1 - y2) / np.sqrt(2)

    # Find the two local maxima of eye (upper and lower lobes)
    # SNM = minimum of the two maxima
    mid = len(eye) // 2
    snm_low  = np.max(eye[:mid])
    snm_high = np.max(eye[mid:])
    snm = min(snm_low, snm_high)

    return snm, x, y1, y2

def snm_to_ber(snm, sigma):
    return 0.5 * erfc(snm / (sigma * np.sqrt(2)))

def plot_butterfly(x, y1, y2, snm, recommended_ber,
                   output_path="snm_butterfly.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.plot(x,  y1, color="royalblue", linewidth=2, label="Inv1: Q_bar = f(Q)")
    ax1.plot(y1, x,  color="tomato",    linewidth=2, label="Inv2: Q = f(Q_bar)")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Unity line")

    ax1.set_xlabel("Q (V)", fontsize=12)
    ax1.set_ylabel("Q_bar (V)", fontsize=12)
    ax1.set_title("6T SRAM Butterfly Curve", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.text(0.05, 0.92,
             f"SNM = {snm*1000:.2f} mV\nBER = {recommended_ber:.3e}",
             transform=ax1.transAxes, fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax2 = axes[1]
    sigmas = np.linspace(0.010, 0.050, 200)
    bers   = [snm_to_ber(snm, s) for s in sigmas]
    ax2.semilogy(sigmas * 1000, bers, color="purple", linewidth=2)
    ax2.axvline(x=25, color="gray", linestyle="--",
                linewidth=1, label="Typical sigma (25mV)")
    ax2.axhline(y=recommended_ber, color="tomato", linestyle="--",
                linewidth=1, label=f"BER = {recommended_ber:.2e}")
    ax2.plot(25, recommended_ber, "ro", markersize=8, zorder=5)
    ax2.set_xlabel("Noise sigma (mV)", fontsize=12)
    ax2.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax2.set_title("BER vs Noise Sigma", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle(
        "FreePDK45 6T SRAM — SNM Analysis | VDD=1.0V | TT Corner | 27°C",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    vtc1_path = sys.argv[1] if len(sys.argv) > 1 else "vtc1.txt"
    vtc2_path = sys.argv[2] if len(sys.argv) > 2 else "vtc2.txt"

    print(f"\nLoading VTC curves...")
    vin1, vout1 = load_vtc(vtc1_path)
    vin2, vout2 = load_vtc(vtc2_path)
    print(f"Curve 1: {len(vin1)} points")
    print(f"Curve 2: {len(vin2)} points")

    snm, x, y1, y2 = calculate_snm(vin1, vout1)
    recommended_ber = snm_to_ber(snm, sigma=0.025)

    print(f"\n{'='*50}")
    print(f"Static Noise Margin (SNM) = {snm*1000:.2f} mV")
    print(f"{'='*50}")

    sigmas = {
        "best case  (15mV)": 0.015,
        "typical    (25mV)": 0.025,
        "worst case (35mV)": 0.035,
    }
    print(f"\n{'Noise sigma':<25} {'BER':>15}")
    print("-" * 42)
    for label, sigma in sigmas.items():
        ber = snm_to_ber(snm, sigma)
        print(f"{label:<25} {ber:>15.4e}")

    print(f"\n{'='*50}")
    print(f"Recommended BER (sigma=25mV): {recommended_ber:.6e}")
    print(f"{'='*50}\n")

    plot_butterfly(x, y1, y2, snm, recommended_ber)

    with open("extracted_ber.txt", "w") as f:
        f.write(f"SNM_mV={snm*1000:.4f}\n")
        f.write(f"BER={recommended_ber:.6e}\n")
        f.write(f"sigma_mV=25.0\n")
        f.write(f"VDD=1.0\n")
        f.write(f"corner=TT\n")
        f.write(f"temp_C=27\n")

    print(f"Results saved to extracted_ber.txt")
    print(f"Use BER={recommended_ber:.4e} in sram_model.py")