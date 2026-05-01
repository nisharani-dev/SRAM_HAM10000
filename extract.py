# extract_weights.py
import torch
import numpy as np
import os
import timm

def load_model(pth_path, num_classes=7):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip "backbone." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[len("backbone."):] if k.startswith("backbone.") else k
        new_state_dict[key] = v

    model.load_state_dict(new_state_dict, strict=True)  # strict=True to catch mismatches
    model.eval()
    return model

def extract_and_save(pth_path, output_dir="weights"):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(pth_path)

    summary = []
    total_params = 0

    for name, param in model.named_parameters():
        data_f32 = param.detach().cpu().numpy()
        data_f16 = data_f32.astype(np.float16)

        safe_name = name.replace(".", "_")
        np.save(os.path.join(output_dir, f"{safe_name}_f32.npy"), data_f32)
        np.save(os.path.join(output_dir, f"{safe_name}_f16.npy"), data_f16)

        num_params = data_f32.size
        total_params += num_params
        summary.append({
            "layer": name,
            "shape": data_f32.shape,
            "params": num_params,
        })

    print(f"\n{'='*60}")
    print(f"Extracted {len(summary)} layers")
    print(f"Total parameters : {total_params:,}")
    print(f"Float32 size     : {total_params * 4 / 1024**2:.2f} MB")
    print(f"Float16 size     : {total_params * 2 / 1024**2:.2f} MB")
    print(f"SRAM words needed: {total_params:,}")
    print(f"Leakage (total)  : {total_params * 1.13 / 1e6:.4f} mW")
    print(f"{'='*60}\n")

    print(f"{'Layer':<50} {'Shape':<25} {'Params':>10}")
    print("-" * 90)
    for s in summary:
        print(f"{s['layer']:<50} {str(s['shape']):<25} {s['params']:>10,}")

    print(f"\nSaved to: {os.path.abspath(output_dir)}/")
    return summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_weights.py <path_to_model.pth>")
        sys.exit(1)
    extract_and_save(sys.argv[1])