#!/usr/bin/env python3
"""把 generate_data.py 产出的逐样本 npy 打包成 Pulsar2 每输入一个 tar.gz。

用法: python make_calib_tars.py <task_dir>
"""
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


def pack_files(files: list[Path], out_tar: Path, limit: int = 64) -> int:
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(files)
    if limit:
        # 均匀采样，尽量覆盖早期/中期/后期
        if len(files) > limit:
            idx = sorted({round(i * (len(files) - 1) / (limit - 1))
                          for i in range(limit)})
            files = [files[i] for i in idx]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(files):
            shutil.copy2(f, td / f"{i:04d}.npy")
        with tarfile.open(out_tar, "w:gz") as tar:
            for npy in sorted(td.glob("*.npy")):
                tar.add(npy, arcname=npy.name)
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    export_dir = args.task_dir / "export"
    calib_src = args.src or (export_dir / "calib_dataset")
    calib_out = args.out or (export_dir / "calib_data")

    enc_names = ["encoder_input", "encoder_input_lengths"]
    # (保存目录名, ONNX 输入名)
    dec_names = [
        ("tokens", "tokens"),
        ("n_layer_self_k_cache", "in_n_layer_self_k_cache"),
        ("n_layer_self_v_cache", "in_n_layer_self_v_cache"),
        ("n_layer_cross_k", "n_layer_cross_k"),
        ("n_layer_cross_v", "n_layer_cross_v"),
        ("pe", "pe"),
        ("self_attn_mask", "self_attn_mask"),
        ("cross_attn_mask", "cross_attn_mask"),
    ]
    for name in enc_names:
        files = sorted((calib_src / "encoder").glob(f"*/{name}.npy"))
        n = pack_files(files, calib_out / f"{name}.tar.gz", limit=args.limit)
        print(f"encoder {name}: {n} samples -> {calib_out / (name + '.tar.gz')}")
    for save_name, tensor_name in dec_names:
        files = sorted((calib_src / "decoder").glob(f"*/{save_name}/*.npy"))
        n = pack_files(files, calib_out / f"{tensor_name}.tar.gz", limit=args.limit)
        print(f"decoder {tensor_name}: {n} samples -> {calib_out / (tensor_name + '.tar.gz')}")


if __name__ == "__main__":
    main()
