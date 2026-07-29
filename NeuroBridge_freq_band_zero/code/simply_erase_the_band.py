"""
Simply erase EEG frequency bands via Fourier mask (rFFT -> zero bins -> irFFT).

Bands (Hz):
  delta       0.5 – 4
  theta       4   – 8
  alpha       8   – 13
  beta        13  – 30
  low_gamma   30  – 45
  gamma       45  – 70
  high_gamma  70  – 100

Usage:
  python "simply erase the band" --bands alpha
  python "simply erase the band" --bands delta theta alpha beta low_gamma gamma high_gamma
  python "simply erase the band" --all_bands_separately
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np

# Seven target frequency bands (f_lo, f_hi) in Hz
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
    "gamma": (45.0, 70.0),
    "high_gamma": (70.0, 100.0),
}

BAND_NAMES = list(EEG_BANDS.keys())


def fourier_mask_zero(
    x: np.ndarray,
    sfreq: float,
    band_names: Iterable[str],
) -> np.ndarray:
    """
    Zero selected frequency bands with a Fourier mask.

    x: array with time on the last axis, e.g. (..., T)
    sfreq: sampling rate in Hz (Things-EEG preprocess default: 250)
    """
    names = [n.lower().strip().replace("-", "_") for n in band_names]
    for name in names:
        if name not in EEG_BANDS:
            raise ValueError(f"Unknown band '{name}'. Choose from: {BAND_NAMES}")

    x = np.asarray(x, dtype=np.float64)
    spectrum = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq)

    mask = np.zeros_like(freqs, dtype=bool)
    for name in names:
        f_lo, f_hi = EEG_BANDS[name]
        mask |= (freqs >= f_lo) & (freqs < f_hi)

    spectrum[..., mask] = 0.0  # <-- Fourier mask: set target bins to zero
    return np.fft.irfft(spectrum, n=x.shape[-1], axis=-1)


def zero_all_bands_separately(
    x: np.ndarray,
    sfreq: float,
) -> dict[str, np.ndarray]:
    """Return one copy per band, each with only that band erased."""
    return {name: fourier_mask_zero(x, sfreq, [name]) for name in BAND_NAMES}


def process_npy(
    input_path: str,
    output_path: str,
    sfreq: float,
    band_names: list[str],
) -> None:
    data = np.load(input_path)
    filtered = fourier_mask_zero(data, sfreq=sfreq, band_names=band_names)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, filtered.astype(data.dtype))
    print(f"  saved {output_path}  shape={filtered.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fourier-mask zero EEG bands (delta/theta/alpha/beta/low_gamma/gamma/high_gamma)"
    )
    parser.add_argument(
        "--input_dir",
        default="./data/things_eeg/preprocessed_eeg",
        type=str,
        help="directory with sub-XX/train.npy and test.npy",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="output directory (default: input_dir + band suffix)",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=BAND_NAMES,
        default=None,
        help="bands to zero via Fourier mask",
    )
    parser.add_argument(
        "--all_bands_separately",
        action="store_true",
        help="write 7 outputs, each with one band zeroed",
    )
    parser.add_argument("--sfreq", default=250, type=float, help="sampling rate in Hz")
    parser.add_argument(
        "--sub_ids",
        nargs="+",
        type=int,
        default=None,
        help="subject ids (default: all sub-XX under input_dir)",
    )
    args = parser.parse_args()

    if args.all_bands_separately and args.bands:
        parser.error("Use either --bands or --all_bands_separately, not both.")
    if not args.all_bands_separately and not args.bands:
        parser.error("Specify --bands or --all_bands_separately.")

    if args.output_dir is None:
        if args.all_bands_separately:
            args.output_dir = f"{args.input_dir.rstrip('/\\')}_erase_each_band_fourier"
        else:
            band_tag = "-".join(args.bands)
            args.output_dir = f"{args.input_dir.rstrip('/\\')}_zero_{band_tag}_fourier"

    sub_dirs = sorted(
        d
        for d in os.listdir(args.input_dir)
        if d.startswith("sub-") and os.path.isdir(os.path.join(args.input_dir, d))
    )
    if args.sub_ids:
        wanted = {f"sub-{i:02d}" for i in args.sub_ids}
        sub_dirs = [d for d in sub_dirs if d in wanted]
    if not sub_dirs:
        raise FileNotFoundError(f"No subject folders found under {args.input_dir}")

    print(f"method=fourier_mask, sfreq={args.sfreq}")
    print(f"bands={BAND_NAMES if args.all_bands_separately else args.bands}")
    print(f"input : {args.input_dir}")
    print(f"output: {args.output_dir}")

    for sub_dir in sub_dirs:
        print(f"\n{sub_dir}")
        sub_in = os.path.join(args.input_dir, sub_dir)
        sub_out = os.path.join(args.output_dir, sub_dir)

        for split in ("train", "test"):
            in_path = os.path.join(sub_in, f"{split}.npy")
            if not os.path.isfile(in_path):
                print(f"  skip missing {in_path}")
                continue

            if args.all_bands_separately:
                data = np.load(in_path)
                ablated = zero_all_bands_separately(data, sfreq=args.sfreq)
                for band_name, filtered in ablated.items():
                    out_path = os.path.join(sub_out, band_name, f"{split}.npy")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, filtered.astype(data.dtype))
                    print(f"  saved {out_path}  shape={filtered.shape}")
            else:
                out_path = os.path.join(sub_out, f"{split}.npy")
                process_npy(in_path, out_path, args.sfreq, args.bands)

    print("\nDone.")


if __name__ == "__main__":
    main()
