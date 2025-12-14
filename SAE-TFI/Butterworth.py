#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-Pass Frequency Enhancement for Audio Signals

This script applies a Butterworth high-pass filter to an audio signal,
amplifies the filtered component with different gains, and adds it back
to the original waveform to generate enhanced outputs.

Example:
  python scripts/highpass_enhance.py \
    --input data/input.wav \
    --output_dir outputs/type1 \
    --sr 16000 \
    --order 6 \
    --cutoffs 600 1000 1500 2000 \
    --gains 0.3 0.6 1.0 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile


# =========================
# Filter Utilities
# =========================

def design_butterworth_filter(
    cutoff_hz: float,
    sample_rate: int,
    filter_type: str = "highpass",
    order: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a digital Butterworth IIR filter.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency in Hz.
    sample_rate : int
        Sampling rate in Hz.
    filter_type : str
        "highpass" or "lowpass".
    order : int
        Filter order.

    Returns
    -------
    b, a : ndarray
        Numerator (b) and denominator (a) coefficients.
    """
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    nyq = sample_rate / 2.0
    normalized_cutoff = cutoff_hz / nyq
    if not (0.0 < normalized_cutoff < 1.0):
        raise ValueError(
            f"cutoff_hz must be in (0, Nyquist). Got cutoff={cutoff_hz}, Nyquist={nyq}"
        )

    b, a = signal.butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return b, a


def enhance_high_frequency(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    gain: float,
    order: int = 6,
) -> np.ndarray:
    """
    High-pass filter the signal and add the amplified filtered component.

    enhanced = audio + gain * highpass(audio)

    Parameters
    ----------
    audio : ndarray
        Input waveform (mono).
    sample_rate : int
        Sampling rate in Hz.
    cutoff_hz : float
        High-pass cutoff frequency in Hz.
    gain : float
        Scaling factor applied to the high-pass filtered component.
    order : int
        Butterworth filter order.

    Returns
    -------
    enhanced_audio : ndarray
        Enhanced waveform (float32).
    """
    b, a = design_butterworth_filter(
        cutoff_hz=cutoff_hz,
        sample_rate=sample_rate,
        filter_type="highpass",
        order=order,
    )

    filtered = signal.filtfilt(b, a, audio).astype(np.float32)
    enhanced = audio.astype(np.float32) + (gain * filtered).astype(np.float32)

    # Optional safety: prevent NaNs/Infs
    if not np.isfinite(enhanced).all():
        raise RuntimeError("Enhanced audio contains NaN or Inf values.")

    return enhanced


def write_wav_float32(path: Path, sample_rate: int, audio_f32: np.ndarray) -> None:
    """
    Write a float32 WAV file (scipy.io.wavfile.write supports float32).

    Parameters
    ----------
    path : Path
        Output path.
    sample_rate : int
        Sampling rate in Hz.
    audio_f32 : ndarray
        Audio waveform in float32.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sample_rate, audio_f32.astype(np.float32))


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Butterworth high-pass enhancement to an audio file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input WAV/MP3/FLAC (decoded by librosa).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save enhanced WAV files.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sampling rate used for loading/resampling (default: 16000).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=6,
        help="Butterworth filter order (default: 6).",
    )
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=[600, 1000, 1500, 2000],
        help="One or more cutoff frequencies in Hz (default: 600 1000 1500 2000).",
    )
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=[0.3, 0.6, 1.0, 2.0],
        help="One or more gain coefficients (default: 0.3 0.6 1.0 2.0).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="type1",
        help="Prefix used in output filenames (default: type1).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable peak normalization before processing (default: normalize enabled).",
    )
    return parser.parse_args()


def peak_normalize(audio: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Peak-normalize audio to [-1, 1] based on max absolute amplitude.
    This helps avoid inconsistent scaling across inputs.

    Parameters
    ----------
    audio : ndarray
        Input audio.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    normalized : ndarray
        Peak-normalized audio.
    """
    peak = float(np.max(np.abs(audio)))
    if peak < eps:
        return audio
    return (audio / peak).astype(np.float32)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Load audio (mono) with librosa, resample to target sr
    audio, _ = librosa.load(str(args.input), sr=args.sr, mono=True)

    if audio.size == 0:
        raise RuntimeError("Loaded audio is empty.")

    if not args.no_normalize:
        audio = peak_normalize(audio)

    # Generate outputs
    for cutoff in args.cutoffs:
        for gain in args.gains:
            enhanced = enhance_high_frequency(
                audio=audio,
                sample_rate=args.sr,
                cutoff_hz=float(cutoff),
                gain=float(gain),
                order=int(args.order),
            )

            # Output naming: <stem>_<prefix>_<cutoff>Hz_gain<gain>.wav
            stem = args.input.stem
            out_name = f"{stem}_{args.prefix}_{int(cutoff)}Hz_gain{gain}.wav"
            out_path = args.output_dir / out_name

            write_wav_float32(out_path, args.sr, enhanced)

    print(f"Done. Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
