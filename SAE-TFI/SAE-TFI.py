#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STFT-Based High-Frequency Magnitude Enhancement

This script enhances high-frequency components of an audio signal
in the STFT domain by amplifying magnitude values above a cutoff
frequency while preserving the original phase.

Typical use cases:
- Frequency-domain perturbation
- Spectral manipulation experiments
- Audio robustness and sensitivity analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile


# =========================
# Visualization Utilities
# =========================

def plot_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 64,
    save_path: Path | None = None,
) -> None:
    """
    Plot and optionally save a linear-frequency STFT spectrogram.

    Parameters
    ----------
    audio : ndarray
        Input audio signal.
    sample_rate : int
        Sampling rate in Hz.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length.
    save_path : Path or None
        If provided, save the figure to this path.
    """
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spec_db,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()
    plt.close()


# =========================
# Core Processing Logic
# =========================

def enhance_stft_magnitude(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    gain: float,
    n_fft: int = 512,
    hop_length: int = 64,
    window: str = "hann",
) -> np.ndarray:
    """
    Amplify STFT magnitude above a cutoff frequency while keeping phase unchanged.

    enhanced_mag[f >= cutoff] = mag[f] * gain

    Parameters
    ----------
    audio : ndarray
        Input waveform (mono).
    sample_rate : int
        Sampling rate in Hz.
    cutoff_hz : float
        Frequency threshold for enhancement.
    gain : float
        Magnitude scaling factor.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length.
    window : str
        STFT window type.

    Returns
    -------
    enhanced_audio : ndarray
        Time-domain reconstructed waveform (float32).
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Map cutoff frequency to STFT bin index
    nyquist = sample_rate / 2.0
    cutoff_bin = int(magnitude.shape[0] * cutoff_hz / nyquist)
    cutoff_bin = np.clip(cutoff_bin, 0, magnitude.shape[0] - 1)

    magnitude[cutoff_bin:, :] *= gain

    enhanced_stft = magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=hop_length,
        window=window,
    )

    return enhanced_audio.astype(np.float32)


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STFT-based high-frequency magnitude enhancement."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input audio file (WAV/MP3/FLAC).",
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
        help="Target sampling rate (default: 16000).",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=512,
        help="STFT FFT size (default: 512).",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=64,
        help="STFT hop length (default: 64).",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        help="STFT window function (default: hann).",
    )
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=[600, 1000, 1500, 2000],
        help="Cutoff frequencies in Hz.",
    )
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=[1.3, 1.6, 2.0, 3.0],
        help="Magnitude gain coefficients.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    audio, _ = librosa.load(str(args.input), sr=args.sr, mono=True)

    for cutoff in args.cutoffs:
        for gain in args.gains:
            enhanced = enhance_stft_magnitude(
                audio=audio,
                sample_rate=args.sr,
                cutoff_hz=cutoff,
                gain=gain,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                window=args.window,
            )

            out_name = (
                f"{args.input.stem}_type2_{int(cutoff)}Hz_gain{gain}.wav"
            )
            out_path = args.output_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            wavfile.write(out_path, args.sr, enhanced)

    print(f"Processing completed. Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
