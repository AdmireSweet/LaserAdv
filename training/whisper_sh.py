#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper Batch Transcription Evaluation

This script uses OpenAI Whisper to batch-transcribe WAV files in a directory
and evaluate attack success based on transcription results.

Typical use cases:
- Physical adversarial audio attack evaluation
- Robustness analysis of ASR systems
- Artifact evaluation and reproducibility
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import whisper


# =========================
# Utility Functions
# =========================

def normalize_text(text: str) -> str:
    """
    Normalize transcription text for robust comparison.

    Parameters
    ----------
    text : str
        Raw transcription output.

    Returns
    -------
    normalized : str
        Lowercased and stripped text.
    """
    return text.strip().lower()


def evaluate_transcription(
    text: str,
    target: str = "go home",
) -> Tuple[bool, bool]:
    """
    Evaluate transcription result.

    Parameters
    ----------
    text : str
        Transcribed text.
    target : str
        Target command phrase.

    Returns
    -------
    success : bool
        Whether the transcription exactly matches the target.
    contains_target : bool
        Whether the transcription contains the target keyword.
    """
    text_norm = normalize_text(text)
    target_norm = normalize_text(target)

    success = text_norm in {
        target_norm,
        f"{target_norm}!",
        f"{target_norm}.",
    }
    contains_target = target_norm in text_norm

    return success, contains_target


# =========================
# Core Evaluation Logic
# =========================

def run_batch_evaluation(
    audio_dir: Path,
    model_name: str,
    prefix: str,
    target_phrase: str,
    index_range: range,
    output_file: Path,
) -> None:
    """
    Run batch Whisper transcription and evaluation.

    Parameters
    ----------
    audio_dir : Path
        Directory containing WAV files.
    model_name : str
        Whisper model name (e.g., base, small, medium).
    prefix : str
        Filename prefix pattern (e.g., "{i}syn").
    target_phrase : str
        Target transcription phrase.
    index_range : range
        Range of indices to evaluate.
    output_file : Path
        Path to output result log file.
    """
    model = whisper.load_model(model_name)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a", encoding="utf-8") as f:
        for i in index_range:
            success_count = 0
            fail_count = 0
            partial_count = 0

            for wav_path in sorted(audio_dir.glob(f"{i}{prefix}*.wav")):
                result = model.transcribe(str(wav_path))
                text = result.get("text", "")

                success, contains_target = evaluate_transcription(
                    text=text,
                    target=target_phrase,
                )

                if success:
                    success_count += 1
                elif contains_target:
                    partial_count += 1
                else:
                    fail_count += 1

                print(
                    f"[{wav_path.name}] "
                    f"-> \"{text}\" "
                    f"(success={success}, contains={contains_target})"
                )

            f.write(
                f"index={i}\t"
                f"success={success_count}\t"
                f"partial={partial_count}\t"
                f"fail={fail_count}\n"
            )

            print(
                f"[Summary index={i}] "
                f"success={success_count}, "
                f"partial={partial_count}, "
                f"fail={fail_count}"
            )


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of adversarial audio using Whisper ASR."
    )
    parser.add_argument(
        "--audio_dir",
        type=Path,
        required=True,
        help="Directory containing WAV files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="Whisper model name (default: base).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="syn",
        help="Filename prefix after index (default: syn).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="go home",
        help="Target transcription phrase.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start index (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=50,
        help="End index (inclusive).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/record.txt"),
        help="Output log file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {args.audio_dir}")

    run_batch_evaluation(
        audio_dir=args.audio_dir,
        model_name=args.model,
        prefix=args.prefix,
        target_phrase=args.target,
        index_range=range(args.start, args.end + 1),
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
