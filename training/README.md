- # Physical Audio Adversarial Attack Pipeline

  This repository contains the **training, generation, and evaluation pipeline** for **LaserAdv**, a physical audio adversarial attack that injects perturbations through **laser-induced vibrations** and targets automatic speech recognition (ASR) systems.

  The code focuses on generating **robust physical adversarial audio examples** that remain effective under real-world distortions, including microphone frequency response limitations and laser-channel-specific effects.

  ---

  ## Overview

  This repository supports the following stages of the LaserAdv pipeline:

  1. **Adversarial perturbation generation** against a white-box ASR model
  2. **Physical channel modeling** using room impulse responses (RIRs), including laser-specific RIRs. 
  3. **Playback-and-record evaluation** under real-world conditions
  4. **White-box ASR evaluation** using Mozilla DeepSpeech
  5. **Black-box transfer evaluation** using Whisper (as an example)

  The implementation is primarily based on **TensorFlow 1.x** and **Mozilla DeepSpeech v0.1.0**, following prior work on robust audio adversarial examples.

  ---

  ## Repository Structure

  ```text
  .
  ├── attack.py                     # Main adversarial example generation script
  ├── filterbanks.npy               # Precomputed filterbank features
  ├── make_checkpoint.py            # Convert DeepSpeech .pb model to TF checkpoint
  ├── recognize.py                  # ASR inference using DeepSpeech
  ├── record.py                     # Playback-and-record helper for physical tests
  ├── tf_logits.py                  # TensorFlow graph utilities for ASR logits
  ├── weight_decay_optimizers.py    # Optimizers with weight decay (TF-based)
  ├── whisper_sh.py                 # Black-box transfer evaluation using Whisper
  ├── rir_100/                      # Laser-specific room impulse responses
  └── README.md

## Environment & Dependencies

### Required Software

- TensorFlow **≤ 1.8.0**
- numpy
- scipy
- librosa
- pyaudio

> ⚠️ **Important:**
>  This repository requires **TensorFlow 1.x** due to compatibility with DeepSpeech v0.1.0.

------

## DeepSpeech Preparation

### 1. Clone Mozilla DeepSpeech

Clone DeepSpeech into the same directory level as this repository:

```
git clone https://github.com/mozilla/DeepSpeech.git
```

### 2. Checkout the Required Version

```
cd DeepSpeech
git checkout tags/v0.1.0
cd ..
```

### 3. Download Pretrained DeepSpeech Models

```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
tar -xzf deepspeech-0.1.0-models.tar.gz
```

### 4. Verify Model Integrity

```
md5sum models/output_graph.pb
```

Expected checksum:

```
08a9e6e8dc450007a0df0a37956bc795  output_graph.pb
```

------

## Convert DeepSpeech Model to TensorFlow Checkpoint

The adversarial optimization code requires a TensorFlow checkpoint rather than a frozen `.pb` graph.

```
python make_checkpoint.py
```

This script converts `output_graph.pb` into a TensorFlow checkpoint that can be used by `attack.py`.

------

## Room Impulse Responses (RIRs)

Room impulse responses are used to simulate physical channel effects during adversarial optimization.

All RIR files must be **16 kHz, mono WAV**.

### Laser-Specific RIRs (`rir_100/`)

This repository includes a directory:

```
rir_100/
```

which contains **laser-specific impulse responses** measured or modeled for laser-to-microphone signal injection.

Compared to conventional acoustic RIRs, these responses capture:

- Frequency-selective attenuation of laser-induced signals
- Hardware-dependent coupling between lasers and MEMS microphones
- Physical distortions observed in real laser injection experiments

These RIRs can be directly used during adversarial example generation.

------

## Generating Adversarial Examples

Assume:

- `sample.wav` is a **16 kHz mono** audio file
- `rir_100/` contains laser-specific RIR WAV files

### Targeted Attack Example

Generate adversarial examples recognized as `"hello world"`:

```
mkdir results
python attack.py \
  --in sample.wav \
  --imp rir_100/*.wav \
  --target "hello world" \
  --out results
```

Generated adversarial audio files will be saved in the `results/` directory.

------

## Physical Playback and Recording

To evaluate robustness under real-world conditions, adversarial examples can be played back through a speaker and re-recorded using a microphone.

```
python record.py ae.wav ae_recorded.wav
```

This step simulates over-the-air and laser-based physical attack scenarios.

------

## White-Box ASR Evaluation (DeepSpeech)

Recognize the recorded audio using DeepSpeech:

```
python recognize.py \
  models/output_graph.pb \
  ae_recorded.wav \
  models/alphabet.txt \
  models/lm.binary \
  models/trie
```

The transcription result will be written to:

```
ae_recorded.txt
```

------

## Black-Box Transfer Evaluation (Whisper)

In addition to DeepSpeech, this repository provides an **optional black-box transfer evaluation** using OpenAI Whisper.

### Purpose

- Evaluate cross-model transferability of physical adversarial examples
- Demonstrate that LaserAdv perturbations are not model-specific
- Provide an example of black-box ASR evaluation

> **Note:**
>  Whisper is used **only as an evaluation target**.
>  The attack generation does **not assume any access to Whisper internals**.

### Usage

```
python whisper_sh.py \
  --audio_dir results \
  --model base \
  --target "hello world"
```

This script batch-transcribes adversarial audio files and reports whether the target phrase appears in the transcription.

------

## Code Origin and Licensing

Most of the code in this repository is based on: [hiromu/robust_audio_ae: Robust Audio Adversarial Example for a Physical Attack](https://github.com/hiromu/robust_audio_ae?tab=License-1-ov-file), and is distributed under the **2-clause BSD License**.

Exceptions:

- **recognize.py**
   Based on Mozilla DeepSpeech client code
   Licensed under the **Mozilla Public License v2.0**
- **weight_decay_optimizers.py**
   Taken from TensorFlow
   Licensed under the **Apache License v2.0**

Please refer to individual file headers for detailed licensing information.

------

## Intended Use

This repository is intended **solely for academic research and security evaluation**, including:

- Studying laser-based physical audio adversarial attacks
- Evaluating ASR robustness under physical channel distortions
- Reproducing results from the LaserAdv project

It is **not intended** for misuse or deployment in production systems.