# Adversarial Attacks on Black-box Speech Recognition Systems via Laser Injection

This repository contains the **complete research artifact** for **LaserAdv**, a physical audio adversarial attack that injects carefully crafted perturbations through **laser-induced vibrations** and targets automatic speech recognition (ASR) systems equipped with MEMS microphones.

The project integrates **training pipelines**, **frequency-domain enhancement methods**, and **physical-world evaluation assets** to enable reproducible research on laser-based audio attacks.

---

## Project Overview

LaserAdv addresses key challenges in physical audio adversarial attacks, including:

- Limited sensitivity of microphones to laser-induced signals
- Frequency-selective fading (FSF) introduced by laser channels
- Robustness under physical playback and recording

This repository provides:

- A **training and generation pipeline** 
- A **new enhancement module (SAE-TFI)** 
- A curated **audio dataset** for training and evaluation
- A **demonstration video** showing real-world attack execution

---

## Repository Structure

```text
.
├── training/          # Adversarial example training and generation 
├── SAE-TFI/           # Selective Amplitude Enhancement via Time-Frequency Interconversion
├── wavs/              # Training audio set (60 samples)
├── AttackVideo.mp4    # Physical attack demonstration video
├── README.md
└── LICENSE
```

## training/

The `training/` directory contains the **core adversarial example generation and training pipeline**.

This component:

- Implements white-box adversarial optimization against ASR models
- Incorporates physical channel modeling via room impulse responses (RIRs)
- Serves as the foundation upon which LaserAdv builds

------

## SAE-TFI/

The `SAE-TFI/` directory (Selective Amplitude Enhancement based on Time-Frequency Interconversion) is designed to mitigate **frequency-selective fading (FSF)** introduced by laser-based physical channels.

Key characteristics:

- Operates in the STFT domain
- Selectively amplifies high-frequency components
- Preserves phase information during reconstruction
- Compensates for laser-induced high-frequency attenuation

------

## wavs/

The `wavs/` directory contains **60 audio samples** used for training and evaluation.

These samples are intentionally diverse, covering:

- Different spoken contents
- Multiple speakers
- Varying loudness and recording conditions

The dataset is designed to improve robustness and generalization of adversarial perturbations under realistic conditions.

> Note: These audio files are provided for research and reproducibility purposes.

------

## AttackVideo.mp4

`AttackVideo.mp4` is a **demonstration video** showcasing the LaserAdv attack in a real physical setting.

The video illustrates:

- Laser-based audio injection
- Over-the-air signal capture by a target device
- Successful command injection under physical constraints

This video serves as qualitative evidence of attack feasibility.

------

## License

This project is released under the **BSD 2-Clause License**, inherited from prior work on robust audio adversarial examples.

- The `training/` directory contains inherited code under the BSD 2-Clause License
- Newly developed components (e.g., SAE-TFI) are released under the same license for simplicity and compatibility

See the `LICENSE` file for full details.

------

## Intended Use

This repository is intended **solely for academic research and security evaluation**, including:

- Studying laser-based physical audio adversarial attacks
- Evaluating robustness of ASR systems under physical distortions
- Reproducing experimental results reported in the LaserAdv project

It is **not intended** for misuse or deployment in real-world systems.

------

## Citation

If you use this repository in academic work, please cite the corresponding **LaserAdv** paper.

```
@inproceedings{zhang2024laseradv,
  title={$\{$LaserAdv$\}$: Laser adversarial attacks on speech recognition systems},
  author={Zhang, Guoming and Ma, Xiaohui and Zhang, Huiting and Xiang, Zhijie and Ji, Xiaoyu and Yang, Yanni and Cheng, Xiuzhen and Hu, Pengfei},
  booktitle={33rd USENIX Security Symposium (USENIX Security 24)},
  pages={3945--3961},
  year={2024}
}
```