# Physical Adversarial Perturbation for Laser-Based Audio Injection

This repository provides a **reproducible implementation of physical adversarial perturbation enhancement methods** used in **LaserAdv**, focusing on mitigating real-world distortions introduced by laser-based audio injection channels. 

Specifically, the code addresses two critical challenges identified in our feasibility analysis:

1. **Low sensitivity of MEMS microphones**, which limits the received perturbation amplitude.
2. **Frequency-Selective Fading (FSF)** in laser channels, which disproportionately attenuates high-frequency components.

These two issues correspond to **two complementary perturbation enhancement strategies** implemented in this repository.

---

## Paper Context: Physical Adversarial Perturbation in LaserAdv

According to our analysis, two kinds of distortion must be carefully considered when designing physical adversarial perturbations for LaserAdv:

### Dealing with Low Sensitivity

Some MEMS microphones exhibit low sensitivity to laser-induced vibrations, resulting in adversarial perturbations with insufficient amplitude. Since perturbation intensity is a key factor in attack success, reduced amplitude leads to a lower attack success rate.

To address this issue, LaserAdv constrains the perturbation amplitude during generation. The upper bound parameter is determined by the device’s frequency response, ensuring that even small-amplitude perturbations remain effective.
A lower bound is also introduced to avoid overly restrictive constraints.

**Corresponding implementation:**  
➡️ *Time-domain high-pass filtering and amplitude enhancement*  
(see **Method I** below)

---

### Dealing with FSF Channel

The laser channel introduces **Frequency-Selective Fading (FSF)**, where high-frequency components experience significantly stronger attenuation than low-frequency components. This distortion is fundamentally different from ambient noise or uniform attenuation and cannot be mitigated by adding random noise.

To compensate for this effect, LaserAdv proposes **Selective Amplitude Enhancement based on Time-Frequency Interconversion (SAE-TFI)**. By operating in the STFT domain, high-frequency magnitude components are selectively amplified
while preserving phase information.

**Corresponding implementation:**  
➡️ *STFT-domain selective amplitude enhancement*  
(see **Method II** below)

---

## Repository Structure

```text
.
├── scripts/
│   ├── Butterworth.py          	# Method I: Time-domain enhancement
│   └── SAE-TFI.py     				# Method II: SAE-TFI (STFT-domain)
├── requirements.txt
├── README.md
└── LICENSE
```



## Installation

We recommend using a virtual environment.

```
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.8
- NumPy
- SciPy
- Librosa
- SoundFile
- Matplotlib (optional, for visualization)

------

# Method I: Butterworth (Time-Domain High-Pass Enhancement)

This method enhances adversarial perturbations in the **time domain** by isolating high-frequency components using a Butterworth high-pass filter, amplifying them, and adding them back to the original waveform.

### Usage

```
python Butterworth.py \
  --input data/input.wav \
  --output_dir outputs/type1 \
  --sr 16000 \
  --order 6 \
  --cutoffs 600 1000 1500 2000 \
  --gains 0.3 0.6 1.0 2.0
```

### Output Naming

```
<input_stem>_type1_<cutoff>Hz_gain<gain>.wav
```

### Method Overview

Given an input signal $x(t)$, the enhanced signal is computed as:
$$
y(t) = x(t) + \alpha \cdot \text{HPF}(x(t))
$$
where:

- $\text{HPF}(\cdot)$ is a Butterworth high-pass filter
- $\alpha$ is a gain coefficient

This approach compensates for reduced microphone sensitivity by reinforcing high-frequency perturbation components.

------

# Method II: SAE-TFI (STFT-Based Selective Amplitude Enhancement)

This method implements **Selective Amplitude Enhancement based on
 Time-Frequency Interconversion (SAE-TFI)** to mitigate FSF-induced distortion.

### Usage

```
python SAE-TFI.py \
  --input data/input.wav \
  --output_dir outputs/type2 \
  --sr 16000 \
  --n_fft 512 \
  --hop_length 64 \
  --cutoffs 600 1000 1500 2000 \
  --gains 1.3 1.6 2.0 3.0
```

### Output Naming

```
<input_stem>_type2_<cutoff>Hz_gain<gain>.wav
```

### Method Overview

Let δ denote the generated perturbation. The enhancement process is:

1. **STFT**

$$
S(\delta) = \text{STFT}\{\delta(t)\}(\tau, \omega)
$$

1. **Magnitude & Phase Decomposition**

$$
\text{Amp} = |S(\delta)|,\quad
\text{Phase} = \angle S(\delta)
$$

1. **Selective Amplitude Enhancement**

$$
\hat{\text{Amp}} =
\begin{cases}
\text{coef} \cdot \text{Amp}, & f > f_c \\
\text{Amp}, & \text{otherwise}
\end{cases}
$$

1. **Inverse STFT**

$$
\hat{\delta} = \text{iSTFT}(\hat{\text{Amp}} \cdot e^{j\cdot \text{Phase}})
$$

This method alleviates high-frequency attenuation caused by FSF while preserving temporal alignment and phase consistency.

------

## Notes on Reproducibility

- All outputs are written as **float32 WAV**
- No randomness is involved in either pipeline
- Phase is strictly preserved in SAE-TFI
- Filtering uses deterministic STFT / iSTFT parameters

------

## Intended Use

This code is intended for **research and experimental purposes**, including:

- Physical adversarial audio attacks
- Laser-based injection robustness analysis
- Security and signal processing research

It is **not intended** for perceptual audio enhancement or production audio processing.

------

## License

This project is released under the **MIT License**.
 See the `LICENSE` file for details.