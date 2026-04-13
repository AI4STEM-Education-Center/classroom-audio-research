"""STFT/iSTFT transforms for MeanFlow-TSE.

Vendored from: https://github.com/rikishimizu/MeanFlow-TSE/blob/main/utils/transforms.py
License: MIT

CRITICAL: MeanFlow-TSE uses n_fft=510 (NOT 512). This gives 256 freq bins,
and real+imag concatenated = 512 features, matching UDiT input_dim=512.
Using n_fft=512 would give 514 features and silently produce garbage.
"""

import torch


def stft_torch(signal, n_fft=510, hop_length=128, win_length=510, concat_dim=1):
    """Compute STFT, return real+imag concatenated along freq dimension.

    Args:
        signal: (batch, time) input waveform.
        n_fft: FFT size. Default 510 for MeanFlow-TSE.
        hop_length: Hop length. Default 128.
        win_length: Window length. Default 510.
        concat_dim: Dimension to concatenate real/imag. Default 1 (freq).

    Returns:
        (batch, freq*2, frames) tensor with real and imag parts concatenated.
    """
    window = torch.hann_window(win_length).to(signal.device)
    spec = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    spec_concat = torch.cat([spec.real, spec.imag], dim=concat_dim)
    return spec_concat


def istft_torch(spec_concat, n_fft=510, hop_length=128, win_length=510, length=None):
    """Reconstruct signal from concatenated real+imag STFT.

    Args:
        spec_concat: (batch, freq*2, frames) tensor.
        n_fft: FFT size. Default 510.
        hop_length: Hop length. Default 128.
        win_length: Window length. Default 510.
        length: Desired output length (samples).

    Returns:
        (batch, time) reconstructed waveform.
    """
    freq = n_fft // 2 + 1
    spec_real = spec_concat[:, :freq, :]
    spec_imag = spec_concat[:, freq:, :]
    spec = torch.complex(spec_real, spec_imag)

    window = torch.hann_window(win_length).to(spec.device)
    recon_signal = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )
    return recon_signal
