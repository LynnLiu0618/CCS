"""Window-local BJTU spectral and bearing-envelope features."""

import numpy as np
import torch


@torch.inference_mode()
def extract_features(windows, device='cpu'):
    windows = np.asarray(windows)
    if windows.ndim != 3 or windows.shape[1:] != (2048, 18) or not np.isfinite(windows).all():
        raise ValueError('Expected finite synchronized windows with shape (N, 2048, 18).')
    if not len(windows):
        return np.empty((0, 2628), dtype=np.float32)
    hann = torch.hann_window(2048, device=device, dtype=torch.float64)
    result = []
    for start in range(0, len(windows), 64):
        wave = torch.as_tensor(windows[start:start + 64], device=device, dtype=torch.float64)
        triads = wave.transpose(1, 2).reshape(-1, 6, 3, 2048)
        triads = triads - triads.mean(-1, keepdim=True)
        triads = triads / triads.square().mean((-1, -2), keepdim=True).sqrt().clamp_min(1e-8)
        spectrum = torch.fft.rfft(triads * hann, dim=-1)[..., 1:1025]
        bearing_spectrum = spectrum[:, 5]
        power = spectrum.abs().square() / 768.
        bands = power.reshape(-1, 6, 3, 64, 16).mean(-1).log1p().flatten(2)
        grouped = spectrum.reshape(-1, 6, 3, 16, 64)
        correlations = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            cross = (grouped[:, :, left] * grouped[:, :, right].conj()).mean(-1)
            denominator = (grouped[:, :, left].abs().square().mean(-1)
                           * grouped[:, :, right].abs().square().mean(-1)).sqrt().clamp_min(1e-8)
            correlations.extend([(cross / denominator).real, (cross / denominator).imag])
        mean_square = triads.square().mean(-1).clamp_min(1e-8)
        moments = torch.stack([
            mean_square.log(), triads.abs().mean(-1) / mean_square.sqrt(),
            (triads.pow(4).mean(-1) / mean_square.square()).clamp_max(100).log1p(),
            (triads.abs().amax(-1) / mean_square.sqrt()).log1p(),
        ], dim=-1).flatten(2)
        base = torch.cat([bands, torch.cat(correlations, dim=-1), moments], dim=2).flatten(1)

        # Bearing channels are the sixth triad. Reuse its normalization and FFT.
        spectrum = bearing_spectrum
        envelopes = []
        for low, high in ((0, 4000), (4000, 8000), (8000, 16000), (16000, 32000)):
            analytic = torch.zeros((*spectrum.shape[:-1], 2048), dtype=torch.complex128, device=device)
            left, right = max(1, int(low / 31.25)), min(1024, int(high / 31.25))
            analytic[..., left:right] = 2 * spectrum[..., left - 1:right - 1]
            envelope = torch.fft.ifft(analytic, dim=-1).abs()
            energy = envelope.square().mean(-1, keepdim=True)
            normalized = envelope / energy.sqrt().clamp_min(1e-8)
            centered = normalized - normalized.mean(-1, keepdim=True)
            modulation = torch.fft.rfft(centered * hann, dim=-1)[..., 1:65].abs().square() / 768.
            statistics = torch.cat([
                energy.log1p(), normalized.mean(-1, keepdim=True), centered.square().mean(-1, keepdim=True),
                normalized.pow(4).mean(-1, keepdim=True).clamp_max(100).log1p(),
                normalized.max(-1, keepdim=True).values.log1p(),
            ], dim=-1)
            envelopes.append(torch.cat([modulation.log1p(), statistics], dim=-1))
        extra = torch.stack(envelopes, dim=2).flatten(1)
        result.append(torch.cat([base, extra], dim=1).float().cpu().numpy())
    return np.concatenate(result)
