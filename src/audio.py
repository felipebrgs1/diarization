import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F


def load_audio(audio_path, target_sample_rate=16000):
    """
    Load audio file, resample to target_sample_rate, convert to mono, 
    and apply normalization.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 1. Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 2. Resample
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
        
    # 3. Normalization (Peak normalization to -1dB)
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val * 0.9
        
    # 4. Simple Bandpass Filter (Optional but good for voice)
    # Removing frequencies below 80Hz and above 8000Hz
    waveform = F.highpass_biquad(waveform, sample_rate, 80.0)
    waveform = F.lowpass_biquad(waveform, sample_rate, 8000.0)

    return waveform, sample_rate

