import torch
import soundfile as sf

def load_audio(audio_path):
    """Load audio file and convert to torch tensor."""
    waveform, sample_rate = sf.read(audio_path, dtype='float32')
    # Convert to torch tensor and add channel dimension if needed
    waveform = torch.from_numpy(waveform).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension for mono audio
    elif waveform.ndim == 2:
        waveform = waveform.T  # Transpose to (channel, time) format
    return waveform, sample_rate
