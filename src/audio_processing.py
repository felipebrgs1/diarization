import subprocess
from pathlib import Path

def preprocess_audio(input_path: Path, output_path: Path) -> bool:
    """
    Applies telephony-optimized filters:
    - Resample to 16kHz mono
    - Bandpass: 200Hz - 3400Hz
    - Denoise: afftdn
    - Normalization: loudnorm
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1",
        "-af", "highpass=f=200,lowpass=f=3400,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing {input_path.name}: {e.stderr.decode()}")
        return False
