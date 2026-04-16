def get_diarization_annotation(diarization_output):
    """Extract best available diarization annotation from pipeline output."""
    if hasattr(diarization_output, "exclusive_speaker_diarization"):
        return diarization_output.exclusive_speaker_diarization
    if hasattr(diarization_output, "speaker_diarization"):
        return diarization_output.speaker_diarization
    return diarization_output

def build_speaker_turns(diarization_annotation):
    """Convert diarization annotation to a sorted list of speaker turns."""
    turns = []
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    turns.sort(key=lambda x: x["start"])
    return turns

def segment_overlap(start_a, end_a, start_b, end_b):
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))

def assign_speaker(chunk_start, chunk_end, speaker_turns):
    """Assign speaker by maximum overlap; fallback to nearest turn."""
    best_speaker = None
    best_overlap = 0.0

    for turn in speaker_turns:
        overlap = segment_overlap(chunk_start, chunk_end, turn["start"], turn["end"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    if best_speaker is not None:
        return best_speaker

    if not speaker_turns:
        return "SPEAKER_00"

    mid = (chunk_start + chunk_end) / 2
    nearest = min(
        speaker_turns,
        key=lambda turn: 0.0
        if turn["start"] <= mid <= turn["end"]
        else min(abs(turn["start"] - mid), abs(turn["end"] - mid)),
    )
    return nearest["speaker"]
