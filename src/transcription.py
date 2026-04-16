from src.diarization import assign_speaker

def build_segments(chunks, speaker_turns):
    """Create speaker-attributed text segments from ASR chunks."""
    segments = []
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "").strip()
        timestamp = chunk.get("timestamp")

        if not chunk_text or not timestamp or timestamp[0] is None:
            continue

        chunk_start = float(timestamp[0])
        chunk_end = timestamp[1]

        if chunk_end is None:
            next_start = None
            if idx + 1 < len(chunks):
                next_ts = chunks[idx + 1].get("timestamp")
                if next_ts:
                    next_start = next_ts[0]
            chunk_end = float(next_start) if next_start is not None else chunk_start + 0.5
        else:
            chunk_end = float(chunk_end)

        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 0.01

        speaker = assign_speaker(chunk_start, chunk_end, speaker_turns)
        segments.append({
            "start": chunk_start,
            "end": chunk_end,
            "speaker": speaker,
            "text": chunk_text,
        })

    return segments

def merge_consecutive_segments(segments, max_gap_seconds=0.4):
    """Merge adjacent segments of same speaker when time gap is short."""
    merged = []
    for seg in segments:
        if (
            merged
            and merged[-1]["speaker"] == seg["speaker"]
            and (seg["start"] - merged[-1]["end"]) <= max_gap_seconds
        ):
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merged[-1]["text"] = f"{merged[-1]['text']} {seg['text']}".strip()
        else:
            merged.append(seg.copy())
    return merged
