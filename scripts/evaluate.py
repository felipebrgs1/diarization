import re
from pathlib import Path
import jiwer

TRANSCRIPTION_DIR = Path("audio/transcription")
GROUND_TRUTH_DIR = Path("audio/ground_truth")

def parse_md(path: Path):
    """Parses segments from markdown file."""
    if not path.exists():
        return []
    
    segments = []
    # Regex to match [MM:SS] SPEAKER: Text
    pattern = re.compile(r"\*\*\[(\d{2}:\d{2})\] (.*?):\*\*  \n(.*)")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    matches = pattern.findall(content)
    for m in matches:
        segments.append({
            "time": m[0],
            "speaker": m[1].strip(),
            "text": m[2].strip()
        })
    return segments

def evaluate_file(gt_path: Path, pred_path: Path):
    gt_segments = parse_md(gt_path)
    pred_segments = parse_md(pred_path)
    
    if not gt_segments:
        print(f"  [!] Arquivo de referência vazio ou não encontrado: {gt_path.name}")
        return None

    if not pred_segments:
        print(f"  [!] Arquivo gerado vazio ou não encontrado: {pred_path.name}")
        return None

    # 1. Text Accuracy (WER)
    gt_text = " ".join([s["text"] for s in gt_segments])
    pred_text = " ".join([s["text"] for s in pred_segments])
    
    wer = jiwer.wer(gt_text, pred_text)
    mer = jiwer.mer(gt_text, pred_text) # Match Error Rate
    
    # 2. Speaker Attribution Accuracy
    # Note: This is a simplified version. For complex diarization, we'd use DER.
    # We try to map predicted labels to GT labels by frequency/overlap.
    correct_speaker_assignments = 0
    total_segments = min(len(gt_segments), len(pred_segments))
    
    for i in range(total_segments):
        # Very basic check: do they share same index? 
        # In a real scenario we'd use time overlap.
        if gt_segments[i]["speaker"] == pred_segments[i]["speaker"]:
            correct_speaker_assignments += 1
            
    speaker_acc = (correct_speaker_assignments / total_segments) if total_segments > 0 else 0
    
    return {
        "name": gt_path.stem,
        "wer": wer,
        "mer": mer,
        "speaker_acc": speaker_acc,
        "segments_count": len(pred_segments),
        "gt_count": len(gt_segments)
    }

def main():
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    
    gt_files = list(GROUND_TRUTH_DIR.glob("*.md"))
    if not gt_files:
        print(f"\nNenhum arquivo de referência encontrado em {GROUND_TRUTH_DIR}")
        print("Coloque arquivos .md no formato:")
        print("**[00:00] SPEAKER_00:**  \nTexto da transcrição")
        return

    print(f"\nIniciando Benchmark ({len(gt_files)} arquivos)...")
    print("-" * 60)
    
    results = []
    for gt_file in gt_files:
        pred_file = TRANSCRIPTION_DIR / gt_file.name
        res = evaluate_file(gt_file, pred_file)
        if res:
            results.append(res)
            print(f"File: {res['name']}")
            print(f"  WER: {res['wer']:.2%} (Erro de Palavra - Menor é melhor)")
            print(f"  Speaker Acc: {res['speaker_acc']:.2%} (Igualdade de locutor por segmento)")
            print(f"  Segments: {res['segments_count']} (Pred) / {res['gt_count']} (GT)")
            print("-" * 30)

    if results:
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_spk = sum(r["speaker_acc"] for r in results) / len(results)
        print("\nRESULTADO MÉDIO:")
        print(f"  Média WER: {avg_wer:.2%}")
        print(f"  Média Speaker Acc: {avg_spk:.2%}")

if __name__ == "__main__":
    main()
