"""moe-speech-plus のフィルタリング統計と filelist 生成
usage:
  python vast_build_filelist.py stats                 # 閾値候補ごとの残存時間を集計
  python vast_build_filelist.py build MOS AGREE       # filelist.txt / holdout を生成 (例: build 2.5 0.90)
"""
import sys, json, unicodedata, re
from pathlib import Path
from difflib import SequenceMatcher
from multiprocessing import Pool

ROOT = Path('/data/ms/data')
OUT_DIR = Path('/data/StableTTSEx/filelists')
HOLDOUT_N = 5
MIN_DUR, MAX_DUR = 1.0, 15.0

_strip_re = re.compile(r'[^ぁ-んァ-ヴー一-龯a-zA-Z0-9]')

def norm(s):
    return _strip_re.sub('', unicodedata.normalize('NFKC', s or ''))

def scan_speaker(spk_dir):
    """1話者分の json を読み、(uuid, [utterance dict]) を返す"""
    rows = []
    for j in sorted(spk_dir.glob('wav/*.json')):
        wav = j.with_suffix('.wav')
        if not wav.exists():
            continue
        try:
            d = json.loads(j.read_text(encoding='utf-8'))
        except Exception:
            continue
        aw = d.get('anime_whisper_transcription') or ''
        pk = d.get('parakeet_jp_transcription') or ''
        na, np_ = norm(aw), norm(pk)
        if not na:
            agree = 0.0
        else:
            agree = SequenceMatcher(None, na, np_).ratio()
        rows.append({
            'wav': str(wav),
            'text': aw.strip(),
            'dur': float(d.get('duration') or 0),
            'mos': float(d.get('speechMOS') or 0),
            'agree': agree,
        })
    return spk_dir.name, rows

def load_all():
    spk_dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    with Pool(32) as pool:
        return dict(pool.map(scan_speaker, spk_dirs))

def passes(r, mos_th, agree_th):
    return (MIN_DUR <= r['dur'] <= MAX_DUR and r['mos'] >= mos_th
            and r['agree'] >= agree_th and len(r['text']) >= 2)

def main():
    mode = sys.argv[1]
    data = load_all()
    n_spk = len(data)
    all_rows = [r for rows in data.values() for r in rows]
    total_h = sum(r['dur'] for r in all_rows) / 3600
    print(f'speakers={n_spk} utterances={len(all_rows)} total={total_h:.1f}h')

    if mode == 'stats':
        import statistics
        moss = sorted(r['mos'] for r in all_rows)
        print('speechMOS deciles:', [round(moss[int(len(moss)*q/10)], 2) for q in range(10)])
        ags = sorted(r['agree'] for r in all_rows)
        print('agreement deciles:', [round(ags[int(len(ags)*q/10)], 3) for q in range(10)])
        for mos_th in (2.0, 2.3, 2.5, 2.8, 3.0):
            for ag_th in (0.85, 0.90, 0.95):
                kept = [r for r in all_rows if passes(r, mos_th, ag_th)]
                h = sum(r['dur'] for r in kept) / 3600
                print(f'  MOS>={mos_th} agree>={ag_th}: {len(kept):7d} utts {h:7.1f}h')
        return

    mos_th, ag_th = float(sys.argv[2]), float(sys.argv[3])
    # ホールドアウト: 発話数中央値付近の5話者（学習から除外して評価に使う）
    sizes = sorted(data.items(), key=lambda kv: len(kv[1]))
    mid = len(sizes) // 2
    holdout = [uuid for uuid, _ in sizes[mid:mid + HOLDOUT_N]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'holdout_speakers.txt').write_text('\n'.join(holdout) + '\n', encoding='utf-8')

    kept, kept_h = [], 0.0
    for uuid, rows in data.items():
        if uuid in holdout:
            continue
        for r in rows:
            if passes(r, mos_th, ag_th):
                kept.append(f"{r['wav']}|{r['text']}")
                kept_h += r['dur']
    with open(OUT_DIR / 'filelist.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(kept) + '\n')
    print(f'holdout: {holdout}')
    print(f'filelist.txt: {len(kept)} utts, {kept_h/3600:.1f}h (MOS>={mos_th}, agree>={ag_th})')
    print('BUILD DONE')

if __name__ == '__main__':
    main()
