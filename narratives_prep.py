
import csv
import json
import anthropic # type: ignore
from pathlib import Path
import pandas as pd
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
from claude_api_key import load_vault

bids_dir = Path("/home/NEU480/datasets/narratives")
fetch_audio           = lambda file_name: wavfile.read(bids_dir / "stimuli" / f"{file_name}_audio.wav")
fetch_transcript      = lambda file_name: Path(bids_dir / "stimuli" / "transcripts" / f"{file_name}_transcript.txt").read_text()
fetch_segments        = lambda file_name: json.loads((bids_dir / "stimuli" / "whisperx" / f"{file_name}_audio.json").read_text())["segments"]
fetch_whisperx_transcript = lambda file_name: " ".join(s["text"].strip() for s in fetch_segments(file_name))

def fetch_subject_list(task_name, exclude_subjects=[]):
    subjects = []
    for tsv in sorted(bids_dir.glob("sub-*/sub-*_scans.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        tasks = {f.split("task-")[1].split("_")[0] for f in df["filename"] if "task-" in f}
        if task_name in tasks and tsv.parent.name not in exclude_subjects:
            subjects.append(tsv.parent.name)
    return subjects

out_dir = Path(".")

_CACHE_DIR = Path("/scratch/network/ih2422/narratives_cache")

def load_bold_masked(sub, task, mask_img, standardize=True):
    from nilearn.maskers import NiftiMasker
    _CACHE_DIR.mkdir(exist_ok=True)
    std_tag = "std" if standardize else "nostd"
    cache_path = _CACHE_DIR / f"{sub}_task-{task}_{std_tag}.npy"
    if cache_path.exists():
        print(f"[cache] loading {cache_path.name}")
        return np.load(cache_path)
    bold_path = bids_dir / "derivatives" / f"afni-nosmooth/{sub}/func/{sub}_task-{task}_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz"
    if not bold_path.exists():
        return None
    data = NiftiMasker(mask_img=mask_img, standardize=standardize).fit_transform(str(bold_path))
    np.save(cache_path, data)
    print(f"[cache] saved {cache_path.name}")
    return data

def isc_filter(subjects, task, mask_img, standardize=True, n_sd=2, title=None, plot=True):
    """
    Compute leave-one-out ISC and return (isc_scores, retained_subjects).
    Subjects whose mean whole-brain ISC falls more than n_sd SDs below the
    group mean are flagged as low-attention and excluded from the retained list.
    """
    all_data, loaded = {}, []
    for sub in subjects:
        data = load_bold_masked(sub, task, mask_img, standardize=standardize)
        if data is None:
            print(f"{sub}  — file missing, skipping")
            continue
        all_data[sub] = data
        loaded.append(sub)

    min_trs  = min(d.shape[0] for d in all_data.values())
    data_mat = np.stack([all_data[s][:min_trs] for s in loaded])  # (n_subs, TRs, voxels)

    def _pearson_mean(a, b):
        az = a - a.mean(axis=0);  bz = b - b.mean(axis=0)
        return (az * bz).sum(axis=0) / (
            np.sqrt((az**2).sum(axis=0) * (bz**2).sum(axis=0)) + 1e-8
        )

    isc_scores = {}
    for i, sub in enumerate(loaded):
        others = np.delete(data_mat, i, axis=0).mean(axis=0)
        isc_scores[sub] = _pearson_mean(data_mat[i], others).mean()

    vals   = np.array(list(isc_scores.values()))
    mean   = vals.mean()
    std    = vals.std()
    thresh = mean - n_sd * std

    print("Subject ISC (mean r across voxels):")
    for sub in sorted(isc_scores, key=isc_scores.get):
        flag = "  <-- LOW ISC" if isc_scores[sub] < thresh else ""
        print(f"  {sub}  r={isc_scores[sub]:.4f}{flag}")
    print(f"\nMean={mean:.4f}  SD={std:.4f}  Threshold (mean-{n_sd}SD)={thresh:.4f}")

    excluded = [s for s in loaded if isc_scores[s] < thresh]
    retained = [s for s in loaded if isc_scores[s] >= thresh]
    print(f"Excluded: {excluded}")
    print(f"Retained: {len(retained)}/{len(loaded)} subjects")

    if plot:
        colors = ["#d62728" if isc_scores[s] < thresh else "#1f77b4" for s in loaded]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(loaded, [isc_scores[s] for s in loaded], color=colors)
        ax.axhline(thresh, color="black", linestyle="--", linewidth=1,
                   label=f"Threshold ({thresh:.3f})")
        ax.set_xticklabels(loaded, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean ISC (r)")
        ax.set_title(title or f"{task} — Inter-Subject Correlation")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return isc_scores, retained


def annotate_places(story_name, transcript, valid_places, out_path):
    if out_path.exists():
        print(f"{out_path.name} already exists — skipping API call.")
        with open(out_path) as f:
            return list(csv.DictReader(f, delimiter="\t"))

    # Authenticate with the API key
    api_key = load_vault()
    client = anthropic.Anthropic()
    labels_str = ", ".join(f'"{p}"' for p in valid_places)

    prompt = f"""You are analyzing the story "{story_name}".

Step 1 — Segment: Divide the full transcript into contiguous segments and label each with one of: {labels_str}, or "other".

Use a named place label ONLY when a listener would clearly picture themselves inside that specific location — the scene is grounded there, characters are physically present, and the setting is the obvious foreground of the action. If there is any doubt about where the scene is set, or if the text is transitional, reflective, or could take place anywhere, label it "other".

Step 2 — List all segments in their original story order as a JSON array, including "other" segments. Each element must be:
  {{"place": "<label>", "excerpt": "<one or more full sentences from the transcript that represent this segment>"}}

Rules:
- Do not reorder segments.
- Excerpts must be full sentences taken directly from the transcript. No ellipses.
- The same place label may appear many times if it recurs in the story.
- When in doubt, use "other".

Output ONLY the JSON array, no markdown fences.

Transcript:
{transcript}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": prompt}]
    )

    text = next(b.text for b in response.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    rows = json.loads(text)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["place", "excerpt"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    place_counts = {p: sum(1 for r in rows if r["place"] == p) for p in valid_places + ["other"]}
    print(f"Saved {out_path.name} — {len(rows)} rows: {place_counts}")
    return rows

def annotate_places_old(story_name, transcript, valid_places, out_path):
    if out_path.exists():
        print(f"{out_path.name} already exists — skipping API call.")
        with open(out_path) as f:
            return list(csv.DictReader(f, delimiter="\t"))

    api_key = load_vault()
    client = anthropic.Anthropic()
    labels_str = ", ".join(f'"{p}"' for p in valid_places)

    prompt = f"""You are analyzing the story "{story_name}".

Step 1 — Segment: Divide the full transcript into contiguous segments, each covering a single scene or location. For every segment, assign one of these place labels: {labels_str}, or "other" for anything that does not clearly belong to one of these places.

Step 2 — List all segments in their original story order as a JSON array, including "other" segments. Each element must be:
  {{"place": "<label>", "excerpt": "<one or more full sentences from the transcript that represent this segment>"}}

Rules:
- Do not reorder segments.
- Excerpts must be full sentences taken directly from the transcript. No ellipses.
- The same place label may appear many times if it recurs in the story.

Output ONLY the JSON array, no markdown fences.

Transcript:
{transcript}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": prompt}]
    )

    text = next(b.text for b in response.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    rows = json.loads(text)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["place", "excerpt"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    place_counts = {p: sum(1 for r in rows if r["place"] == p) for p in valid_places + ["other"]}
    print(f"Saved {out_path.name} — {len(rows)} rows: {place_counts}")
    return rows

def align_tsv_to_whisperx(tsv_path, segments, transcript, padding_start=0):
    """
    Match each TSV excerpt (generated from whisperx transcript) to the whisperx
    word list by forward-scanning for the first 4 cleaned words of each excerpt.
    Uses word-level timestamps directly; interpolates for the rare words missing them.
    padding_start unused — whisperx timestamps are already absolute.
    """
    import re

    words = []
    for seg in segments:
        seg_words = seg.get("words", [])
        for i, w in enumerate(seg_words):
            if "start" not in w or "end" not in w:
                prev_t = next((seg_words[j]["end"]   for j in range(i-1, -1, -1) if "end"   in seg_words[j]), seg["start"])
                next_t = next((seg_words[j]["start"] for j in range(i+1, len(seg_words)) if "start" in seg_words[j]), seg["end"])
                w = {**w, "start": (prev_t + next_t) / 2, "end": (prev_t + next_t) / 2}
            clean = re.sub(r"[^a-z0-9']+", "", w["word"].lower())
            if clean:
                words.append({"clean": clean, "start": w["start"]})

    def tokenize(text):
        tokens = re.split(r"\s+", text.strip())
        return [t for t in (re.sub(r"[^a-z0-9']+", "", tok.lower()) for tok in tokens) if t]

    rows_raw = list(csv.DictReader(open(tsv_path), delimiter="\t"))

    # Two-pass matching:
    # Pass 1 — strict 4-word forward sweep (cursor advances only here).
    #   Handles repeated phrases correctly: cursor has already passed earlier
    #   occurrences, so we land on the intended one.
    # Pass 2 — global fallback with shorter keys / word offsets (cursor fixed).
    #   Handles out-of-order TSV rows and minor transcription variants without
    #   corrupting cursor state.
    # Final sort by word-list index gives chronological order.
    cursor = 0
    matched = []

    for row in rows_raw:
        toks = tokenize(row["excerpt"])
        result = None

        if len(toks) >= 4:
            for i in range(cursor, len(words) - 3):
                if all(words[i + j]["clean"] == toks[j] for j in range(4)):
                    result = (i, words[i]["start"])
                    cursor = i + 1
                    break

        if result is None:
            for offset in range(min(3, len(toks))):
                for n in range(min(4, len(toks) - offset), 1, -1):
                    key = toks[offset:offset + n]
                    for i in range(len(words) - n + 1):
                        if all(words[i + j]["clean"] == key[j] for j in range(n)):
                            result = (i, words[i]["start"])
                            break
                    if result:
                        break
                if result:
                    break

        if result:
            matched.append({"place": row["place"], "start": result[1], "_idx": result[0], "excerpt": row["excerpt"]})
        else:
            print(f"  [no match] {row['excerpt'][:60]}")

    matched.sort(key=lambda r: r["_idx"])
    rows = [{"place": r["place"], "start": r["start"], "excerpt": r["excerpt"]} for r in matched]
    for i in range(len(rows) - 1):
        rows[i]["end"] = rows[i + 1]["start"]
    rows[-1]["end"] = segments[-1]["end"]

    for r in rows:
        words_exc = r["excerpt"].split()
        preview = " ".join(words_exc[:5]) + " ... " + " ".join(words_exc[-5:]) if len(words_exc) > 10 else r["excerpt"]
        print(f"{r['start']:7.1f}s  [{r['place']}]  {preview}")

    return rows

def plot_place_line(aligned_rows, title, place_order, place_colors):
    label_to_y = {p: i for i, p in enumerate(place_order)}
    label_to_y["other"] = -1

    fig, ax = plt.subplots(figsize=(14, 3))
    for row in aligned_rows:
        y = label_to_y.get(row["place"], -1)
        color = place_colors.get(row["place"], "#aaaaaa")
        ax.plot([row["start"], row["end"]], [y, y], color=color, linewidth=6, solid_capstyle="butt")

    ax.set_yticks([-1] + list(range(len(place_order))))
    ax.set_yticklabels(["other"] + place_order)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_xlim(left=0, right=aligned_rows[-1]["end"] + 10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
    print(aligned_rows[-1]["end"])