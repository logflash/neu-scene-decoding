import csv
import json
import types
import sys
import anthropic
from pathlib import Path
import pandas as pd
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
from claude_api_key import load_vault

# brainiak uses mpi4py at import time; stub it out for single-node use
# ATTRIBUTION: Claude Code wrote this to help me fix SRM dependency issues.
if 'mpi4py' not in sys.modules:
    _mpi4py = types.ModuleType('mpi4py')
    _MPI    = types.ModuleType('mpi4py.MPI')
    class _Comm:
        def Get_size(self):                     return 1
        def Get_rank(self):                     return 0
        def bcast(self, obj, root=0):           return obj
        def Barrier(self):                      pass
        def allgather(self, obj):               return [obj]
        def allreduce(self, obj, op=None):      return obj
        def reduce(self, obj, op=None, root=0): return obj
    _MPI.COMM_SELF  = _Comm() # type: ignore
    _MPI.COMM_WORLD = _Comm() # type: ignore
    _MPI.SUM = 'SUM' # type: ignore
    _MPI.MIN = 'MIN' # type: ignore
    _mpi4py.MPI = _MPI # type: ignore
    sys.modules['mpi4py']     = _mpi4py
    sys.modules['mpi4py.MPI'] = _MPI

# The rest of this code (except for the prompts for Claude) were written by me first.
# I used Claude to edit JUST the docstrings, to make them more consistent.

# BIDS directory
bids_dir = Path("/home/NEU480/datasets/narratives")

# Quick functions to fetch particular files from BIDS
fetch_audio               = lambda file_name: wavfile.read(bids_dir / "stimuli" / f"{file_name}_audio.wav")
fetch_transcript          = lambda file_name: Path(bids_dir / "stimuli" / "transcripts" / f"{file_name}_transcript.txt").read_text()
fetch_segments            = lambda file_name: json.loads((bids_dir / "stimuli" / "whisperx" / f"{file_name}_audio.json").read_text())["segments"]
fetch_whisperx_transcript = lambda file_name: " ".join(s["text"].strip() for s in fetch_segments(file_name))

def fetch_subject_list(task_name, exclude_subjects=[]):
    """Return subject IDs from the BIDS directory that have data for a given task.

    Parameters:
    ----------
    task_name: str
        Task identifier (e.g., "tunnel").
    exclude_subjects: list of str
        Subject IDs to skip (e.g., ["sub-004", "sub-013"]).

    Return:
    ------
    subjects: list of str
        Sorted list of subject IDs with valid data for the task.
    """
    subjects = []
    for tsv in sorted(bids_dir.glob("sub-*/sub-*_scans.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        tasks = {f.split("task-")[1].split("_")[0] for f in df["filename"] if "task-" in f}
        if task_name in tasks and tsv.parent.name not in exclude_subjects:
            subjects.append(tsv.parent.name)
    return subjects

out_dir = Path(".")

_CACHE_DIR = Path("/scratch/network/ih2422/narratives_cache")
_VERBOSE = False

def load_bold_masked(sub, task, mask_img, standardize=True):
    """Load and mask BOLD data for one subject, with disk caching.

    Parameters:
    ----------
    sub: str
        Subject ID (e.g., "sub-001").
    task: str
        Task identifier (e.g., "tunnel").
    mask_img: Nifti1Image
        Brain mask applied during NiftiMasker extraction.
    standardize: bool
        If True, z-score each voxel's timeseries across TRs.

    Return:
    ------
    data: np.array [n_trs, n_voxels] or None
        Masked BOLD data, or None if the BOLD file does not exist.
    """
    from nilearn.maskers import NiftiMasker
    _CACHE_DIR.mkdir(exist_ok=True)
    std_tag = "std" if standardize else "nostd"
    cache_path = _CACHE_DIR / f"{sub}_task-{task}_{std_tag}.npy"
    if cache_path.exists():
        if _VERBOSE:
            print(f"[cache] loading {cache_path.name}")
        return np.load(cache_path)
    bold_path = bids_dir / "derivatives" / f"afni-nosmooth/{sub}/func/{sub}_task-{task}_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz"
    if not bold_path.exists():
        return None
    data = NiftiMasker(mask_img=mask_img, standardize=standardize).fit_transform(str(bold_path))
    np.save(cache_path, data)
    if _VERBOSE:
        print(f"[cache] saved {cache_path.name}")
    return data

def isc_filter(subjects, task, mask_img, standardize=True, n_sd=2, title=None, plot=True):
    """Compute leave-one-out ISC and return scores and retained subjects.

    Subjects whose mean whole-brain ISC falls more than n_sd SDs below the
    group mean are excluded. Results are cached to disk.

    Parameters:
    ----------
    subjects: list of str
        Subject IDs to include.
    task: str
        Task identifier (e.g., "tunnel").
    mask_img: Nifti1Image
        Brain mask passed to load_bold_masked.
    standardize: bool
        If True, z-score BOLD data across TRs before computing ISC.
    n_sd: float
        Number of SDs below the group mean used as the exclusion threshold.
    title: str or None
        Plot title; defaults to "{task} — Inter-Subject Correlation".
    plot: bool
        If True, display a bar chart of per-subject ISC scores.

    Return:
    ------
    isc_scores: dict {str: float}
        Mean whole-brain ISC for each subject.
    retained: list of str
        Subject IDs whose ISC is at or above the exclusion threshold.
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    std_tag    = "std" if standardize else "nostd"
    cache_path = _CACHE_DIR / f"isc_task-{task}_{std_tag}.json"

    if cache_path.exists():
        if _VERBOSE:
            print(f"[cache] loading {cache_path.name}")
        with open(cache_path) as f:
            isc_scores = json.load(f)
        loaded = list(isc_scores.keys())
    else:
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
            isc_scores[sub] = float(_pearson_mean(data_mat[i], others).mean())

        with open(cache_path, "w") as f:
            json.dump(isc_scores, f)
        if _VERBOSE:
            print(f"[cache] saved {cache_path.name}")

    vals   = np.array(list(isc_scores.values()))
    mean   = vals.mean()
    std    = vals.std()
    thresh = mean - n_sd * std

    if _VERBOSE:
        print("Subject ISC (mean r across voxels):")
        for sub in sorted(isc_scores, key=lambda s: isc_scores[s]):
            flag = "  <-- LOW ISC" if isc_scores[sub] < thresh else ""
            print(f"  {sub}  r={isc_scores[sub]:.4f}{flag}")
        print(f"\nMean={mean:.4f}  SD={std:.4f}  Threshold (mean-{n_sd}SD)={thresh:.4f}")

    excluded = [s for s in loaded if isc_scores[s] < thresh]
    retained = [s for s in loaded if isc_scores[s] >= thresh]

    if _VERBOSE:
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


def annotate_scenes(story_name, transcript, valid_scenes, out_path):
    """Segment a story transcript into labeled scene instances using Claude.

    Results are written to out_path as a TSV and cached; subsequent calls
    with the same out_path skip the API call and return the cached result.

    Parameters:
    ----------
    story_name: str
        Human-readable story title used in the prompt.
    transcript: str
        Full plaintext transcript to annotate.
    valid_scenes: list of str
        Scene label vocabulary (e.g., ["home", "bus", "office"]).
    out_path: Path
        Destination TSV file for caching annotations.

    Return:
    ------
    rows: list of dict
        List of {"scene": str, "excerpt": str} dicts in story order.
    """
    if out_path.exists():
        if _VERBOSE:
            print(f"{out_path.name} already exists — skipping API call.")
        with open(out_path) as f:
            return list(csv.DictReader(f, delimiter="\t"))

    # Authenticate with the API key
    api_key = load_vault()
    client = anthropic.Anthropic()
    labels_str = ", ".join(f'"{p}"' for p in valid_scenes)

    # ATTRIBUTION: Claude Code wrote this prompt.
    prompt = f"""You are analyzing the story "{story_name}".

Step 1 — Segment: Divide the full transcript into contiguous segments and label each with one of: {labels_str}, or "other".

Use a named scene label ONLY when a listener would clearly picture themselves inside that specific location — the scene is grounded there, characters are physically present, and the setting is the obvious foreground of the action. If there is any doubt about where the scene is set, or if the text is transitional, reflective, or could take scene anywhere, label it "other".

Step 2 — List all segments in their original story order as a JSON array, including "other" segments. Each element must be:
  {{"scene": "<label>", "excerpt": "<one or more full sentences from the transcript that represent this segment>"}}

Rules:
- Do not reorder segments.
- Excerpts must be full sentences taken directly from the transcript. No ellipses.
- The same scene label may appear many times if it recurs in the story.
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
        writer = csv.DictWriter(f, fieldnames=["scene", "excerpt"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    scene_counts = {p: sum(1 for r in rows if r["scene"] == p) for p in valid_scenes + ["other"]}
    print(f"Saved {out_path.name} — {len(rows)} rows: {scene_counts}")
    return rows

def align_tsv_to_whisperx(tsv_path, segments, transcript, padding_start=0):
    """Align TSV scene annotations to WhisperX word-level timestamps.

    Uses a two-pass forward scan to match each excerpt against the WhisperX
    word list, then assigns start/end times to each scene instance.

    Parameters:
    ----------
    tsv_path: Path
        TSV file with (scene, excerpt) columns produced by annotate_scenes.
    segments: list of dict
        WhisperX segment list from fetch_segments.
    transcript: str
        Unused; kept for API compatibility.
    padding_start: float
        Unused; WhisperX timestamps are already absolute.

    Return:
    ------
    rows: list of dict
        List of {"scene": str, "start": float, "end": float, "excerpt": str}
        dicts sorted by onset time.
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
            matched.append({"scene": row["scene"], "start": result[1], "_idx": result[0], "excerpt": row["excerpt"]})
        else:
            if _VERBOSE:
                print(f"  [no match] {row['excerpt'][:60]}")

    matched.sort(key=lambda r: r["_idx"])
    rows = [{"scene": r["scene"], "start": r["start"], "excerpt": r["excerpt"]} for r in matched]
    for i in range(len(rows) - 1):
        rows[i]["end"] = rows[i + 1]["start"]
    rows[-1]["end"] = segments[-1]["end"]

    for r in rows:
        words_exc = r["excerpt"].split()
        preview = " ".join(words_exc[:5]) + " ... " + " ".join(words_exc[-5:]) if len(words_exc) > 10 else r["excerpt"]
        if _VERBOSE:
            print(f"{r['start']:7.1f}s  [{r['scene']}]  {preview}")

    return rows

def plot_scene_timeline(aligned_rows, title, scene_order, scene_colors):
    """Plot a horizontal timeline with one row per scene category.

    Parameters:
    ----------
    aligned_rows: list of dict
        Scene instances with "scene", "start", "end" keys.
    title: str
        Plot title.
    scene_order: list of str
        Scene labels defining y-axis order.
    scene_colors: dict {str: str}
        Hex color per scene label; unlabeled rows default to grey.
    """
    label_to_y = {p: i for i, p in enumerate(scene_order)}
    label_to_y["other"] = -1

    _, ax = plt.subplots(figsize=(8, 3))
    for row in aligned_rows:
        y = label_to_y.get(row["scene"], -1)
        color = scene_colors.get(row["scene"], "#aaaaaa")
        ax.plot([row["start"], row["end"]], [y, y], color=color, linewidth=6, solid_capstyle="butt")

    ax.set_yticks([-1] + list(range(len(scene_order))))
    ax.set_yticklabels(["other"] + scene_order)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_xlim(left=0, right=aligned_rows[-1]["end"] + 10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_instance_onsets(instances, title, scene_order, scene_colors):
    """Plot scene instance onsets as 10-second horizontal bars.

    Parameters:
    ----------
    instances: list of dict
        Scene instances with "scene", "start", "end" keys.
    title: str
        Plot title.
    scene_order: list of str
        Scene labels defining y-axis order.
    scene_colors: dict {str: str}
        Hex color per scene label.
    """

    label_to_y = {p: i for i, p in enumerate(scene_order)}

    _, ax = plt.subplots(figsize=(8, 3))
    for row in instances:
        y     = label_to_y.get(row["scene"], -1)
        color = scene_colors.get(row["scene"], "#aaaaaa")
        ax.plot(
            [row["start"], row["start"] + 10],
            [y, y],
            color=color, linewidth=6, solid_capstyle="butt"
        )

    ax.set_yticks(list(range(len(scene_order))))
    ax.set_yticklabels(scene_order)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_xlim(left=0, right=instances[-1]["end"] + 10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()