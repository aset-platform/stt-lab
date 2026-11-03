"""Repo guard: checks files before they are committed.

Rules
  1. Audio may only be committed as recordings/sample-* and must be at most 10 minutes long.
  2. No staged file name or added line may contain a term from the local .guard-terms file (one term
     per line, case-insensitive, whole words; '#' starts a comment). The file is optional.

Usage
  python scripts/repo_guard.py            check what is staged (the pre-commit hook runs this)
  python scripts/repo_guard.py --all      check every tracked file (CI runs this)
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_AUDIO_S = 10 * 60
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".aiff", ".flac", ".mp4", ".aac", ".opus"}
TERMS_FILE = ROOT / ".guard-terms"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def audio_seconds(path):
    """Duration from the container, or by decoding when the container doesn't say (browser .webm)."""
    import av  # PyAV: installed with the project (via faster-whisper), or `pip install av` in CI
    with av.open(str(path)) as c:
        if c.duration:
            return c.duration / 1_000_000
        stream = c.streams.audio[0]
        return sum(f.samples for f in c.decode(stream)) / stream.rate


def load_terms():
    if not TERMS_FILE.is_file():
        return None
    terms = [t.split("#")[0].strip() for t in TERMS_FILE.read_text().splitlines()]
    terms = [t for t in terms if t]
    return re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.I) if terms else None


def main():
    check_all = "--all" in sys.argv
    files = git("ls-files").split() if check_all else git("diff", "--cached", "--name-only", "--diff-filter=ACMR").split()
    problems = []

    for f in files:
        if Path(f).suffix.lower() not in AUDIO_EXT:
            continue
        if not (f.startswith("recordings/sample-") and "/" not in f[len("recordings/"):]):
            problems.append(f"{f}: audio can only be committed as recordings/sample-*")
            continue
        try:
            secs = audio_seconds(ROOT / f)
        except Exception as e:
            problems.append(f"{f}: can't read its length ({e}); not allowed")
            continue
        if secs > MAX_AUDIO_S:
            problems.append(f"{f}: {secs / 60:.1f} min of audio; samples must be 10 min or shorter")

    terms = load_terms()
    if terms:
        for f in files:
            if m := terms.search(f):
                problems.append(f"{f}: file name contains a blocked term ({m.group(0)!r})")
        added = git("diff", "--cached", "-U0", "--no-color") if not check_all else ""
        if check_all:
            for f in files:
                p = ROOT / f
                if p.suffix.lower() in AUDIO_EXT or p.suffix in {".bin", ".npz"}:
                    continue
                try:
                    text = p.read_text()
                except (UnicodeDecodeError, OSError):
                    continue
                for n, line in enumerate(text.splitlines(), 1):
                    if m := terms.search(line):
                        problems.append(f"{f}:{n}: contains a blocked term ({m.group(0)!r})")
        else:
            current = None
            for line in added.splitlines():
                if line.startswith("+++ "):
                    current = line[6:] if line.startswith("+++ b/") else None
                elif line.startswith("+") and current and (m := terms.search(line)):
                    problems.append(f"{current}: added line contains a blocked term ({m.group(0)!r})")

    if problems:
        print("repo_guard: blocked\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
