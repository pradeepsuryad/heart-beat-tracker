"""
Download beating-heart endoscopic video sequences for the heart-beat-tracker.

Downloads rectified04.zip and rectified05.zip (in-vivo cardiac sequences) from
the Recasens/HamlynRectifiedDataset mirror on Hugging Face, extracts the left-
camera image sequence (image01/), and stitches it into an .avi for the tracker.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dest data --sequences 4 5
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

HF_BASE = (
    "https://huggingface.co/datasets/Recasens/HamlynRectifiedDataset/resolve/main"
)

# Sequence numbers available; 4 and 5 are in-vivo cardiac
CARDIAC_SEQUENCES = [4, 5]


def download_zip(url: str, dest_dir: Path) -> Path | None:
    """Stream-download a zip from url into dest_dir. Returns path to zip file."""
    try:
        import requests
    except ImportError:
        sys.exit("[ERROR] 'requests' not installed. Run: pip install requests")

    zip_path = dest_dir / Path(url).name
    if zip_path.exists():
        print(f"[SKIP] Already downloaded: {zip_path.name}")
        return zip_path

    print(f"[INFO] Downloading {url}")
    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        return None

    total = int(resp.headers.get("content-length", 0))
    written = 0
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            written += len(chunk)
            if total:
                print(f"\r  {zip_path.name}: {written / total * 100:.1f}%", end="", flush=True)
    print()
    print(f"[OK]   Saved: {zip_path}")
    return zip_path


def extract_image_sequence(zip_path: Path, dest_dir: Path, subfolder: str = "image01") -> Path | None:
    """Extract a named subfolder from the zip into dest_dir."""
    out_dir = dest_dir / f"{zip_path.stem}_{subfolder}"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[SKIP] Already extracted: {out_dir.name}/")
        return out_dir

    print(f"[INFO] Extracting {subfolder}/ from {zip_path.name} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if subfolder in m and not m.endswith("/")]
            if not members:
                print(f"[WARN] No '{subfolder}' entries found in {zip_path.name}")
                print("       Available top-level entries:", sorted({m.split("/")[0] for m in zf.namelist()}))
                return None
            for member in members:
                filename = Path(member).name
                target = out_dir / filename
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        print(f"[OK]   Extracted {len(members)} files → {out_dir.name}/")
    except Exception as exc:
        print(f"[ERROR] Extraction failed: {exc}")
        return None
    return out_dir


def stitch_to_avi(image_dir: Path, out_path: Path, fps: float = 25.0) -> bool:
    """Stitch a sorted image directory into an .avi file."""
    try:
        import cv2
    except ImportError:
        sys.exit("[ERROR] opencv-python not installed. Run: pip install -r requirements.txt")

    images = sorted(image_dir.glob("*.png")) or sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"[ERROR] No images found in {image_dir}")
        return False

    first = cv2.imread(str(images[0]))
    if first is None:
        print(f"[ERROR] Could not read {images[0]}")
        return False

    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        print(f"[ERROR] Could not open VideoWriter for {out_path}")
        return False

    print(f"[INFO] Stitching {len(images)} frames ({w}×{h}) → {out_path.name}")
    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is not None:
            writer.write(frame)
        if (i + 1) % 200 == 0:
            print(f"\r  {i + 1}/{len(images)} frames", end="", flush=True)
    print()
    writer.release()
    print(f"[OK]   Video saved: {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Hamlyn beating-heart dataset")
    parser.add_argument("--dest", default="data", help="Output directory (default: data/)")
    parser.add_argument(
        "--sequences",
        nargs="+",
        type=int,
        default=CARDIAC_SEQUENCES,
        help="Sequence numbers to download (default: 4 5)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate for stitched .avi (default: 25)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []

    for seq in args.sequences:
        zip_name = f"rectified{seq:02d}.zip"
        url = f"{HF_BASE}/{zip_name}"

        zip_path = download_zip(url, dest)
        if zip_path is None:
            continue

        img_dir = extract_image_sequence(zip_path, dest, subfolder="image01")
        if img_dir is None:
            continue

        avi_path = dest / f"hamlyn_seq{seq:02d}.avi"
        if avi_path.exists():
            print(f"[SKIP] Video already exists: {avi_path}")
            produced.append(avi_path)
            continue

        if stitch_to_avi(img_dir, avi_path, fps=args.fps):
            produced.append(avi_path)

    if not produced:
        print("\n[ERROR] No videos produced. Check errors above.")
        sys.exit(1)

    print("\n--- Ready to use ---")
    for p in produced:
        print(f"  {p}")

    first = produced[0]
    print(
        f"\nRun the tracker:\n"
        f"  python src/tracker.py "
        f"--input {first} "
        f"--output outputs/{first.stem}_flow.mp4 "
        f"--scale 0.5"
    )


if __name__ == "__main__":
    main()
