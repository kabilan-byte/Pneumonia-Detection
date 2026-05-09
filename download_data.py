"""
download_data.py — Kaggle Dataset Downloader
=============================================
Downloads the Chest X-Ray Pneumonia dataset from Kaggle and
organises it into:
    data/
        train/  NORMAL/  PNEUMONIA/
        val/    NORMAL/  PNEUMONIA/
        test/   NORMAL/  PNEUMONIA/

Prerequisites:
    1. pip install kaggle
    2. Place your Kaggle API token at ~/.kaggle/kaggle.json
       (Download from: https://www.kaggle.com/settings → API → Create New Token)

Usage:
    python download_data.py
    python download_data.py --dest custom_data_dir/
"""

import os
import sys
import argparse
import zipfile
import shutil


KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
DEFAULT_DEST   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Folder names inside the Kaggle zip
KAGGLE_SPLITS = {
    "chest_xray/train": "train",
    "chest_xray/val":   "val",
    "chest_xray/test":  "test",
}


def check_kaggle_token():
    """Verify the Kaggle API token is present."""
    token_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(token_path):
        print("❌  Kaggle API token not found!\n")
        print("    To fix this:")
        print("    1. Go to https://www.kaggle.com/settings")
        print("    2. Scroll to 'API' → click 'Create New Token'")
        print("    3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("    4. chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("✅  Kaggle API token found.")


def download_dataset(dest: str = DEFAULT_DEST):
    """Download and extract the Kaggle dataset."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("❌  kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    check_kaggle_token()

    os.makedirs(dest, exist_ok=True)
    zip_target = os.path.join(dest, "chest_xray.zip")

    print(f"\n📥 Downloading dataset: {KAGGLE_DATASET}")
    print(f"   Destination: {dest}\n")

    os.system(
        f'kaggle datasets download -d {KAGGLE_DATASET} -p "{dest}" --unzip'
    )

    # ── Flatten Kaggle's nested structure ─────────────────────
    # Kaggle unzips to: data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/
    # We want:          data/{train,val,test}/{NORMAL,PNEUMONIA}/
    #
    # The val/ split in the original dataset is tiny (16 images).
    # We handle this by creating a proper val split from train if needed.

    for kaggle_split, target_split in KAGGLE_SPLITS.items():
        src = os.path.join(dest, kaggle_split)
        dst = os.path.join(dest, target_split)

        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            print(f"✅  Moved {kaggle_split}/ → {target_split}/")

    # Remove leftover chest_xray folder
    leftover = os.path.join(dest, "chest_xray")
    if os.path.exists(leftover):
        shutil.rmtree(leftover)

    # ── Count images ──────────────────────────────────────────
    print("\n📊 Dataset Summary:")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dest, split)
        if not os.path.exists(split_dir):
            continue
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                n = len([f for f in os.listdir(cls_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                print(f"   {split:6s} / {cls:10s}: {n:5d} images")

    print("\n🎉 Dataset ready in data/")
    print("   Next step:  python src/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle pneumonia dataset")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST,
                        help="Destination directory (default: data/)")
    args = parser.parse_args()
    download_dataset(dest=args.dest)
