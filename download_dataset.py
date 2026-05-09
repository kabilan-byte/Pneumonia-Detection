"""
download_dataset.py — Download Chest X-Ray Dataset (No Kaggle account needed)
===============================================================================
Uses a direct download approach with opendatasets OR manual instructions.

OPTION 1 (Automatic): Uses opendatasets library
    pip install opendatasets
    python download_dataset.py

OPTION 2 (Manual): Download from browser and place files manually
    See instructions printed by this script.
"""

import os
import sys
import zipfile
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def try_opendatasets():
    """Try downloading via opendatasets (asks Kaggle username + key once)."""
    try:
        import opendatasets as od
        print("📥 Downloading via opendatasets...")
        print("   You need your Kaggle username and API key.")
        print("   Get it from: https://kaggle.com/settings → API → Create New Token\n")
        od.download(
            "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
            data_dir=DATA_DIR,
        )
        # opendatasets puts files in: data/chest-xray-pneumonia/chest_xray/
        restructure_opendatasets()
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"⚠️  opendatasets failed: {e}")
        return False


def restructure_opendatasets():
    """Move files from opendatasets nested structure to data/{train,val,test}/"""
    nested = os.path.join(DATA_DIR, "chest-xray-pneumonia", "chest_xray")
    if not os.path.exists(nested):
        return

    for split in ["train", "val", "test"]:
        src = os.path.join(nested, split)
        dst = os.path.join(DATA_DIR, split)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"✅ Moved {split}/ → data/{split}/")

    # Cleanup
    shutil.rmtree(os.path.join(DATA_DIR, "chest-xray-pneumonia"), ignore_errors=True)
    print("\n🎉 Dataset ready in data/")


def try_kaggle_cli():
    """Try using kaggle CLI if key exists."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        return False

    print("📥 Kaggle key found! Downloading via Kaggle CLI...")
    os.makedirs(DATA_DIR, exist_ok=True)
    ret = os.system(
        f'kaggle datasets download -d paultimothymooney/chest-xray-pneumonia '
        f'-p "{DATA_DIR}" --unzip'
    )
    if ret == 0:
        # Restructure Kaggle's nested layout
        for split in ["train", "val", "test"]:
            src = os.path.join(DATA_DIR, "chest_xray", split)
            dst = os.path.join(DATA_DIR, split)
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
        leftover = os.path.join(DATA_DIR, "chest_xray")
        if os.path.exists(leftover):
            shutil.rmtree(leftover)
        print("✅ Done! Files in data/")
        return True
    return False


def extract_zip_if_present():
    """If user manually placed a zip file, extract it."""
    zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chest-xray-pneumonia.zip")
    if os.path.exists(zip_path):
        print(f"📦 Found zip file: {zip_path}")
        print("   Extracting...")
        os.makedirs(DATA_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        restructure_opendatasets()
        return True
    return False


def print_manual_instructions():
    """Show step-by-step manual download instructions."""
    print("""
╔══════════════════════════════════════════════════════════╗
║          MANUAL DATASET DOWNLOAD INSTRUCTIONS            ║
╚══════════════════════════════════════════════════════════╝

No automatic download method worked. Follow these steps:

STEP 1: Open this URL in your browser:
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

STEP 2: Sign in to Kaggle (free account)

STEP 3: Click the "Download" button (top right) → gets a zip file

STEP 4: Move the downloaded zip to your project folder:
  /Users/sriram/Pneumonia Detection/chest-xray-pneumonia.zip

STEP 5: Run this script again:
  python download_dataset.py

  ── OR unzip manually and arrange like this ──

  data/
    train/
      NORMAL/     ← paste chest_xray/train/NORMAL images here
      PNEUMONIA/  ← paste chest_xray/train/PNEUMONIA images here
    val/
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/

STEP 6: Then run training:
  python src/train.py

══════════════════════════════════════════════════════════

FASTEST ALTERNATIVE — Get API key in 2 minutes:
  1. kaggle.com/settings → API → Create New Token → downloads kaggle.json
  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
  3. chmod 600 ~/.kaggle/kaggle.json
  4. python download_dataset.py
""")


if __name__ == "__main__":
    print("🫁 Pneumonia Detection — Dataset Downloader")
    print("=" * 55)

    # Try methods in order
    if try_kaggle_cli():
        sys.exit(0)

    if extract_zip_if_present():
        sys.exit(0)

    # Try opendatasets
    try:
        import opendatasets  # noqa
        if try_opendatasets():
            sys.exit(0)
    except ImportError:
        print("💡 Installing opendatasets for easier download...")
        os.system(f"{sys.executable} -m pip install opendatasets -q")
        if try_opendatasets():
            sys.exit(0)

    # Nothing worked — show manual instructions
    print_manual_instructions()
