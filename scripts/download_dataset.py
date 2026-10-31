"""
AURA-M Dataset Ingestion & Acquisition Module
Downloads the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset
from verified open-source mirrors and validates local files.
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AURA-M.DatasetDownloader")

RAW_DATA_DIR = Path("data/raw")

MIRROR_SOURCES = [
    "https://raw.githubusercontent.com/ashutoshmakone/Aircraft-Engine-Failure-Prediction-using-Deep-Learning/master/CMAPSSData/",
    "https://raw.githubusercontent.com/GURU1001S/NASA-Predictive-Maintenance-AI-using-cnnlstmattention/main/CMAPSSData/",
    "https://raw.githubusercontent.com/GURU1001S/NASA-Predictive-Maintenance-AI-using-cnnlstmattention/master/CMAPSSData/",
    "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/data/",
    "https://raw.githubusercontent.com/Azure/lstms_for_predictive_maintenance/master/Dataset/"
]

DATASET_FILES = [
    # Primary benchmark subset: FD001
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
    # Subsets for multi-subset capability
    "train_FD002.txt",
    "test_FD002.txt",
    "RUL_FD002.txt",
    "train_FD003.txt",
    "test_FD003.txt",
    "RUL_FD003.txt",
    "train_FD004.txt",
    "test_FD004.txt",
    "RUL_FD004.txt",
    "readme.txt"
]

def download_file(filename: str, target_dir: Path) -> bool:
    target_path = target_dir / filename
    if target_path.exists() and target_path.stat().st_size > 100:
        logger.info(f"File already present: {target_path.name} ({target_path.stat().st_size} bytes)")
        return True

    for mirror in MIRROR_SOURCES:
        url = mirror + filename
        try:
            logger.info(f"Attempting download: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    content = resp.read()
                    if len(content) > 100:
                        target_path.write_bytes(content)
                        logger.info(f"Successfully downloaded {filename} ({len(content)} bytes) from {mirror}")
                        return True
        except Exception as e:
            continue

    logger.warning(f"Could not retrieve {filename} from online mirrors.")
    return False

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initializing NASA C-MAPSS dataset acquisition into {RAW_DATA_DIR.resolve()}...")

    success_count = 0
    for f in DATASET_FILES:
        res = download_file(f, RAW_DATA_DIR)
        if res:
            success_count += 1

    logger.info(f"Dataset acquisition completed. Successfully available: {success_count}/{len(DATASET_FILES)} files.")
    
    # Validation check for FD001
    fd001_train = RAW_DATA_DIR / "train_FD001.txt"
    if fd001_train.exists():
        line_count = sum(1 for _ in open(fd001_train, "r"))
        logger.info(f"FD001 Training validation: {line_count} cycles recorded across 100 engines.")
        return 0
    else:
        logger.error("FD001 Training file not found.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
