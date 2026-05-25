"""
One-time script to restructure the Tiny ImageNet val split into
ImageFolder-compatible format (class subdirectories).

Download first:
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
  unzip tiny-imagenet-200.zip -d data/

Then run:
  python prepare_tinyimagenet.py --data-dir data/tiny-imagenet-200
"""

import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/tiny-imagenet-200")
    args = p.parse_args()

    val_dir = Path(args.data_dir) / "val"
    images_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"

    # drop the unlabeled test split — not used, saves ~500 MB
    test_dir = Path(args.data_dir) / "test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("Removed unused test/ split.")

    if not images_dir.exists():
        print(
            f"Val images already restructured or not found at {images_dir}. Nothing to do."
        )
        return

    # parse annotations: filename -> class_id
    mapping = {}
    with open(annotations) as f:
        for line in f:
            parts = line.strip().split("\t")
            mapping[parts[0]] = parts[1]

    # move each image into val/<class_id>/
    for fname, class_id in mapping.items():
        src = images_dir / fname
        dst_dir = val_dir / class_id
        dst_dir.mkdir(exist_ok=True)
        shutil.move(str(src), str(dst_dir / fname))

    images_dir.rmdir()
    print(
        f"Done. Val split restructured into {len(set(mapping.values()))} class folders."
    )


if __name__ == "__main__":
    main()
