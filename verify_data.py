import os
import random
from typing import List, Tuple, Dict
from math import ceil

import matplotlib.pyplot as plt  # new import for image display

# Paths – modify if your directory layout differs
DATA_DIR = os.path.join("vqa_coco_dataset", "val2014-resised")
TRAIN_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.TrainImages.txt")
DEV_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.DevImages.txt")
TEST_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.TestImages.txt")


def parse_line(line: str) -> Tuple[str, str, str]:
    """Split a single record line into image filename, question and answer."""
    # Each line looks like: `COCO_val2014_000000393225.jpg#0\tIs this a creamy soup ? no`
    # We split first on tab, then split the question/answer part on the last whitespace.
    image_tag, qa_text = line.rstrip("\n").split("\t", maxsplit=1)

    # Remove postfix like `#0` from image filename
    image_filename = image_tag.split("#")[0]

    # The answer is the last token after whitespace, the remaining is the question
    *question_tokens, answer = qa_text.split()
    question = " ".join(question_tokens)

    return image_filename, question, answer


def load_list(list_path: str) -> List[Tuple[str, str, str]]:
    """Load every record from list file."""
    with open(list_path, "r", encoding="utf-8") as f:
        return [parse_line(line) for line in f if line.strip()]


def verify_images(records: List[Tuple[str, str, str]], image_dir: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """Check whether image files referenced in records exist on disk.

    Returns a dictionary with keys 'ok' and 'missing'.
    """
    ok, missing = [], []
    for filename, question, answer in records:
        if os.path.exists(os.path.join(image_dir, filename)):
            ok.append((filename, question, answer))
        else:
            missing.append((filename, question, answer))
    return {"ok": ok, "missing": missing}


def sample_records(records: List[Tuple[str, str, str]], k: int = 5) -> List[Tuple[str, str, str]]:
    """Return a deterministic sample for reproducibility."""
    random.seed(42)
    return random.sample(records, k) if len(records) >= k else records


def display_samples(records: List[Tuple[str, str, str]], image_dir: str, cols: int = 3) -> None:
    """Display a grid of sample images with their question & answer."""
    if not records:
        print("No records to display.")
        return

    rows = ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    # Ensure axes is 2-D array for uniform indexing
    if rows == 1:
        axes = [axes]
    for idx, ax in enumerate(axes.flat if rows > 1 else axes):
        if idx >= len(records):
            ax.axis("off")
            continue
        filename, question, answer = records[idx]
        img_path = os.path.join(image_dir, filename)
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        # Wrap long text via small font size
        ax.set_title(f"Q: {question}\nA: {answer}", fontsize=8)
    plt.tight_layout()
    plt.show()


def main() -> None:
    splits = {
        "train": TRAIN_LIST,
        "dev": DEV_LIST,
        "test": TEST_LIST,
    }

    for split_name, path in splits.items():
        print(f"Processing {split_name} list -> {path} ...")
        records = load_list(path)
        stats = verify_images(records, DATA_DIR)
        print(f"  Total records: {len(records):,}")
        print(f"  Images found: {len(stats['ok']):,}")
        print(f"  Missing images: {len(stats['missing']):,}\n")

    # Show a few sample pairs from training set
    train_records = load_list(TRAIN_LIST)
    sampled = sample_records(train_records, k=6)
    print("Example records (train split):")
    for img, q, a in sampled:
        print(f"  Image: {img} | Question: {q} | Answer: {a}")

    # Display images
    display_samples(sampled, DATA_DIR)


if __name__ == "__main__":
    main() 