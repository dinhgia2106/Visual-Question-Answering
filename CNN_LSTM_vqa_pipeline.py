import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchvision import transforms
from PIL import Image
import spacy

# -----------------------------
# 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT
# -----------------------------
# PyTorch, TorchText, torchvision đã được import bên trên

# -----------------------------
# 3. CÀI ĐẶT GIÁ TRỊ NGẪU NHIÊN CỐ ĐỊNH
# -----------------------------

def set_seed(seed: int) -> None:
    """Đặt seed cho tất cả các thư viện để tái hiện kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Gọi ngay khi import module
_DEFAULT_SEED = 59
set_seed(_DEFAULT_SEED)


# -----------------------------
# 4. HÀM ĐỌC DỮ LIỆU TỪ FILE .TXT
# -----------------------------

def _load_split(list_path: str):
    """Đọc một split (train/val/test) và trả về list các dict."""
    samples = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Mỗi dòng: <img_tag>\t<question>? <answer>
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img_tag, qa_text = parts[0], parts[1]

            # Tách img_tag lấy phần trước dấu '#'
            img_filename = img_tag.split("#")[0]

            qa = qa_text.split("?")
            if len(qa) == 3:
                answer = qa[2].strip()
            else:
                answer = qa[1].strip()
            question = qa[0].strip() + "?"

            samples.append(
                {
                    "image_path": img_filename,
                    "question": question,
                    "answer": answer,
                }
            )
    return samples


# -----------------------------
# 5. XÂY DỰNG BỘ TỪ VỰNG
# -----------------------------

# Load spaCy English tokenizer
_eng = spacy.load("en_core_web_sm")



def _token_generator(data_iter):
    for sample in data_iter:
        question = sample["question"]
        yield [tok.text for tok in _eng.tokenizer(question)]


def build_vocab(train_data, min_freq: int = 2):
    specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
    counter = Counter()
    for tokens in _token_generator(train_data):
        counter.update(tokens)
    
    vocab = {token: i for i, token in enumerate(specials)}
    idx = len(specials)
    
    for word, count in counter.items():
        if count >= min_freq:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def build_label_mapping(train_data):
    classes = {sample["answer"] for sample in train_data}
    label2idx = {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}
    idx2label = {idx: cls_name for cls_name, idx in label2idx.items()}
    return label2idx, idx2label


def tokenize(question: str, vocab, max_seq_len: int):
    tokens = [tok.text for tok in _eng.tokenizer(question)]
    unk_idx = vocab.get("<unk>")
    seq = [vocab.get(token, unk_idx) for token in tokens]
    if len(seq) < max_seq_len:
        pad_idx = vocab.get("<pad>")
        seq += [pad_idx] * (max_seq_len - len(seq))
    else:
        seq = seq[:max_seq_len]
    return seq


class VQADataset(Dataset):
    def __init__(
        self,
        data,
        vocab,
        label2idx,
        max_seq_len: int = 20,
        transform=None,
        img_dir: str = "val2014-resised",
    ):
        self.data = data
        self.vocab = vocab
        self.label2idx = label2idx
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = os.path.join(self.img_dir, sample["image_path"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        question_tensor = torch.tensor(
            tokenize(sample["question"], self.vocab, self.max_seq_len),
            dtype=torch.long,
        )
        label_tensor = torch.tensor(
            self.label2idx[sample["answer"]], dtype=torch.long
        )
        return img, question_tensor, label_tensor


# -----------------------------
# 9. PYTORCH TRANSFORMS
# -----------------------------

data_transform = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.CenterCrop(180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
}


# -----------------------------
# 10, 11. TẠO DATASET VÀ DATALOADER TIỆN DỤNG
# -----------------------------

def build_loaders(
    train_list: str,
    val_list: str,
    test_list: str,
    img_dir: str,
    batch_size_train: int = 256,
    batch_size_test: int = 32,
    max_seq_len: int = 20,
):
    train_data = _load_split(train_list)
    val_data = _load_split(val_list)
    test_data = _load_split(test_list)

    vocab = build_vocab(train_data)
    label2idx, idx2label = build_label_mapping(train_data)

    train_dataset = VQADataset(
        train_data,
        vocab=vocab,
        label2idx=label2idx,
        max_seq_len=max_seq_len,
        transform=data_transform["train"],
        img_dir=img_dir,
    )
    val_dataset = VQADataset(
        val_data,
        vocab=vocab,
        label2idx=label2idx,
        max_seq_len=max_seq_len,
        transform=data_transform["val"],
        img_dir=img_dir,
    )
    test_dataset = VQADataset(
        test_data,
        vocab=vocab,
        label2idx=label2idx,
        max_seq_len=max_seq_len,
        transform=data_transform["val"],
        img_dir=img_dir,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_test, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=4
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "vocab": vocab,
        "label2idx": label2idx,
        "idx2label": idx2label,
    }


# -----------------------------
# Module có thể được gọi để kiểm tra nhanh
# -----------------------------
if __name__ == "__main__":
    # Các hằng mặc định – người dùng có thể chỉnh sửa cho phù hợp cây thư mục
    TRAIN_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.TrainImages.txt")
    DEV_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.DevImages.txt")
    TEST_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.TestImages.txt")
    IMG_DIR = os.path.join("vqa_coco_dataset", "val2014-resised")

    loaders = build_loaders(
        train_list=TRAIN_LIST,
        val_list=DEV_LIST,
        test_list=TEST_LIST,
        img_dir=IMG_DIR,
    )

    print("Train batches:", len(loaders["train_loader"]))
    print("Vocab size:", len(loaders["vocab"]))
    print("Classes:", loaders["label2idx"]) 