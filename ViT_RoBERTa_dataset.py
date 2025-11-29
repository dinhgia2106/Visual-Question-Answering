import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class VQADataset(Dataset):
    def __init__(
        self,
        data,
        label2idx,
        img_feature_extractor,
        text_tokenizer,
        device,
        transforms=None,
        img_dir='val2014-resised'
    ):
        self.data = data
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.img_feature_extractor = img_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.data[index]['image_path'])
        # Ensure image is RGB
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        if self.img_feature_extractor:
            img = self.img_feature_extractor(images=img, return_tensors="pt")
            # Move to device and squeeze batch dimension
            img = {k: v.to(self.device).squeeze(0) for k, v in img.items()}

        question = self.data[index]['question']
        if self.text_tokenizer:
            question = self.text_tokenizer(
                question,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_tensors="pt"
            )
            # Move to device and squeeze batch dimension
            question = {k: v.to(self.device).squeeze(0) for k, v in question.items()}

        label = self.data[index]['answer']
        label = torch.tensor(
            self.label2idx[label],
            dtype=torch.long
        ).to(self.device)

        sample = {
            'image': img,
            'question': question,
            'label': label
        }

        return sample
