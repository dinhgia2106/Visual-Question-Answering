import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, AutoTokenizer

from ViT_RoBERTa_dataset import VQADataset
from ViT_RoBERTa_model import VQAModel, TextEncoder, VisualEncoder, Classifier

def _load_split(list_path: str):
    """Read a split (train/val/test) and return a list of dicts."""
    samples = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Each line: <img_tag>\t<question>? <answer>
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img_tag, qa_text = parts[0], parts[1]

            # Split img_tag to get filename before '#'
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

def build_label_mapping(train_data):
    classes = {sample["answer"] for sample in train_data}
    label2idx = {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}
    return label2idx

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            images = inputs['image'].to(device)
            questions = inputs['question'].to(device)
            labels = inputs['label'].to(device)
            
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    loss = sum(losses) / len(losses) if losses else 0
    acc = correct / total if total > 0 else 0
    
    return loss, acc

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device
):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        batch_train_losses = []
        
        model.train()
        for idx, inputs in enumerate(train_loader):
            images = inputs['image'].to(device)
            questions = inputs['question'].to(device)
            labels = inputs['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            batch_train_losses.append(loss.item())
            
        train_loss = sum(batch_train_losses) / len(batch_train_losses) if batch_train_losses else 0
        train_losses.append(train_loss)
        
        val_loss, val_acc = evaluate(
            model, val_loader,
            criterion, device
        )
        val_losses.append(val_loss)
        
        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}')
        
        scheduler.step()
        
    return train_losses, val_losses

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define paths
    TRAIN_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.TrainImages.txt")
    DEV_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.DevImages.txt")
    TEST_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.TestImages.txt")
    IMG_DIR = os.path.join("vqa_coco_dataset", "val2014-resised")

    # Load data
    print("Loading data...")
    train_data = _load_split(TRAIN_LIST)
    val_data = _load_split(DEV_LIST)
    test_data = _load_split(TEST_LIST)

    label2idx = build_label_mapping(train_data)
    n_classes = len(label2idx)
    print(f"Number of classes: {n_classes}")

    # Transforms
    data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.CenterCrop(size=180),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
    ])

    # Feature Extractor and Tokenizer
    print("Initializing Feature Extractor and Tokenizer...")
    img_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Datasets
    print("Creating Datasets...")
    train_dataset = VQADataset(
        train_data,
        label2idx=label2idx,
        img_feature_extractor=img_feature_extractor,
        text_tokenizer=text_tokenizer,
        device=device,
        transforms=data_transform,
        img_dir=IMG_DIR
    )
    val_dataset = VQADataset(
        val_data,
        label2idx=label2idx,
        img_feature_extractor=img_feature_extractor,
        text_tokenizer=text_tokenizer,
        device=device,
        img_dir=IMG_DIR
    )
    test_dataset = VQADataset(
        test_data,
        label2idx=label2idx,
        img_feature_extractor=img_feature_extractor,
        text_tokenizer=text_tokenizer,
        device=device,
        img_dir=IMG_DIR
    )

    # DataLoaders
    train_batch_size = 256
    test_batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    # Model
    print("Initializing Model...")
    hidden_size = 256
    dropout_prob = 0.2

    text_encoder = TextEncoder().to(device)
    visual_encoder = VisualEncoder().to(device)
    classifier = Classifier(
        hidden_size=hidden_size,
        dropout_prob=dropout_prob,
        n_classes=n_classes
    ).to(device)

    model = VQAModel(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        classifier=classifier
    ).to(device)
    
    model.freeze()

    # Training Setup
    lr = 1e-3
    epochs = 50
    scheduler_step_size = epochs * 0.8
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=0.1
    )

    # Train
    print("Starting Training...")
    train_losses, val_losses = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs,
        device
    )

    # Evaluate
    print('Evaluation on val / test dataset')
    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device
    )
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f'Val accuracy: {val_acc}')
    print(f'Test accuracy: {test_acc}')

if __name__ == "__main__":
    main()
