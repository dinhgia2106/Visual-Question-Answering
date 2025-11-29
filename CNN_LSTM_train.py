import torch
import torch.nn as nn
import os
from vqa_model import VQAModel
from vqa_pipeline import build_loaders

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    
    with torch.no_grad():
        for image, question, labels in dataloader:
            image = image.to(device)
            question = question.to(device)
            labels = labels.to(device)
            
            outputs = model(image, question)
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
    device,
    epochs
):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        batch_train_losses = []
        model.train()
        
        for idx, (images, questions, labels) in enumerate(train_loader):
            images = images.to(device)
            questions = questions.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_train_losses.append(loss.item())
            
        train_loss = sum(batch_train_losses) / len(batch_train_losses)
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
    # Assuming the dataset is in the 'vqa_coco_dataset' directory relative to this script
    TRAIN_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.TrainImages.txt")
    DEV_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.DevImages.txt")
    TEST_LIST = os.path.join("vqa_coco_dataset", "vaq2.0.TestImages.txt")
    IMG_DIR = os.path.join("vqa_coco_dataset", "val2014-resised")

    # Build loaders
    print("Building dataloaders...")
    loaders = build_loaders(
        train_list=TRAIN_LIST,
        val_list=DEV_LIST,
        test_list=TEST_LIST,
        img_dir=IMG_DIR,
        batch_size_train=16, # Reduced batch size for safety, user can adjust
        batch_size_test=16
    )
    
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    vocab_size = len(loaders["vocab"])
    n_classes = len(loaders["label2idx"])
    
    print(f"Vocab size: {vocab_size}")
    print(f"Number of classes: {n_classes}")

    # Initialize Model
    print("Initializing model...")
    model = VQAModel(
        n_classes=n_classes,
        img_model_name='resnet18', # You can change this if needed
        vocab_size=vocab_size,
        embedding_dim=128,
        n_layers=2,
        hidden_size=256,
        drop_p=0.2
    ).to(device)

    # Hyperparameters
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

    # Training
    print("Starting training...")
    train_losses, val_losses = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs
    )

    # Evaluation
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
    
    print(f'Val accuracy: {val_acc:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()
