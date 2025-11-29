import torch
import torch.nn as nn
import timm

class VQAModel(nn.Module):
    def __init__(
        self,
        n_classes,
        img_model_name,
        vocab_size,  # Added to replace len(vocab) dependency
        embedding_dim,
        n_layers=2,
        hidden_size=256,
        drop_p=0.2
    ):
        super(VQAModel, self).__init__()
        self.image_encoder = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=hidden_size
        )

        for param in self.image_encoder.parameters():
            param.requires_grad = True

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_p
        )

        # hidden_size * 3 because:
        # img_features (hidden_size) + lstm_out (hidden_size * 2 for bidirectional)
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = nn.Dropout(drop_p)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, img, text):
        img_features = self.image_encoder(img)

        text_emb = self.embedding(text)
        lstm_out, _ = self.lstm1(text_emb)

        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]

        combined = torch.cat((img_features, lstm_out), dim=1)
        x = self.fc1(combined)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Example initialization
    # Assuming some dummy values for vocab and classes since they are not provided
    classes = ['yes', 'no'] 
    vocab_size = 1000 

    n_classes = len(classes)
    img_model_name = 'resnet18'
    hidden_size = 256
    n_layers = 2
    embedding_dim = 128
    drop_p = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VQAModel(
        n_classes=n_classes,
        img_model_name=img_model_name,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        hidden_size=hidden_size,
        drop_p=drop_p
    ).to(device)

    print("Model initialized successfully:")
    print(model)