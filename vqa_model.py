import torch
import torch.nn as nn
import timm


class CNNLSTMModel(nn.Module):
    """Mô hình kết hợp CNN (xử lý ảnh) và LSTM (xử lý câu hỏi)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 512,
        cnn_backbone: str = "resnet50",
        num_classes: int = 2,
        padding_idx: int = 0,
        pretrained: bool = True,
    ):
        super().__init__()

        # CNN backbone – lấy đặc trưng ảnh
        self.cnn = timm.create_model(
            cnn_backbone,
            pretrained=pretrained,
            num_classes=0,  # không có fully connected cuối
        )
        cnn_out_dim = self.cnn.num_features

        # Embedding + LSTM cho câu hỏi
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Kết hợp hai nhánh
        self.fc = nn.Linear(cnn_out_dim + hidden_dim, num_classes)

    def forward(self, images: torch.Tensor, questions: torch.Tensor):
        # Hình ảnh qua CNN
        img_feat = self.cnn(images)  # shape: (B, cnn_out_dim)

        # Câu hỏi qua embedding + LSTM
        emb = self.embedding(questions)  # (B, T, embed_dim)
        _, (h_last, _) = self.lstm(emb)  # h_last: (1, B, hidden_dim)
        txt_feat = h_last.squeeze(0)  # (B, hidden_dim)

        # Kết hợp và phân lớp
        features = torch.cat([img_feat, txt_feat], dim=1)
        logits = self.fc(features)
        return logits 