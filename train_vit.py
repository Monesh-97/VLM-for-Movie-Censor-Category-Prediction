import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset_vit import MovieCertificateViTDataset
from model_vit import VideoViTClassifier

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    dataset = MovieCertificateViTDataset(
        excel_path="Dataset/dataset.xlsx",
        video_dir="Dataset",
        num_frames=16 # adjust based on GPU VRAM availability
    )
    # Colab can handle larger batch sizes with standard ViT
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 2. Init Model
    print("Loading Pretrained Vision Transformer...")
    model = VideoViTClassifier(num_classes=3, num_frames=16).to(device)
    
    # Optional: Freeze the ViT backbone to only train the linear Temporal Classifier head! 
    # Uncomment next two lines if you lack GPU VRAM, otherwise fine-tuning the full ViT is better.
    # for param in model.vit.parameters():
    #     param.requires_grad = False
    
    # 3. Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5) # Small LR for ViT fine-tuning
    
    epochs = 10
    print(f"Starting Training for {epochs} epochs over {len(dataset)} videos...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass: Compute Logits
            logits = model(videos)
            
            # Compute cross entropy loss
            loss = criterion(logits, labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        acc = 100 * correct / max(total, 1)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {acc:.2f}%")
        
    # Save the trained model
    out_path = "vit_movie_classifier.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Training Complete! Saved weights to '{out_path}'.")

if __name__ == "__main__":
    train()
