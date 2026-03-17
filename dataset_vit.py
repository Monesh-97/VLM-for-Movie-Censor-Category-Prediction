import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
from pathlib import Path
from torchvision import transforms

class MovieCertificateViTDataset(Dataset):
    def __init__(self, excel_path, video_dir, num_frames=16, transform=None):
        self.df = pd.read_excel(excel_path)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        
        # Mapping labels to integers
        self.cert_to_idx = {'U': 0, 'U/A': 1, 'A': 2}
        
        self.samples = []
        video_files = list(self.video_dir.glob("*.mkv"))
        
        for idx, row in self.df.iterrows():
            name = str(row['Movie Name']).lower().strip()
            cert = self.cert_to_idx[str(row['Certificate']).strip()]
            
            # Match Mkv explicitly
            for vc in video_files:
                if name in vc.name.lower():
                    self.samples.append((str(vc), cert))
                    break
                    
        # Apply ViT required data transformations (224x224 and ImageNet Normalization)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate step to get uniformly spaced frames
        step = max(1, total_frames // self.num_frames)
        
        frames = []
        for i in range(self.num_frames):
            frame_idx = min(i * step, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply transformation
                if self.transform:
                    frame = self.transform(frame) # tensor of [3, 224, 224]
                frames.append(frame)
            else:
                # Black frame padding in rare failure cases
                frames.append(torch.zeros(3, 224, 224))
                
        cap.release()
        
        # Stack frames to standard video tensor [Frames, Channels, Height, Width]
        video_tensor = torch.stack(frames)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)
