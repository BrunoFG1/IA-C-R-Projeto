import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from torch import nn
from torchvision.transforms import transforms 
from PIL import Image
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.gender_mapping = {'Male': 0, 'Female': 1}
        self.race_mapping = {
            'White': 'White',
            'Middle Eastern': 'White',
            'Latino_Hispanic': 'White',
            'Black': 'Black',
            'East Asian': 'Black',
            'Southeast Asian': 'Black',
            'Indian': 'Black'
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_name = self.data.iloc[index]['file']  
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")

        gender_str = self.data.iloc[index]['gender']
        label = self.gender_mapping[gender_str]
        
        orig_race = self.data.iloc[index]['race']
        race = self.race_mapping.get(orig_race)
        
        orig_age = self.data.iloc[index]['age']
        if orig_age in ["3-9", "10-19", "more than 70"]:
            label = 1
            age_group = "Discount"
        else:
            label = 0
            age_group = "No Discount"
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long), race, age_group

PATH = "fairface/FairFace/"
train_csv = PATH + "train_labels.csv"
val_csv = PATH + "val_labels.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = FairFaceDataset(train_csv, PATH, transform=transform)
val_dataset = FairFaceDataset(val_csv, PATH, transform=transform)

train_Dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
val_Dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

model = timm.create_model("resnet50", pretrained=True, num_classes=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

model.to(device)
epochs = 1

for epoch in range(epochs):
    model.train()
    actual_loss = 0.0
    for image, labels, _, _ in tqdm.tqdm(train_Dataloader, desc=f"Época {epoch+1}/{epochs}"):
        image, labels = image.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        actual_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {actual_loss/len(train_Dataloader):.4f}")

model.eval()

