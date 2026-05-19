import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import timm
import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
sys.path.append('.') # garante que ele encontra o main_just_wb
from main_just_wb import FairFaceDataset, PATH, val_csv, transform
from fairness_metrics import demographic_parity, equal_opportunity, calibration_by_group

device = "cuda" if torch.cuda.is_available() else "cpu"

# carrega o conj. de validação
val_dataset = FairFaceDataset(val_csv, PATH, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# carrega o modelo que treinámos
model = timm.create_model("resnet50", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/model_black&white_base.pth", map_location=device))
model.to(device)
model.eval()

# queremos recolher as predicts para calcular as métricas de fairness
all_preds = []
all_labels = []
all_races = []
all_scores = []

with torch.no_grad():
    for images, labels, races, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        # funcao de ativação softmax
        probs = F.softmax(outputs, dim=1) 
        scores = probs[:, 1] # probabilidade da classe discount
        _, preds = torch.max(outputs, 1) # _ pois nao queremos o valor para classificar portanto descarta-se, percorre o tensor ao longo das colunas e procura o mais alto em cada linha

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_races.extend(races)
        all_scores.extend(scores.cpu().numpy())

A = [0 if r == "Black" else 1 for r in all_races]

spd, di = demographic_parity(all_preds, A)
tpr_rates, fpr_rates, eod_tpr, eod_fpr = equal_opportunity(all_labels, all_preds, A)

