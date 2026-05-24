import os
import torch
import pandas as pd
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_path = os.path.join(path, "dataset", "lfw-dataset", "lfw_atributos_cleaned.csv")
folder_lfw = os.path.join(path, "dataset", "lfw-dataset", "lfw-deepfunneled")
output_file = os.path.join(path, "src", "features_lfw.pkl")

# lê o file csv com os atributos cleaned (já com as respetivas raças)
df_atributos = pd.read_csv(csv_path)

map_racas = {}

for _, row in df_atributos.iterrows():
    nome = str(row['person'])
    sw = float(row['White'])
    sb = float(row['Black'])
    
    if sw > 0 and sw > sb:
        map_racas[nome] = "White"
    elif sb > 0 and sb > sw:
        map_racas[nome] = "Black"
    

