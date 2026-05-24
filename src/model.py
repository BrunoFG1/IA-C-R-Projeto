import os
import torch
import pandas as pd
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_path = os.path.join(path, "dataset", "lfw-dataset", "lfw_atributos_cleaned_fixed.csv")
folder_lfw = os.path.join(path, "dataset", "lfw-dataset", "lfw-deepfunneled", "lfw-deepfunneled")
output_file = os.path.join(path, "src", "features_lfw.pkl")

# lê o file csv com os atributos cleaned (já com as respetivas raças)
df_atributos = pd.read_csv(csv_path)

map_racas = {}

df_atributos['categoria_raca'] = "Ignored"
df_atributos.loc[(df_atributos['White'] > 0) & (df_atributos['White'] > df_atributos['Black']), 'categoria_raca'] = "White"
df_atributos.loc[(df_atributos['Black'] > 0) & (df_atributos['Black'] > df_atributos['White']), 'categoria_raca'] = "Black"

# carregar o modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detetor_faces = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
model_recogition = InceptionResnetV1(pretrained='vggface2').eval().to(device)

dados_process = []

# buscar o nome de cada pessoa
name_in_csv = df_atributos['person'].dropna().unique() # .dropna() para evitar nulos

# passar por todas as pastas para ir buscar os apelidos e juntar aos nomes proprios em cima
for name_dir in os.listdir(folder_lfw):
    person_dir = os.path.join(folder_lfw, name_dir)
    if not os.path.isdir(person_dir):
        continue
    
    # procura qual o nome do csv em cima encaixa nas subpastas de cada um
    nome_csv_corresp = None
    for n in name_in_csv:
        if name_dir == n or name_dir.endswith("_" + n):
            nome_csv_corresp = n
            break
    
    # se nao existe correspondente ignoramos
    if not nome_csv_corresp:
        continue
    
    # filtramos todas as linhas desta pessoa específica no csv
    lines_person = df_atributos[df_atributos['person'] == nome_csv_corresp].reset_index(drop=True)
    
    # listar fotos na pasta
    dir_photos = sorted([f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # juntar a foto à respetiva linha no file csv
    for i, foto_name in enumerate(dir_photos):
        if i >= len(lines_person):
            break
        
        # linha atual
        line_i = lines_person.iloc[i]
        race_atrib = line_i['categoria_raca']
        
        # ignoramos fotos onde os scores sao ambiguos
        if race_atrib == "Ignored":
            continue
        
        foto_path = os.path.join(person_dir, foto_name)
        
        try:
            img = Image.open(foto_path).convert('RGB')
            face_cropped = detetor_faces(img)
            if face_cropped is None:
                # se o MTCNN nao conseguir verificar uma cara limpa, salta foto
                continue 
            face_cropped = face_cropped.unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model_recogition(face_cropped)
            
            dados_process.append({
                "identidade": name_dir,     # nome real da pasta
                "foto_id": foto_name,       # nome da imagem 
                "raca": race_atrib,         # target para a métrica de fairness (White / Black)
                "embedding": embedding.squeeze(0).cpu().numpy().tolist()
            })
            
        except Exception as e:
            print(f"Erro a processar a imagem {foto_name}")

if dados_process:
    df_features = pd.DataFrame(dados_process)
    df_features.to_pickle(output_file) # file com os embeddings (feature vectors para cada foto de tam 512)
    print("Tudo feito") 
else:
    print("Erro, nenhuma imagem processada, problema de caminhos talvez")
    
    
    

