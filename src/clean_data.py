import os
import shutil

lfw_dir = "../dataset/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/"

if not os.path.exists(lfw_dir):
    print(f"Erro: A pasta {lfw_dir} nao foi encontrada. Verifica o caminho.")
    exit()

pastas = [f for f in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, f))]

total_inicial = len(pastas)
mantidos = 0
apagados = 0

for pasta_pessoa in pastas:
    caminho_completo = os.path.join(lfw_dir, pasta_pessoa)
    
    ficheiros = [img for img in os.listdir(caminho_completo) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_imagens = len(ficheiros)
    
    if num_imagens < 2:
        shutil.rmtree(caminho_completo)
        apagados += 1
    else:
        mantidos += 1

print("--- Processo Concluido ---")
print(f"Total de pessoas inicialmente: {total_inicial}")
print(f"Pessoas mantidas (2 ou mais fotos): {mantidos}")
print(f"Pastas apagadas (menos de 2 fotos): {apagados}")