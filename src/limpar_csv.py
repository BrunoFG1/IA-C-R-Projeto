import os
import re

CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_ORIGINAL = os.path.join(CAMINHO_BASE, "dataset", "lfw-dataset", "lfw_atributos_cleaned.csv")
CSV_SAIDA = os.path.join(CAMINHO_BASE, "dataset", "lfw-dataset", "lfw_atributos_cleaned_fixed.csv")

print("A iniciar a limpeza inteligente e preenchimento de nomes...")

linhas_corrigidas = ["person,White,Black\n"]
current_person = None

with open(CSV_ORIGINAL, 'r', encoding='utf-8') as f:
    linhas = f.readlines()

for idx, linha in enumerate(linhas[1:], start=1):
    linha_str = linha.strip()
    if not linha_str:
        continue
        
    partes = linha_str.split(',')
    
    # White e Black estão sempre nas duas últimas colunas
    try:
        white_score = float(partes[-2])
        black_score = float(partes[-1])
    except ValueError:
        continue
        
    remainder = partes[:-2]
    first_elem = remainder[0].strip()
    
    # Se o primeiro elemento for um número, é uma linha de continuação da mesma pessoa
    if first_elem.isdigit():
        person_name = current_person
    else:
        # Se for um nome novo, limpamos números ou lixo para isolar o sobrenome
        name_parts = []
        for elem in remainder:
            elem_str = elem.strip()
            if elem_str.isdigit() or re.match(r'^-?\d+\.\d+$', elem_str):
                break
            name_parts.append(elem_str)
        person_name = "_".join(name_parts)
        current_person = person_name # Atualiza o ponteiro de repetição
        
    if person_name:
        linhas_corrigidas.append(f"{person_name},{white_score},{black_score}\n")

with open(CSV_SAIDA, 'w', encoding='utf-8') as f:
    f.writelines(linhas_corrigidas)

print(f"Sucesso! Ficheiro sem falhas guardado em: {CSV_SAIDA}")