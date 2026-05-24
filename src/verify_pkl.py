import os
import pandas as pd

# paths
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkl_file = os.path.join(main_path, "src", "features_lfw.pkl")

if not os.path.exists(pkl_file):
    print(f"Erro, o ficheiro {pkl_file} não existe na pasta")
    exit()

# carrega aqui o arquivo .pkl
df = pd.read_pickle(pkl_file)

# Verificar o tamanho da tabela
print(f"Total de fotos processadas com sucesso: {len(df)}")

# verificar a distribuição por raça
print("\nDistribuição demográfica extraída:")
print(df['raca'].value_counts())

# espreitar as primeiras linhas para garantir que os vetores estão lá
print("\nAmostra das primeiras 3 linhas da tabela:")
print(df[['identidade', 'foto_id', 'raca']].head(3))

# validar o formato do feature vector (embedding)
print("\nValidação do Feature Vector (Embedding):")
primeiro_embedding = df['embedding'].iloc[0]
print(f"Tipo de dados: {type(primeiro_embedding)}")
print(f"Tamanho do vetor (Dimensões): {len(primeiro_embedding)}")
print(f"Amostra dos 5 primeiros números do vetor: {primeiro_embedding[:5]}")