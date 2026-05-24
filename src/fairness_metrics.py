
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix

def gerar_pares_e_distancias(df_grupo, max_pares=5000):
    pares_true = []
    pares_pred_dist = []
    
    # agrupa fotos por pessoa para facilitar a criação de pares
    pessoas = df_grupo.groupby('identidade')
    
    # cria os pares (mesma pessoa -> y_true = 1)
    for nome, grupo in pessoas:
        if len(grupo) >= 2:
            embeddings = list(grupo['embedding'])
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = cosine(embeddings[i], embeddings[j]) # similaridade cosseno
                    pares_pred_dist.append(dist)
                    pares_true.append(1)
                    if len(pares_true) >= max_pares // 2:
                        break
        if len(pares_true) >= max_pares // 2:
            break

    num_trues = len(pares_true)
    
    # cria pares diferentes (pessoas diferentes -> y_true = 0)
    lista_pessoas = list(pessoas.groups.keys())
    while len(pares_true) < num_trues * 2: # Equilibrar 50/50 com verdadeiros (true positives)
        p1, p2 = np.random.choice(lista_pessoas, size=2, replace=False)
        emb1 = pessoas.get_group(p1).sample(1)['embedding'].values[0]
        emb2 = pessoas.get_group(p2).sample(1)['embedding'].values[0]
        
        dist = cosine(emb1, emb2)
        pares_pred_dist.append(dist)
        pares_true.append(0)
        
    return np.array(pares_true), np.array(pares_pred_dist)

def calculate_confusion_matrix_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return tpr, tnr, fpr, fnr

# paridade demográfica - independência
def demographic_parity(y_pred, A):
    total_0 = 0
    total_1 = 0

    sum_scores_0 = 0
    sum_scores_1 = 0
    for i in range(len(y_pred)):
        if A[i] == 0:
            sum_scores_0 += y_pred[i]
            total_0 += 1
        else:
            sum_scores_1 += y_pred[i]
            total_1 += 1

    mean_0 = sum_scores_0 / total_0 if total_0 > 0 else 0
    mean_1 = sum_scores_1 / total_1 if total_1 > 0 else 0

    spd = np.abs(mean_0 - mean_1)
    di = mean_0 / mean_1 if mean_1 > 0 else 0

    return spd, di

# igualdade de oportunidades - separação
def equal_opportunity(y_true, y_pred, A):
    groups = [0, 1]
    tpr_rates = {}
    fpr_rates = {}
    for group in groups:
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(y_pred)):
            if A[i] == group:
                if y_pred[i] == 1:
                    if y_true[i] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if y_true[i] == 1:
                        fn += 1
                    else:
                        tn += 1
        tpr_rates[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_rates[group] = fp / (fp + tn) if (fp + tn) > 0 else 0

    eod_tpr = np.abs(tpr_rates[0] - tpr_rates[1])
    eod_fpr = np.abs(fpr_rates[0] - fpr_rates[1])
    
    return tpr_rates, fpr_rates, eod_tpr, eod_fpr

# calibração - suficiência
def calibration_by_group(y_true, score, A):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    num_bins = len(bins) - 1
    
    positive_g0 = np.zeros(num_bins)
    total_g0 = np.zeros(num_bins)
    
    positive_g1 = np.zeros(num_bins)
    total_g1 = np.zeros(num_bins)
    
    for i in range(len(score)):
        group = A[i]
        s = score[i]
        y_t = y_true[i]
        for b in range(num_bins):
            if b == num_bins - 1:
                condition = (s >= bins[b] and s <= bins[b+1])
            else:
                condition = (s >= bins[b] and s < bins[b+1])
            
            if condition:
                if group == 0:
                    if y_t == 1:
                        positive_g0[b] += 1
                    total_g0[b] += 1
                else:
                    if y_t == 1:
                        positive_g1[b] += 1
                    total_g1[b] += 1
                break
                
    # Calcular a taxa de reais positivos por bin para cada grupo
    calib_g0 = np.zeros(num_bins)
    calib_g1 = np.zeros(num_bins)
    for b in range(num_bins):
        calib_g0[b] = positive_g0[b] / total_g0[b] if total_g0[b] > 0 else 0
        calib_g1[b] = positive_g1[b] / total_g1[b] if total_g1[b] > 0 else 0
        
    return bins, calib_g0, calib_g1

if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl_path = os.path.join(path, "src", "features_lfw.pkl")

    df = pd.read_pickle(pkl_path)
    df_white = df[df['raca'] == 'White']
    df_black = df[df['raca'] == 'Black']
    
    y_true_w, dists_w = gerar_pares_e_distancias(df_white, max_pares=4000)
    y_true_b, dists_b = gerar_pares_e_distancias(df_black, max_pares=1000)
    
    threshold_glob = 0.5
    y_pred_w = (dists_w < threshold_glob).astype(int)
    y_pred_b = (dists_b < threshold_glob).astype(int)
    
    # converter as distâncias para scores de semelhança entre 0 e 1
    scores_w = np.clip(1.0 - dists_w, 0, 1)
    scores_b = np.clip(1.0 - dists_b, 0, 1)
    
    # meter no formato das funções de fairness chamadas a seguir
    y_true_comp = np.concatenate([y_true_w, y_true_b])
    y_pred_comp = np.concatenate([y_pred_w, y_pred_b])
    scores_comp = np.concatenate([scores_w, scores_b])
    
    # atributo sensível, 0 para branco, 1 para preto
    A_comp = np.concatenate([np.zeros(len(y_true_w)), np.ones(len(y_true_b))])
    
    # metricas de fairness
    spd, di = demographic_parity(y_pred_comp, A_comp)
    tpr_rates, fpr_rates, eod_tpr, eod_fpr = equal_opportunity(y_true_comp, y_pred_comp, A_comp)
    bins, calib_white, calib_black = calibration_by_group(y_true_comp, scores_comp, A_comp)
    
    print("Resultado das Métricas de Fairness (Independência, Separação, Suficiência)")
    print("="*60)
    print(f"Limiar de Decisão Biométrico (Similaridade Coseno): {threshold_glob}")
    print("-"*60)
    print("1. Independência (Paridade Demográfica):")
    print(f"  - Diferença de Paridade Estatística (SPD): {spd:.4f} (Ideal: 0)")
    print(f"  - Rácio de Impacto Disparate (DI):         {di:.4f} (Ideal: 1.0, Regra dos 80%: >0.8)")
    print("-"*60)
    print("2. Separação (Igualdade de Oportunidades):")
    print(f"  - Grupo WHITE (0) -> TPR: {tpr_rates[0]:.4f} | FPR: {fpr_rates[0]:.4f}")
    print(f"  - Grupo BLACK (1) -> TPR: {tpr_rates[1]:.4f} | FPR: {fpr_rates[1]:.4f}")
    print(f"  - Diferença de TPR (Equalized Odds TPR):  {eod_tpr:.4f} (Ideal: 0)")
    print(f"  - Diferença de FPR (Equalized Odds FPR):  {eod_fpr:.4f} (Ideal: 0)")
    print("-"*60)
    print("3. Suficiência (Calibração por Grupo):")
    print("  Intervalo de Score  |  Precisão Real WHITE  |  Precisão Real BLACK")
    for b in range(len(bins)-1):
        print(f"  [{bins[b]:.1f} - {bins[b+1]:.1f}]       |        {calib_white[b]:.4f}        |        {calib_black[b]:.4f}")
    print("="*60)
