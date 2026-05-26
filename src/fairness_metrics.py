import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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
                    dist = cosine(embeddings[i], embeddings[j]) # distância de coseno (0 = idêntico, 1 = diferente)
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
    
    # converter as distâncias para scores de semelhança entre 0 e 1 (similaridade cosseno)
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


# VIES MITIGATION 
model_white = LogisticRegression()
model_black = LogisticRegression()

Score_W_reshape = scores_w.reshape(-1, 1)
Y_true_w_reshape = y_true_w.reshape(-1, 1)
model_white.fit(Score_W_reshape, Y_true_w_reshape)

model_w_B1 = model_white.coef_
model_w_B0 = model_white.intercept_

Score_B_reshape = scores_b.reshape(-1, 1)
Y_true_B_reshape = y_true_b.reshape(-1, 1)
model_black.fit(Score_B_reshape, Y_true_B_reshape)

model_B_B1 = model_black.coef_
model_B_B0 = model_black.intercept_

prob_w = model_white.predict_proba(Score_W_reshape)[:, 1]
prob_b = model_black.predict_proba(Score_B_reshape)[:, 1]

pred_w = (prob_w >= 0.5).astype(int)
pred_b = (prob_b >= 0.5).astype(int)

calibrated_values_pred = np.concatenate((pred_w, pred_b))
calibrated_values_prob = np.concatenate((prob_w, prob_b))


spd_c, di_c = demographic_parity(calibrated_values_pred, A_comp)

tpr_rates_c, fpr_rates_c, eod_tpr_c, eod_fpr_c = equal_opportunity(y_true_comp, calibrated_values_pred, A_comp)

_, calib_white_c, calib_black_c = calibration_by_group(y_true_comp, calibrated_values_prob, A_comp)


print("\n" + " " * 15 + "RESULTADOS APÓS MITIGAÇÃO (CALIBRAÇÃO DE PLATT)")
print("="*60)
print("1. Independência (Paridade Demográfica Corrigida):")
print(f"  - Novo SPD: {spd_c:.4f} (Ideal: 0) | Novo DI: {di_c:.4f} (Ideal: 1.0)")
print("-"*60)
print("2. Separação (Igualdade de Oportunidades Corrigida):")
print(f"  - Novo WHITE -> TPR: {tpr_rates_c[0]:.4f} | FPR: {fpr_rates_c[0]:.4f}")
print(f"  - Novo BLACK -> TPR: {tpr_rates_c[1]:.4f} | FPR: {fpr_rates_c[1]:.4f}")
print(f"  - Nova Diferença de TPR (EOD TPR): {eod_tpr_c:.4f} (Ideal: 0)")
print(f"  - Nova Diferença de FPR (EOD FPR): {eod_fpr_c:.4f} (Ideal: 0)")
print("-"*60)
print("3. Suficiência (Nova Calibração por Grupo):")
print("  Intervalo de Score  |  Precisão PÓS-WHITE  |  Precisão PÓS-BLACK")
for b in range(len(bins)-1):
    print(f"  [{bins[b]:.1f} - {bins[b+1]:.1f}]       |        {calib_white_c[b]:.4f}       |        {calib_black_c[b]:.4f}")
print("="*60)


def plot_calibration_comparison(bins, calib_w_antes, calib_b_antes, calib_w_depois, calib_b_depois):
    """
    Gera um gráfico lado a lado comparando a calibração antes e depois por grupo.
    """
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    width = 0.06  # Largura das barras
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # --------------------------------------------------------------------------
    # GRÁFICO 1: ANTES DA CALIBRAÇÃO
    # --------------------------------------------------------------------------
    rects1_w = ax1.bar([c - width/2 for c in bin_centers], calib_w_antes, width, label='WHITE (Antes)', color='#1f77b4')
    rects1_b = ax1.bar([c + width/2 for c in bin_centers], calib_b_antes, width, label='BLACK (Antes)', color='#ff7f0e')
    
    # Linha ideal (Y = X)
    ax1.plot([0, 1], [0, 1], '--', color='gray', label='Calibração Ideal')
    
    ax1.set_title('Sem Calibração (Scores Brutos)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Intervalo de Score', fontsize=12)
    ax1.set_ylabel('Precisão Real (Proporção de Matches)', fontsize=12)
    ax1.set_xticks(bins)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # --------------------------------------------------------------------------
    # GRÁFICO 2: APÓS MITIGAÇÃO (PLATT)
    # --------------------------------------------------------------------------
    rects2_w = ax2.bar([c - width/2 for c in bin_centers], calib_w_depois, width, label='WHITE (Pós-Platt)', color='#2ca02c')
    rects2_b = ax2.bar([c + width/2 for c in bin_centers], calib_b_depois, width, label='BLACK (Pós-Platt)', color='#d62728')
    
    # Linha ideal (Y = X)
    ax2.plot([0, 1], [0, 1], '--', color='gray', label='Calibração Ideal')
    
    ax2.set_title('Com Calibração (Pós-Processamento Platt)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Intervalo de Score', fontsize=12)
    ax2.set_xticks(bins)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')

    # Função interna para meter as etiquetas de texto nas barras
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            if height > 0:  # Só mete texto se a barra não for zero
                ax.annotate(f'{height*100:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 pontos de desvio vertical
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Adicionar as percentagens em cima de cada barra
    autolabel(rects1_w, ax1)
    autolabel(rects1_b, ax1)
    autolabel(rects2_w, ax2)
    autolabel(rects2_b, ax2)
    
    plt.tight_layout()
    
    # Guarda a imagem diretamente na pasta do teu script
    plt.savefig('comparacao_calibracao_fairness.png', dpi=300)
    print("\n[SUCESSO] Gráfico guardado como 'comparacao_calibracao_fairness.png'!")
    plt.show()

# Chamar a função para gerar e guardar o gráfico do teu trabalho
plot_calibration_comparison(
        bins=bins,
        calib_w_antes=calib_white,   # Valores originais calculados no início
        calib_b_antes=calib_black,   # Valores originais calculados no início
        calib_w_depois=calib_white_c, # Teus novos valores pós-mitigação
        calib_b_depois=calib_black_c  # Teus novos valores pós-mitigação
    )