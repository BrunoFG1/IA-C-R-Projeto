
import numpy as np


# paridade demográfica - independência
def demograpchic_parity(y_pred, A):
    total_0 = 0
    total_1 = 0

    sum_scores_0 = 0
    sum_scores_1 = 0
    for i in range(len(y_pred)):
        if A[i] == 0:
            sum_scores_0 += y_pred[i] # soma o score desta instancia que tem race binary 0
            total_0 += 1
        
        else:
            sum_scores_1 += y_pred[i] # soma os scores de race_binary 1
            total_1 += 1

    mean_0 = sum_scores_0 / total_0
    mean_1 = sum_scores_1 / total_1

    # statistical parity difference
    spd = np.abs(mean_0 - mean_1)

    # disparate impact ratio
    di = mean_0 / mean_1

    return spd, di

# igualdade de opurtunidades - separação
def equal_oppurtunity(y_true, y_pred, A):
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
                    if y_true[i] == 0:
                        tn += 1
                    else:
                        fn += 1
    if (tp + fn) > 0:
        tpr = tp / (tp + fn)
    else:
        tpr = 0

    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0

    tpr_rates[group] = tpr
    fpr_rates[group] = fpr
    
    # equal oppurtunity difference
    eod_tpr = tpr_rates[0] - tpr_rates[1]
    eod_fpr = fpr_rates[0] - fpr_rates[1]
    
    return tpr_rates, eod_fpr, eod_tpr


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
                condition = (s >= bins[b] and s <= bins[b+1]) # porque o ultimo bin é o único que tem intervalo fechado no fim
            else:
                condition = (s >= bins[b] and s < bins[b+1]) # restantes têm todos intervalo aberto no fim
            
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
    freq_g0 = np.zeros(num_bins)
    freq_g1 = np.zeros(num_bins)

    for b in range(num_bins):
        if total_g0[b] > 0:
            freq_g0[b] = positive_g0[b] / total_g0[b]
        else:
            freq_g0[b] = 0

        if total_g1[b] > 0:
            freq_g1[b] = positive_g1[b] / total_g1[b]
        else:
            freq_g1[b] = 0

    return freq_g0, freq_g1


        

