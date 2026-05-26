import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exibir_relatorio_consola(threshold, bins, antes, depois):
    """
    Imprime de forma estruturada os resultados das métricas de fairness na consola.
    """
    print("Resultado das Métricas de Fairness (Independência, Separação, Suficiência)")
    print("="*60)
    print(f"Limiar de Decisão Biométrico (Distância Limiar): {threshold}")
    print("-"*60)
    print("1. Independência (Paridade Demográfica):")
    print(f"  - Diferença de Paridade Estatística (SPD): {antes['spd']:.4f} (Ideal: 0)")
    print(f"  - Rácio de Impacto Disparate (DI):         {antes['di']:.4f} (Ideal: 1.0, Regra dos 80%: >0.8)")
    print("-"*60)
    print("2. Separação (Igualdade de Oportunidades):")
    print(f"  - Grupo WHITE (0) -> TPR: {antes['tpr'][0]:.4f} | FPR: {antes['fpr'][0]:.4f}")
    print(f"  - Grupo BLACK (1) -> TPR: {antes['tpr'][1]:.4f} | FPR: {antes['fpr'][1]:.4f}")
    print(f"  - Diferença de TPR (Equalized Odds TPR):  {antes['tpr'][0]-antes['tpr'][1]:.4f} (Ideal: 0)")
    print(f"  - Diferença de FPR (Equalized Odds FPR):  {antes['fpr'][0]-antes['fpr'][1]:.4f} (Ideal: 0)")
    print("-"*60)
    print("3. Suficiência (Calibração por Grupo):")
    print("  Intervalo de Score  |  Precisão Real WHITE  |  Precisão Real BLACK")
    for b in range(len(bins)-1):
        print(f"  [{bins[b]:.1f} - {bins[b+1]:.1f}]       |        {antes['calib_w'][b]:.4f}        |        {antes['calib_b'][b]:.4f}")
    print("="*60)

    print("\n" + " " * 15 + "Resultados após Mitigação (Calibração de Platt)")
    print("="*60)
    print("1. Independência (Paridade Demográfica Corrigida):")
    print(f"  - Novo SPD: {depois['spd']:.4f} (Ideal: 0) | Novo DI: {depois['di']:.4f} (Ideal: 1.0)")
    print("-"*60)
    print("2. Separação (Igualdade de Oportunidades Corrigida):")
    print(f"  - Novo WHITE -> TPR: {depois['tpr'][0]:.4f} | FPR: {depois['fpr'][0]:.4f}")
    print(f"  - Novo BLACK -> TPR: {depois['tpr'][1]:.4f} | FPR: {depois['fpr'][1]:.4f}")
    print(f"  - Nova Diferença de TPR (EOD TPR): {depois['tpr'][0]-depois['tpr'][1]:.4f} (Ideal: 0)")
    print(f"  - Nova Diferença de FPR (EOD FPR): {depois['fpr'][0]-depois['fpr'][1]:.4f} (Ideal: 0)")
    print("-"*60)
    print("3. Suficiência (Nova Calibração por Grupo):")
    print("  Intervalo de Score  |  Precisão PÓS-WHITE  |  Precisão PÓS-BLACK")
    for b in range(len(bins)-1):
        print(f"  [{bins[b]:.1f} - {bins[b+1]:.1f}]       |        {depois['calib_w'][b]:.4f}       |        {depois['calib_b'][b]:.4f}")
    print("="*60)

def gerar_graficos_projeto(bins, antes, depois):
    """
    Agrupa as funções de plot para exportar os arquivos de imagem PNG.
    """
    sns.set_theme(style="whitegrid")
    
    # comparacao de sem calibração vs com calibração
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    width = 0.06
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    ax1.bar([c - width/2 for c in bin_centers], antes['calib_w'], width, label='WHITE (Antes)', color='#1f77b4')
    ax1.bar([c + width/2 for c in bin_centers], antes['calib_b'], width, label='BLACK (Antes)', color='#ff7f0e')
    ax1.plot([0, 1], [0, 1], '--', color='gray', label='Calibração Ideal')
    ax1.set_title('Sem Calibração (Scores Brutos)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Intervalo de Score', fontsize=12)
    ax1.set_ylabel('Precisão Real (Proporção de Matches)', fontsize=12)
    ax1.set_xticks(bins)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    
    ax2.bar([c - width/2 for c in bin_centers], depois['calib_w'], width, label='WHITE (Pós-Platt)', color='#2ca02c')
    ax2.bar([c + width/2 for c in bin_centers], depois['calib_b'], width, label='BLACK (Pós-Platt)', color='#d62728')
    ax2.plot([0, 1], [0, 1], '--', color='gray', label='Calibração Ideal')
    ax2.set_title('Com Calibração (Pós-Processamento Platt)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Intervalo de Score', fontsize=12)
    ax2.set_xticks(bins)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height*100:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(ax1.patches[:5], ax1) # White antes
    autolabel(ax1.patches[5:], ax1) # Black antes
    autolabel(ax2.patches[:5], ax2) # White depois
    autolabel(ax2.patches[5:], ax2) # Black depois
    
    fig1.tight_layout()
    fig1.savefig('comparacao_calibracao_fairness.png', dpi=300)
    print("\n[SUCESSO] Gráfico guardado como 'comparacao_calibracao_fairness.png'!")

    # grafico de paridade demográfica (antes e depois da mitigação) e igualdade de opurtunidades (antes e depois da mitigação)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
    
    dados_eo = {
        'Métrica': ['TPR', 'TPR', 'TPR', 'TPR', 'FPR', 'FPR', 'FPR', 'FPR'],
        'Cenário': ['Antes', 'Antes', 'Pós-Platt', 'Pós-Platt', 'Antes', 'Antes', 'Pós-Platt', 'Pós-Platt'],
        'Grupo Étnico': ['WHITE (0)', 'BLACK (1)', 'WHITE (0)', 'BLACK (1)', 'WHITE (0)', 'BLACK (1)', 'WHITE (0)', 'BLACK (1)'],
        'Valor': [antes['tpr'][0], antes['tpr'][1], depois['tpr'][0], depois['tpr'][1], antes['fpr'][0], antes['fpr'][1], depois['fpr'][0], depois['fpr'][1]]
    }
    df_eo = pd.DataFrame(dados_eo)
    df_eo['Métrica_Cenário'] = df_eo['Métrica'] + ' (' + df_eo['Cenário'] + ')'
    
    sns.barplot(data=df_eo, x='Métrica_Cenário', y='Valor', hue='Grupo Étnico', palette=['#1f77b4', '#ff7f0e'], ax=ax3, edgecolor='black')
    ax3.set_title('Igualdade de Oportunidades (Equalized Odds)', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 1.1])
    
    for p in ax3.patches:
        h = p.get_height()
        if h > 0:
            ax3.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9, weight='bold')

    cenarios = ['Antes da Calibração', 'Pós-Processamento Platt']
    x_indices = np.arange(len(cenarios))
    width_bar = 0.35
    
    rects_spd = ax4.bar(x_indices - width_bar/2, [antes['spd'], depois['spd']], width_bar, label='SPD (Ideal: 0.0)', color='#9467bd', edgecolor='black')
    rects_di = ax4.bar(x_indices + width_bar/2, [antes['di'], depois['di']], width_bar, label='DI (Ideal: 1.0)', color='#bcbd22', edgecolor='black')
    
    ax4.axhline(1.0, linestyle='--', color='green', alpha=0.6, label='DI Perfeito (1.0)')
    ax4.axhline(0.8, linestyle=':', color='red', alpha=0.5, label='Regra dos 80%')
    ax4.set_title('Métricas Agregadas de Paridade Demográfica', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels(cenarios)
    ax4.set_ylim([0, max(antes['di'], depois['di']) * 1.25])
    ax4.legend(loc='upper left')
    
    def label_bars(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax4.annotate(f'{height:.4f}', (rect.get_x() + rect.get_width() / 2, height), ha='center', va='bottom', xytext=(0, 3), textcoords="offset points", fontsize=9, fontweight='bold')
                            
    label_bars(rects_spd)
    label_bars(rects_di)
    
    fig2.tight_layout()
    fig2.savefig('comparacao_taxas_fairness.png', dpi=300)
    print("Gráfico de taxas guardado como 'comparacao_taxas_fairness.png'")
    
    plt.show()