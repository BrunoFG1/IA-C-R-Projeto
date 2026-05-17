import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = "fairface/FairFace/"
train_csv = path + "train_labels.csv"

df_raw = pd.read_csv(train_csv)

df_train = df_raw[df_raw['race'].isin(['White', 'Black'])].copy()

discount_ages = ["0-2", "3-9", "10-19", "more than 70"]
df_train['age'] = df_train['age'].apply(
    lambda x: 'Discount' if x in discount_ages else 'No Discount'
)

print(df_train)

age_intervals = ["No Discount", "Discount"]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.countplot(data=df_train, x="race", hue="gender", palette="pastel")
plt.title("Distribuição por Raça (Apenas Black e White) e Género")
plt.show()

g = sns.catplot(
    data=df_train,
    x="age",           
    hue="gender",      
    col="race",       
    kind="count",      
    order=age_intervals,
    palette="pastel",
    height=5,          
    aspect=1.2,        
    sharey=False       
)
plt.subplots_adjust(hspace=0.8)
plt.show()

plt.figure(figsize=(10, 6))
heatmap_data = df_train.groupby(["race", "gender"]).size().unstack()
sns.heatmap(data=heatmap_data, annot=True, fmt="d", cmap="Blues")
plt.title("Heatmap: Raça vs Género (Apenas Black e White)")
plt.show()

heatmap_ = df_train.groupby(["race", "age"]).size().unstack().reindex(columns=age_intervals)

plt.figure(figsize=(10, 6))
sns.heatmap(
    data=heatmap_, 
    annot=True,          
    fmt="d",             
    cmap="rocket",       
    cbar=True,
    annot_kws={"size": 11, "weight": "bold"}
)
plt.title("Heatmap: Raça vs Grupo de Desconto")
plt.show()