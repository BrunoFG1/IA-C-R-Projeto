import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = "fairface/FairFace/"

train_csv = path + "train_labels.csv"

df_train = pd.read_csv(train_csv)
print(df_train)

age_intervals = ["3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70 >= "]
# Bar plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 6))
sns.countplot(data=df_train, x="race", hue="gender", palette="pastel")

g = sns.catplot(
    data=df_train,
    x="age",           
    hue="gender",      
    col="race",       
    col_wrap=2,       
    kind="count",      
    order=age_intervals,
    palette="pastel",
    height=4,          
    aspect=1.5,        
    sharey=False       
)
plt.subplots_adjust(hspace=0.8)

plt.show()

# Heatmap para rça e genero
plt.figure(figsize=(14, 8))
heatmap_data = df_train.groupby(["race", "gender"]).size().unstack()
sns.heatmap(data=heatmap_data, annot=True, fmt="d")

plt.show()
age_intervals = ["3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
heatmap_ = df_train.groupby(["race", "age"]).size().unstack().reindex(columns=age_intervals)

# Heatmap para idade e raça

plt.figure(figsize=(10,8))
sns.heatmap(
    data=heatmap_, 
    annot=True,          
    fmt="d",             
    cmap="rocket",       
    cbar=True,
    annot_kws={"size": 10, "weight": "bold"}
)
plt.show()
