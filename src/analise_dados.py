import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from fairness_metrics import demographic_parity, equal_opportunity, calibration_by_group

path = "fairface/FairFace/"
train_csv = path + "train_labels.csv"

df_train = pd.read_csv(train_csv)

race_mapping = {
    'White': 'White',
    'Middle Eastern': 'White',
    'Latino_Hispanic': 'White',
    'Black': 'Black',
    'East Asian': 'Black',
    'Southeast Asian': 'Black',
    'Indian': 'Black'
}
df_train['race'] = df_train['race'].map(race_mapping)

discount_ages = ["0-2", "3-9", "10-19", "more than 70"]
df_train['age'] = df_train['age'].apply(
    lambda x: 'Discount' if x in discount_ages else 'No Discount'
)

print(df_train)

age_intervals = ["No Discount", "Discount"]

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

plt.figure(figsize=(14, 8))
heatmap_data = df_train.groupby(["race", "gender"]).size().unstack()
sns.heatmap(data=heatmap_data, annot=True, fmt="d")

plt.show()

heatmap_ = df_train.groupby(["race", "age"]).size().unstack().reindex(columns=age_intervals)

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