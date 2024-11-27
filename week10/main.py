import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("titanic.csv")
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin'], axis=1)
df["Sex"] = df["Sex"].replace({"male" : 0, "female" : 1})

selected_columns = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
df_selected = df[selected_columns]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_selected.values)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"], index=df.index)

print("Original Data:")
print(df.head())
print("\nPrincipal Components:")
principal_df["Survived"] = df["Survived"]
print(principal_df.head())

