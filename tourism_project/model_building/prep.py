
import pandas as pd
from sklearn.model_selection import train_test_split


# Load the raw tourism dataset
df = pd.read_csv("tourism_project/data/tourism.csv")


# Remove identifier/index columns
df.drop(
    columns=["Unnamed: 0", "CustomerID"],
    inplace=True
)


# Target variable
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]


# Split the dataset
# stratify=y keeps the ProdTaken ratio consistent
# across train and test datasets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Save train/test datasets
Xtrain.to_csv(
    "tourism_project/model_building/Xtrain.csv",
    index=False
)

Xtest.to_csv(
    "tourism_project/model_building/Xtest.csv",
    index=False
)

ytrain.to_csv(
    "tourism_project/model_building/ytrain.csv",
    index=False
)

ytest.to_csv(
    "tourism_project/model_building/ytest.csv",
    index=False
)


print("Data prepared: train/test splits written.")

print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)

print("\nProdTaken distribution:")
print(y.value_counts())

print("\nCategorical columns kept as:")
print(
    list(
        X.select_dtypes(include="object").columns
    )
)
