
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("tourism_project/data/tourism.csv")

# Remove identifier/index columns
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# NOTE: Categorical columns such as TypeofContact, Occupation,
# Gender, ProductPitched, MaritalStatus, and Designation
# are intentionally left as raw strings.
# The training pipeline should encode them consistently
# during model training and prediction.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) ProdTaken ratio
# consistent across train/test splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Target distribution:")
print(y.value_counts())

print("Categorical columns kept as:",
      list(X.select_dtypes(include="object").columns))
