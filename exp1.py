
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

data = {
    "Age": [20, 25, None, 30, 35],
    "City": ["Pune", "Nashik", "Pune", "Mumbai", "Nashik"],
    "Marks": [60, 70, 80, None, 90]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

df = df.drop_duplicates()
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

le = LabelEncoder()
df["City"] = le.fit_transform(df["City"])

scaler = StandardScaler()
df[["Age", "Marks"]] = scaler.fit_transform(df[["Age", "Marks"]])

X = df[["Age", "City"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nCleaned Data:")
print(df)
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
