import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

DATA_PATH = "data/data.csv"
MODEL_PATH = "model/gas_model.pkl"

print("📂 Reading data...")

# اقرأ الداتا
df = pd.read_csv(DATA_PATH)

print("📊 Columns:", df.columns)
print(df.head())

# لو مفيش داتا كفاية
if len(df) < 5:
    print("❌ Not enough data to train model")
    exit()

# 🔹 HARD LABEL (قواعد مبدئية)
# لو mq2 عالي → خطر
df["label"] = df["mq2"].apply(lambda x: 1 if x > 1500 else 0)

X = df[["mq2", "temp", "hum"]]
y = df["label"]

print("🧠 Training model...")
model = DecisionTreeClassifier()
model.fit(X, y)

# تأكد إن مجلد model موجود
os.makedirs("model", exist_ok=True)

joblib.dump(model, MODEL_PATH)
print("✅ Model trained & saved as model/gas_model.pkl")
