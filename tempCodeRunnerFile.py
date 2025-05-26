import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load the data
df = pd.read_csv("athlete_events.csv", header=None, low_memory=False)

# Set column names
df.columns = [
    "ID", "Name", "Sex", "Age", "Height", "Weight", "Team", "NOC",
    "Games", "Year", "Season", "City", "Sport", "Event", "Medal"
]

# Drop irrelevant columns
df = df.drop(columns=["ID", "Name", "Team", "NOC", "Games", "City", "Event"])

# Keep only rows where Medal is present (we want to predict medal type)
df = df[df["Medal"].notna()]

# Convert numeric columns, coercing errors to NaN
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df["Height"] = pd.to_numeric(df["Height"], errors='coerce')
df["Weight"] = pd.to_numeric(df["Weight"], errors='coerce')

# Encode categorical columns using LabelEncoder
le_sex = LabelEncoder()
le_season = LabelEncoder()
le_sport = LabelEncoder()
le_medal = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"].astype(str))
df["Season"] = le_season.fit_transform(df["Season"].astype(str))
df["Sport"] = le_sport.fit_transform(df["Sport"].astype(str))
df["Medal_encoded"] = le_medal.fit_transform(df["Medal"].astype(str))

# Drop original Medal column
df = df.drop(columns=["Medal"])

# Drop rows with any NaNs after conversion and encoding
df = df.dropna()

# Features and label
X = df.drop(columns=["Medal_encoded"])
y = df["Medal_encoded"]

# Remove classes with fewer than 2 samples (needed for stratify)
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
mask = y.isin(valid_classes)

X = X.loc[mask]
y = y.loc[mask]

# Train/test split with stratify=y to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model and encoders for future use
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump({
        "Sex": le_sex,
        "Season": le_season,
        "Sport": le_sport,
        "Medal": le_medal
    }, f)

# Show some sample inputs and their predicted outputs
print("Sample Input:", X_test.head(20))
print("Sample Predictions:", y_pred[:20])
