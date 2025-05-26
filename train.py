import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("athlete_events.csv")

# Drop rows with missing Medal values
df = df[df['Medal'].notna()]

# Encode Medal: Gold=1, Silver=2, Bronze=3, No Medal=4
medal_map = {'Gold': 1, 'Silver': 2, 'Bronze': 3, 'No Medal': 4}
df['Medal'] = df['Medal'].map(medal_map)

# Drop rows where Medal is NaN after mapping
df = df[df['Medal'].notna()]

# Encode categorical features
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['Season'] = df['Season'].map({'Summer': 1, 'Winter': 2})
df['Sport'] = df['Sport'].astype('category').cat.codes

# Select features
features = ['Sex', 'Age', 'Height', 'Weight', 'Year', 'Season', 'Sport']
X = df[features]
y = df['Medal'].astype(int)

# Fill any remaining NaNs in X with the column mean
X = X.fillna(X.mean())

# Double-check and print if any NaNs remain
if X.isnull().sum().sum() > 0:
    print("Warning: There are still NaNs in the dataset!")
    print(X.isnull().sum())
    exit()

# Compute class weights
classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model.pkl')

# Evaluate
y_pred = model.predict(X_test)

# Test predictions on random sample
sample = X.sample(20, random_state=42)
sample_scaled = scaler.transform(sample)
predictions = model.predict(sample_scaled)

# Convert Sex, Age, Height, Weight to int for display only
sample_display = sample.copy()
sample_display[['Sex', 'Age', 'Height', 'Weight']] = sample_display[['Sex', 'Age', 'Height', 'Weight']].round().astype(int)

print("\nSample Input:\n", sample_display)
print("Sample Predictions:", predictions)