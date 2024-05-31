import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load dataset
df = pd.read_csv('data_latih.csv')

# Preprocessing
df['narasi'] = df['judul'] + ' ' + df['narasi']
df = df[['narasi', 'label']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['narasi'], df['label'], test_size=0.2, random_state=42)

# Create a pipeline
pipeline = make_pipeline(
    TfidfVectorizer(),
    RandomOverSampler(), 
    LogisticRegression()
)

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'model.pkl')

# Evaluate the model on test data
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluation results on test data:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
