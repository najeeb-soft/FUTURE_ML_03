import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SAMPLE DATA (Simulating Customer Support Tickets)
data = {
    'ticket_text': [
        "My internet is not working and I have a meeting", 
        "I was charged twice for my monthly subscription",
        "How do I change my account password?",
        "The software crashes every time I open the billing tab",
        "I want to cancel my premium membership immediately",
        "Can you help me set up my profile picture?",
        "Urgent: Server is down for our entire office!",
        "Where can I find the receipt for my last payment?"
    ],
    'category': ['Technical', 'Billing', 'Account', 'Technical', 'Billing', 'Account', 'Technical', 'Billing'],
    'priority': ['High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Low']
}

df = pd.DataFrame(data)

# 2. TEXT CLEANING FUNCTION (Required by Task)
def clean_text(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

df['cleaned_text'] = df['ticket_text'].apply(clean_text)

# 3. FEATURE EXTRACTION (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['cleaned_text'])

# 4. MODEL BUILDING (Predicting Category)
y_cat = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. EVALUATION
print("--- Ticket Category Classification Report ---")
print(classification_report(y_test, model.predict(X_test), zero_division=0))

# 6. VISUALIZATION (Confusion Matrix)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
plt.title('Support Ticket Classification Accuracy')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('classification_matrix.png') # For your GitHub
plt.show()

print("Task 2 complete! Results saved as classification_matrix.png")