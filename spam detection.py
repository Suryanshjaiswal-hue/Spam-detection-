 # Step 1: Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 2: Create a small dataset
messages = [
    "Get vaccinated at your nearest center",        # Public Service
    "Flash sale! 50% off on electronics",           # Promotional
    "Don't forget our dinner at 8",                 # Personal
    "New traffic rules effective from Monday",      # Public Service
    "Buy 1 Get 1 Free on all shoes!",               # Promotional
    "Can you send me the notes?",                   # Personal
    "Weather alert: heavy rainfall expected",       # Public Service
    "Exclusive offer just for you!",                # Promotional
    "Let's catch up soon",                          # Personal
]

labels = [
    "public_service",
    "promotional",
    "personal",
    "public_service",
    "promotional",
    "personal",
    "public_service",
    "promotional",
    "personal"
]

# Step 3: Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X, labels)

# Step 5: Predict new messages
new_messages = [
    "Government urges citizens to stay indoors",
    "Mega sale on smartphones this weekend!",
    "Hey, are you free tomorrow evening?"
]

X_new = vectorizer.transform(new_messages)
predictions = model.predict(X_new)

# Step 6: Show predictions
for msg, label in zip(new_messages, predictions):
    print(f"Message: '{msg}' → Category: {label}")