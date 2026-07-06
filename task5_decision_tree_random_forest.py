import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("heart.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display dataset shape
print("\nDataset Shape:", df.shape)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:", accuracy)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True
)

plt.title("Decision Tree")
plt.show()

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

# Compare Decision Tree and Random Forest
print("\nModel Comparison")
print("Decision Tree Accuracy :", accuracy)
print("Random Forest Accuracy :", rf_accuracy)

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance)

# Cross Validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Average Cross Validation Accuracy:", cv_scores.mean())

# Decision Tree with Limited Depth
dt_depth = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_depth.fit(X_train, y_train)

depth_pred = dt_depth.predict(X_test)

depth_accuracy = accuracy_score(y_test, depth_pred)

print("\nDecision Tree Accuracy (max_depth=3):", depth_accuracy)

# Plot Feature Importance
plt.figure(figsize=(10,6))

plt.bar(importance["Feature"], importance["Importance"])

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()