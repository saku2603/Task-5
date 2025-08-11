# Import libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('heart.csv')  # Make sure 'heart.csv' is in the same folder

# Data Exploration
print("\n=== Data Overview ===")
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())
print("\nMissing values:")
print(data.isnull().sum())
print("\nTarget distribution:")
print(data['target'].value_counts())

# Separate features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree with regularization
print("\n=== Decision Tree ===")
dt = DecisionTreeClassifier(
    max_depth=4,  # Reduced to prevent overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt.fit(X_train, y_train)

# Evaluate Decision Tree
print("\nTraining Performance:")
print(classification_report(y_train, dt.predict(X_train)))
print("\nTest Performance:")
print(classification_report(y_test, dt.predict(X_test)))

# Confusion Matrix
cm = confusion_matrix(y_test, dt.predict(X_test))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Decision Tree Confusion Matrix')
plt.savefig('dt_confusion_matrix.png')
plt.show()

# Tree Visualization - Try Graphviz, fallback to plot_tree
try:
    # Add Graphviz to PATH if needed (update with your actual path)
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    
    dot_data = export_graphviz(dt, out_file=None, 
                             feature_names=X.columns,  
                             class_names=['No Disease', 'Disease'],  
                             filled=True, rounded=True,  
                             special_characters=True)  
    graph = graphviz.Source(dot_data, format='png') 
    graph.render("heart_disease_tree", view=False)
    print("\nDecision tree visualization saved as 'heart_disease_tree.png'")
except Exception as e:
    print(f"\nGraphviz error: {e}")
    print("Using matplotlib tree visualization instead:")
    plt.figure(figsize=(20,10))
    plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
    plt.savefig('tree_visualization.png')
    plt.show()

# Train Random Forest with regularization
print("\n=== Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate Random Forest
print("\nTraining Performance:")
print(classification_report(y_train, rf.predict(X_train)))
print("\nTest Performance:")
print(classification_report(y_test, rf.predict(X_test)))

# Feature Importance
importances = rf.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head(5))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances in Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Cross-Validation
print("\n=== Cross-Validation ===")
dt_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print(f"Decision Tree CV Accuracy: {dt_scores.mean():.2f} (+/- {dt_scores.std():.2f})")
print(f"Random Forest CV Accuracy: {rf_scores.mean():.2f} (+/- {rf_scores.std():.2f})")

# Additional Metrics
print("\n=== Additional Metrics ===")
print("Decision Tree Feature Importances:")
print(dt.feature_importances_)
print("\nRandom Forest OOB Score:", rf.oob_score_)