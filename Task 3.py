import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
url = "https://gist.githubusercontent.com/dim4o/c4a67e5309faafcb114df2d35261fa5f/raw/bank.csv"
df = pd.read_csv(url, sep=';')

# 2. Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Prepare features (demographic + behavioral) and target
X = df.drop('y', axis=1)  # all columns except the target
y = df['y']               # target: whether the customer purchased (1) or not (0)

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(" Decision Tree Classifier to Predict Customer Purchase\n")
print(f" Accuracy: {accuracy:.2f}")

# Optional: Focused classification report for the 'yes' class (purchase)
report = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
print("\n🔍 Purchase Prediction Metrics (Class: 'Yes'):")
print(f"Precision: {report['Yes']['precision']:.2f}")
print(f"Recall:    {report['Yes']['recall']:.2f}")
print(f"F1-score:  {report['Yes']['f1-score']:.2f}")

# 7. Confusion Matrix (Visual)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix — Customer Purchase Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# 8. Visualize the Decision Tree
plt.figure(figsize=(14, 6))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree for Customer Purchase Prediction")
plt.tight_layout()
plt.show()
