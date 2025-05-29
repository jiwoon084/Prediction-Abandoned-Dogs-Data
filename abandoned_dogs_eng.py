import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier

# --- Feature Engineering & Dirty Data Injection ---

# 1. Load original data
df = pd.read_csv("train_breed_category.csv")

# 2. Filter to include only dogs
df = df[df['Type'] == 1].reset_index(drop=True)

# 3. Create HasName feature (1 if name exists, 0 if empty or missing)
df['HasName'] = df['Name'].notnull() & (df['Name'].str.strip() != '')
df['HasName'] = df['HasName'].astype(int)

# 4. Prepare clean copy
df_clean = df.copy()

# 5. Define column sets
categorical_candidate_cols = ['Type', 'HasName', 'Gender', 'Color1', 'Color2', 'Color3',
                              'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                              'Sterilized', 'Health', 'Breed']
categorical_cols = [col for col in categorical_candidate_cols if col in df.columns]
numerical_cols = ['Age', 'Quantity', 'Fee']
target_col = 'AdoptionSpeed'

# 6. Label encoding
df_encoded = df.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# 7. Feature selection
X = df_encoded[categorical_cols + numerical_cols].fillna(0)
y = df_encoded[target_col]
kbest = SelectKBest(score_func=chi2, k=15)
X_selected = kbest.fit_transform(X, y)
selected_columns = [col for col, keep in zip(X.columns, kbest.get_support()) if keep]

print("✅ Selected key features:", selected_columns)

# 8. Final DataFrame
df_final = df_clean[selected_columns + [target_col]].copy()

# 9. Inject dirty data into 5% of rows
num_rows = len(df_final)
dirty_count = int(num_rows * 0.05)
dirty_indices = random.sample(range(num_rows), dirty_count)

def insert_dirty(row):
    dirty_type = random.choice(['Age', 'Quantity', 'Sterilized', 'Vaccinated', 'Dewormed', 'Fee'])
    if dirty_type == 'Age':
        row['Age'] = random.choice([-5, 0, 0.5, 250])
    elif dirty_type == 'Quantity':
        row['Quantity'] = random.choice([0, -1, -3])
    elif dirty_type == 'Sterilized':
        row['Sterilized'] = random.choice([-2, 5, 10])
    elif dirty_type == 'Vaccinated':
        row['Vaccinated'] = random.choice([-1, 4, 99])
    elif dirty_type == 'Dewormed':
        row['Dewormed'] = random.choice([-3, 7, 100])
    elif dirty_type == 'Fee':
        row['Fee'] = random.choice([9999, 99999])
    return row

df_final.iloc[dirty_indices] = df_final.iloc[dirty_indices].apply(insert_dirty, axis=1)

# 10. Save
df_final.to_csv("train_dogs_dirty_final_for_project.csv", index=False)
print("🎉 Completed: Dog data with HasName feature and 5% dirty data created!")

# --- Dirty Data Cleaning ---

# 1. Load dataset
df2 = pd.read_csv("train_dogs_dirty_final_for_project.csv")

# 2. Convert all columns to numeric where possible (non-numeric → NaN)
for col in df2.columns:
    try:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    except Exception:
        pass

# 3. Drop rows where target variable 'AdoptionSpeed' is missing
df2 = df2.dropna(subset=['AdoptionSpeed'])

# 4. Clean 'Age' (valid: 1 ≤ Age ≤ 192 and must be integer)
valid_age = df2['Age'][(df2['Age'] >= 1) & (df2['Age'] <= 192) & (df2['Age'] == df2['Age'].astype(int))]
median_age = valid_age.median()
df2['Age'] = df2['Age'].apply(lambda x: median_age if x < 1 or x != int(x) else min(x, 192))

# 5. Clean 'Quantity' (0 or less → 1)
df2['Quantity'] = df2['Quantity'].apply(lambda x: 1 if x <= 0 else x)

# 6. Clean categorical values: 'Vaccinated', 'Dewormed', 'Sterilized' (only 1, 2, 3)
for col in ['Vaccinated', 'Dewormed', 'Sterilized']:
    valid_values = df2[col][df2[col].isin([1, 2, 3])]
    if not valid_values.empty:
        mode_val = valid_values.mode()[0]
        df2[col] = df2[col].apply(lambda x: mode_val if x not in [1, 2, 3] else x)

# 7. Clip 'Fee' at 95th percentile
max_fee = df2['Fee'].quantile(0.95)
df2['Fee'] = df2['Fee'].apply(lambda x: max_fee if x > max_fee else x)

# 8. Clean 'AdoptionSpeed' (only allow 0–4)
valid_speed = df2['AdoptionSpeed'][df2['AdoptionSpeed'].isin([0, 1, 2, 3, 4])]
mode_speed = valid_speed.mode()[0]
df2.loc[~df2['AdoptionSpeed'].isin([0, 1, 2, 3, 4]), 'AdoptionSpeed'] = mode_speed
df2['AdoptionSpeed'] = df2['AdoptionSpeed'].astype(int)

# 9. Clean 'Breed' (only 1, 2, 3 allowed)
if 'Breed' in df2.columns:
    valid_breed = df2['Breed'][df2['Breed'].isin([1, 2, 3])]
    if not valid_breed.empty:
        mode_breed = valid_breed.mode()[0]
        df2.loc[~df2['Breed'].isin([1, 2, 3]), 'Breed'] = mode_breed
        df2['Breed'] = df2['Breed'].astype(int)

# 10. Clean 'HasName' (only 0 or 1)
if 'HasName' in df2.columns:
    valid_name = df2['HasName'][df2['HasName'].isin([0, 1])]
    if not valid_name.empty:
        mode_name = valid_name.mode()[0]
        df2.loc[~df2['HasName'].isin([0, 1]), 'HasName'] = mode_name
        df2['HasName'] = df2['HasName'].astype(int)

# 11. Clean 'Gender' (only 1, 2, 3 allowed)
if 'Gender' in df2.columns:
    valid_gender = df2['Gender'][df2['Gender'].isin([1, 2, 3])]
    if not valid_gender.empty:
        mode_gender = valid_gender.mode()[0]
        df2.loc[~df2['Gender'].isin([1, 2, 3]), 'Gender'] = mode_gender
        df2['Gender'] = df2['Gender'].astype(int)

# 12. Clean 'Color1', 'Color2', 'Color3' (non-negative only)
for col in ['Color1', 'Color2', 'Color3']:
    if col in df2.columns:
        valid_color = df2[col][df2[col] >= 0]
        if not valid_color.empty:
            mode_color = valid_color.mode()[0]
            df2.loc[df2[col] < 0, col] = mode_color
            df2[col] = df2[col].astype(int)

# 13. Clean 'MaturitySize' (only 1–4)
if 'MaturitySize' in df2.columns:
    valid_size = df2['MaturitySize'][df2['MaturitySize'].isin([1, 2, 3, 4])]
    if not valid_size.empty:
        mode_size = valid_size.mode()[0]
        df2.loc[~df2['MaturitySize'].isin([1, 2, 3, 4]), 'MaturitySize'] = mode_size
        df2['MaturitySize'] = df2['MaturitySize'].astype(int)

# 14. Clean 'FurLength' (only 1–3)
if 'FurLength' in df2.columns:
    valid_fur = df2['FurLength'][df2['FurLength'].isin([1, 2, 3])]
    if not valid_fur.empty:
        mode_fur = valid_fur.mode()[0]
        df2.loc[~df2['FurLength'].isin([1, 2, 3]), 'FurLength'] = mode_fur
        df2['FurLength'] = df2['FurLength'].astype(int)

# 15. Clean 'Health' (only 1–3)
if 'Health' in df2.columns:
    valid_health = df2['Health'][df2['Health'].isin([1, 2, 3])]
    if not valid_health.empty:
        mode_health = valid_health.mode()[0]
        df2.loc[~df2['Health'].isin([1, 2, 3]), 'Health'] = mode_health
        df2['Health'] = df2['Health'].astype(int)

# 16. Final imputation for any remaining missing values
for col in df2.columns:
    if col == 'AdoptionSpeed':
        continue
    if df2[col].isnull().sum() > 0:
        if df2[col].dtype == 'object':
            df2[col] = df2[col].fillna(df2[col].mode()[0])
        else:
            df2[col] = df2[col].fillna(df2[col].median())

# 17. Save cleaned dataset
df2.to_csv("cleaned_dogs_data.csv", index=False)
print("✅ Data cleaning complete → 'cleaned_dogs_data.csv' saved.")

# --- Data Scaling ---

# 1. Load the CSV file
df3 = pd.read_csv('cleaned_dogs_data.csv')

# 2. Select numerical columns
numerical_cols = ['Age', 'Quantity', 'Fee']

# 3. Initialize and apply StandardScaler
scaler = StandardScaler()
df3[numerical_cols] = scaler.fit_transform(df3[numerical_cols])

# 4. Display the first 5 rows of scaled data
print("First 5 rows of scaled data:")
print(df3.head())

# 5. Save the scaled data to a CSV file
df3.to_csv('Abandoned Dog Data.csv', index=False)

# --- Regression ---

# 1. Import data
df4 = pd.read_csv("Abandoned Dog Data.csv")
target_col = 'AdoptionSpeed'
X = df4.drop(columns=[target_col])
y = df4[target_col]

# 2. Model definition
models = {
    "LinearRegression": LinearRegression(),
    "PolynomialRegression_deg2": make_pipeline(PolynomialFeatures(2), Ridge()),
    "PolynomialRegression_deg3": make_pipeline(PolynomialFeatures(3), Ridge())
}

# 3. K-Fold setting
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = {name: {'RMSE': [], 'MAE': [], 'R2': []} for name in models}

# 4. K-Fold application
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[name]['RMSE'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
        results[name]['MAE'].append(mean_absolute_error(y_val, y_pred))
        results[name]['R2'].append(r2_score(y_val, y_pred))

# 5. Summary of Average Results
avg_results = {
    name: {metric: np.mean(scores) for metric, scores in metrics.items()}
    for name, metrics in results.items()
}
results_df = pd.DataFrame(avg_results).T
print("📊 10-Fold Regression Model Performance (Average):")
print(results_df)

# 6. Visualization
plt.figure(figsize=(18, 6))
bar_width = 0.25
index = np.arange(len(results_df))
plt.bar(index, results_df['RMSE'], bar_width, label='RMSE')
plt.bar(index + bar_width, results_df['MAE'], bar_width, label='MAE')
plt.bar(index + 2*bar_width, results_df['R2'], bar_width, label='R²')
plt.xticks(index + bar_width, results_df.index, rotation=15)
plt.title('10-Fold Regression Performance Comparison (Average)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Multi-Class Classification ---

# 1. Load & separate the data
df5 = pd.read_csv("Abandoned Dog Data.csv")
target_col = 'AdoptionSpeed'
X = df5.drop(columns=[target_col])
y = df5[target_col]

# 2. Categorical encoding if needed
categorical_cols = X.select_dtypes(include='object').columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 3. Stratified K-Fold setting
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
confusion_all = np.zeros((5, 5), dtype=int)

# 4. Model learning and evaluation
for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    accuracies.append(accuracy_score(y_val, y_pred))
    confusion_all += confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3, 4])

# 5. Result output
print(f"\n✅ Average Accuracy (10-Fold): {np.mean(accuracies):.4f}")
print("\n✅ Total Confusion Matrix (Accumulated):")
for row in confusion_all:
    print("[" + " ".join(f"{v:3d}" for v in row) + "]")
print("\n✅ Final Classification Report:")
print(classification_report(y_val, y_pred, digits=4))

# --- Binary Classification (Simplified) ---

# 1. Import data
df6 = pd.read_csv("Abandoned Dog Data.csv")

# 2. Target binarization: 0~3 → 1 (adopted), 4 → 0 (not adopted)
df6['AdoptionBinary'] = df6['AdoptionSpeed'].apply(lambda x: 1 if x < 4 else 0)

# 3. X, y setting
X = df6.drop(columns=['AdoptionSpeed', 'AdoptionBinary'])
y = df6['AdoptionBinary']

# 4. Categorical variable encoding
categorical_cols = X.select_dtypes(include='object').columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 5. Stratified K-Fold setting
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
conf_matrix_total = np.zeros((2, 2), dtype=int)

# 6. Repeat the model learning and evaluation
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
    conf_matrix_total += confusion_matrix(y_val, y_pred, labels=[0, 1])

    print(f"[Fold {fold}] Accuracy: {acc:.4f}")

# 7. Print the average accuracy and result
def format_conf_matrix(matrix):
    return "\n".join(["[" + " ".join(f"{val:3d}" for val in row) + "]" for row in matrix])

print(f"\n✅ Average Accuracy (10-Fold): {np.mean(accuracies):.4f}\n")
print("✅ Cumulative Confusion Matrix:")
print(format_conf_matrix(conf_matrix_total))
print("\n✅ Final Fold Classification Report:")
print(classification_report(y_val, y_pred, digits=4))