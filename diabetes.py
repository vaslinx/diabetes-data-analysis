import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.info())
print(df.describe())

# -----------------------------
# 2. Data Cleaning
# -----------------------------

# Replace unrealistic zero values with median
(df == 0).sum()

cols_with_zero_nan=["Glucose","BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
for col in cols_with_zero_nan:
    df[col]=df[col].replace(0, np.nan)
df.isnull().sum()

df["Glucose"].fillna(df["Glucose"].median())
df["BloodPressure"].fillna(df["BloodPressure"].median())
df["SkinThickness"].fillna(df["SkinThickness"].median())
df["Insulin"].fillna(df["Insulin"].median())
df["BMI"].fillna(df["BMI"].median())
df["DiabetesPedigreeFunction"].fillna(df["DiabetesPedigreeFunction"].median())
df["Age"].fillna(df["Age"].median())
df.isnull().sum()

# -----------------------------
# 3. Feature Engineering
# -----------------------------

# Create Age Groups
bins = [0, 17, 40, 60, 80, 120]
labels = [
    '0-17 (child)',
    '18-40 (young)',
    '41-60 (adult)',
    '61-80 (senior)',
    '81+ (older)'
]
df['AgeGroup']=pd.cut(df['Age'], bins=bins, labels=labels)

# -----------------------------
# 4. Data Scaling
# -----------------------------

from sklearn.preprocessing import StandardScaler

df_scaled=df.copy()
features=df_scaled.drop(['Outcome', 'AgeGroup'], axis=1)
scaler=StandardScaler()
scaled=scaler.fit_transform(features)

df_scaled[features.columns]=scaled
df_scaled.head()

# -----------------------------
# 5. Correlation Heatmap
# -----------------------------

from sklearn.preprocessing import StandardScaler

df_scaled=df.copy()
features=df_scaled.drop(['Outcome', 'AgeGroup'], axis=1)
scaler=StandardScaler()
scaled=scaler.fit_transform(features)

df_scaled[features.columns]=scaled
numeric_df = df_scaled.drop(columns=['AgeGroup'])

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# -----------------------------
# 6. Distribution Plots
# -----------------------------

features_list= ['Glucose', 'BMI', 'BloodPressure', 'Age']
for col in features_list:
    plt.figure(figsize=(7, 4))
    
    sns.kdeplot(df[df['Outcome']==0][col],label="No diabetes",fill=True) 
    sns.kdeplot(df[df['Outcome']==1][col],label="Diabetes", fill=True) 
    
    plt.title(f"Distribution of {col} by Outcome")
    plt.legend()
    plt.show()

# -----------------------------
# 7. Age Group Analysis
# -----------------------------

lt.figure(figsize=(7,5))
sns.countplot(x='AgeGroup', hue='Outcome', data=df, palette='pastel')
plt.title("Diabetes distribution across Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 8. Glucose level Analysis
# -----------------------------

plt.figure(figsize=(7,4))

sns.boxplot(x='AgeGroup', y='Glucose', hue='AgeGroup', data=df, legend=False, palette='pastel')

plt.title("Glucose levels by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Glucose level")
plt.show()

# -----------------------------
# 9. Feature Importance
# -----------------------------

x=df_scaled.drop(['Outcome', 'AgeGroup'], axis=1)
y=df_scaled['Outcome']

model=RandomForestClassifier(random_state=42)

model.fit(x,y)

importances=model.feature_importances_
features=x.columns
plt.figure(figsize=(8,9))

sns.barplot(x=importances, y=features, hue=importances, legend=False, palette='pastel')

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
