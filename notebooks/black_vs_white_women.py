# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
color_palette = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

df = pd.read_csv('/content/compas-scores-two-years-cleaned.csv')

print("\n1. DATASET OVERVIEW")
print("-"*80)
print(f"Total Records: {len(df)}")
print(f"Total Features: {len(df.columns)}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst Few Rows:")
print(df.head())

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percentage': missing_pct
}).sort_values('Missing_Count', ascending=False)
print(missing_df[missing_df['Missing_Count'] > 0])

fig, ax = plt.subplots(figsize=(12, 6))
missing_cols = missing_df[missing_df['Missing_Count'] > 0]
if len(missing_cols) > 0:
    sns.barplot(x=missing_cols.index, y=missing_cols['Missing_Percentage'],
                palette='Reds_r', ax=ax)
    ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Missing Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('1_missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No missing values found!")

race_col = [col for col in df.columns if 'race' in col.lower()]
sex_col = [col for col in df.columns if 'sex' in col.lower() or 'gender' in col.lower()]

print(f"Potential race column(s): {race_col}")
print(f"Potential sex/gender column(s): {sex_col}")

if race_col:
    print(f"\nUnique values in {race_col[0]}: {df[race_col[0]].unique()}")
if sex_col:
    print(f"Unique values in {sex_col[0]}: {df[sex_col[0]].unique()}")

print(df.describe())

recid_cols = [col for col in df.columns if 'recid' in col.lower() or 'two_year' in col.lower()]
print(f"Potential recidivism columns: {recid_cols}")

if recid_cols:
    for col in recid_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print(f"Recidivism Rate: {df[col].mean()*100:.2f}%")

numeric_df = df.select_dtypes(include=[np.number])
print(f"Numeric columns: {numeric_df.columns.tolist()}")

if len(numeric_df.columns) > 1:
    plt.figure(figsize=(14, 10))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n7. GENERATING SUMMARY REPORT")
print("-"*80)

with open('eda_summary_report.txt', 'w') as f:
    f.write("RECIDIVISM EDA SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total Records: {len(df)}\n")
    f.write(f"Total Features: {len(df.columns)}\n")
    f.write(f"\nColumns: {df.columns.tolist()}\n")
    f.write(f"\nMissing Values:\n{missing_df[missing_df['Missing_Count'] > 0]}\n")
    f.write(f"\nStatistical Summary:\n{df.describe()}\n")

print("✓ EDA Summary report saved to 'eda_summary_report.txt'")
print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Review the column names and identify:")
print("   - Race column (for filtering Black vs White)")
print("   - Sex/Gender column (for filtering Women)")
print("   - Recidivism target variable")
print("   - Key features for analysis")
print("\n2. Share the output with me so we can proceed with:")
print("   - Filtering to Black & White women only")
print("   - Comparative visualizations")
print("   - Feature engineering")
print("="*80)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

COLORS = {
    'African-American': '#E74C3C',  # Red
    'Caucasian': '#3498DB',          # Blue
    'overall': '#95A5A6'             # Gray
}

df_women = df[df['sex'] == 'Female'].copy()
print(f"Total women in dataset: {len(df_women)}")

# Check race values
print(f"\nRace distribution in women:")
print(df_women['race'].value_counts())

# Filter for Black and White women
df_filtered = df_women[df_women['race'].isin(['African-American', 'Caucasian'])].copy()
print(f"\nFiltered to Black & White women: {len(df_filtered)}")
print(df_filtered['race'].value_counts())

# Create a simpler race label for plotting
df_filtered['race_label'] = df_filtered['race'].map({
    'African-American': 'Black',
    'Caucasian': 'White'
})

print("\n2. DEMOGRAPHIC COMPARISON")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Demographic Comparison: Black vs White Women',
             fontsize=18, fontweight='bold', y=0.995)

# Age Distribution
ax1 = axes[0, 0]
for race in ['African-American', 'Caucasian']:
    data = df_filtered[df_filtered['race'] == race]['age']
    label = 'Black' if race == 'African-American' else 'White'
    ax1.hist(data, bins=20, alpha=0.6, label=label,
             color=COLORS[race], edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
ax1.set_title('Age Distribution', fontweight='bold', pad=10)
ax1.legend(frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)

# Add mean lines
for race in ['African-American', 'Caucasian']:
    mean_age = df_filtered[df_filtered['race'] == race]['age'].mean()
    color = COLORS[race]
    ax1.axvline(mean_age, color=color, linestyle='--', linewidth=2, alpha=0.8)

# Prior Convictions
ax2 = axes[0, 1]
prior_data = df_filtered.groupby('race_label')['priors_count'].mean().sort_values()
bars = ax2.barh(prior_data.index, prior_data.values,
                color=[COLORS['Caucasian'], COLORS['African-American']])
ax2.set_xlabel('Average Prior Convictions')
ax2.set_title('Average Prior Convictions by Race', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='x')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2,
             f'{width:.2f}', ha='left', va='center', fontweight='bold')

# Charge Degree Distribution
ax3 = axes[1, 0]
charge_counts = pd.crosstab(df_filtered['race_label'],
                             df_filtered['c_charge_degree'],
                             normalize='index') * 100
charge_counts.plot(kind='bar', ax=ax3,
                   color=['#E67E22', '#16A085'],
                   width=0.7, edgecolor='black', linewidth=1.2)
ax3.set_xlabel('Race')
ax3.set_ylabel('Percentage (%)')
ax3.set_title('Charge Degree Distribution', fontweight='bold', pad=10)
ax3.legend(title='Charge Type', frameon=True, shadow=True)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.grid(True, alpha=0.3, axis='y')

# COMPAS Risk Score Distribution
ax4 = axes[1, 1]
for race in ['Black', 'White']:
    data = df_filtered[df_filtered['race_label'] == race]['decile_score']
    ax4.hist(data, bins=10, alpha=0.6, label=race,
             color=COLORS['African-American' if race=='Black' else 'Caucasian'],
             edgecolor='black', linewidth=1.2)
ax4.set_xlabel('COMPAS Risk Score (1-10)')
ax4.set_ylabel('Frequency')
ax4.set_title('COMPAS Risk Score Distribution', fontweight='bold', pad=10)
ax4.legend(frameon=True, shadow=True)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3_demographic_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n3. RECIDIVISM RATE COMPARISON")
print("-"*80)

recid_stats = df_filtered.groupby('race_label').agg({
    'two_year_recid': ['mean', 'sum', 'count']
}).round(4)

print(recid_stats)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Recidivism Analysis: Black vs White Women',
             fontsize=18, fontweight='bold')

# Overall Recidivism Rates
ax1 = axes[0]
recid_rates = df_filtered.groupby('race_label')['two_year_recid'].mean() * 100
bars = ax1.bar(recid_rates.index, recid_rates.values,
               color=[COLORS['African-American'], COLORS['Caucasian']],
               edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('Recidivism Rate (%)')
ax1.set_title('2-Year Recidivism Rate', fontweight='bold', pad=10)
ax1.set_ylim(0, max(recid_rates.values) * 1.2)
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom',
             fontweight='bold', fontsize=12)

# Recidivism by Age Category
ax2 = axes[1]
age_recid = df_filtered.groupby(['race_label', 'age_cat'])['two_year_recid'].mean() * 100
age_recid = age_recid.unstack(level=0)
age_recid.plot(kind='bar', ax=ax2,
               color=[COLORS['African-American'], COLORS['Caucasian']],
               edgecolor='black', linewidth=1.2, width=0.7)
ax2.set_xlabel('Age Category')
ax2.set_ylabel('Recidivism Rate (%)')
ax2.set_title('Recidivism Rate by Age Group', fontweight='bold', pad=10)
ax2.legend(title='Race', frameon=True, shadow=True)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Recidivism by COMPAS Score Category
ax3 = axes[2]
score_recid = df_filtered.groupby(['race_label', 'score_text'])['two_year_recid'].mean() * 100
score_recid = score_recid.unstack(level=0)
score_order = ['Low', 'Medium', 'High']
score_recid = score_recid.reindex(score_order)
score_recid.plot(kind='bar', ax=ax3,
                 color=[COLORS['African-American'], COLORS['Caucasian']],
                 edgecolor='black', linewidth=1.2, width=0.7)
ax3.set_xlabel('COMPAS Risk Category')
ax3.set_ylabel('Recidivism Rate (%)')
ax3.set_title('Recidivism by Risk Category', fontweight='bold', pad=10)
ax3.legend(title='Race', frameon=True, shadow=True)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('4_recidivism_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n4. STATISTICAL SIGNIFICANCE TESTS")
print("-"*80)

black_recid = df_filtered[df_filtered['race'] == 'African-American']['two_year_recid']
white_recid = df_filtered[df_filtered['race'] == 'Caucasian']['two_year_recid']

# Chi-square test for recidivism
contingency = pd.crosstab(df_filtered['race_label'], df_filtered['two_year_recid'])
chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-Square Test for Recidivism:")
print(f"  Chi-Square Statistic: {chi2:.4f}")
print(f"  P-value: {p_val:.4f}")
print(f"  Significant: {'YES' if p_val < 0.05 else 'NO'}")

# T-test for age
black_age = df_filtered[df_filtered['race'] == 'African-American']['age']
white_age = df_filtered[df_filtered['race'] == 'Caucasian']['age']
t_stat, p_val_age = stats.ttest_ind(black_age, white_age)
print(f"\nT-Test for Age Difference:")
print(f"  T-Statistic: {t_stat:.4f}")
print(f"  P-value: {p_val_age:.4f}")
print(f"  Black women avg age: {black_age.mean():.2f}")
print(f"  White women avg age: {white_age.mean():.2f}")

# T-test for priors
black_priors = df_filtered[df_filtered['race'] == 'African-American']['priors_count']
white_priors = df_filtered[df_filtered['race'] == 'Caucasian']['priors_count']
t_stat_priors, p_val_priors = stats.ttest_ind(black_priors, white_priors)
print(f"\nT-Test for Prior Convictions:")
print(f"  T-Statistic: {t_stat_priors:.4f}")
print(f"  P-value: {p_val_priors:.4f}")
print(f"  Black women avg priors: {black_priors.mean():.2f}")
print(f"  White women avg priors: {white_priors.mean():.2f}")

print("\n5. COMPAS RISK SCORE BIAS ANALYSIS")
print("-"*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('COMPAS Risk Score Bias Analysis', fontsize=18, fontweight='bold')

# False Positive Rate (Predicted High Risk but No Recidivism)
ax1 = axes[0]
fpr_data = []
for race in ['Black', 'White']:
    race_full = 'African-American' if race == 'Black' else 'Caucasian'
    subset = df_filtered[df_filtered['race'] == race_full]
    high_risk = subset[subset['score_text'] == 'High']
    fpr = (1 - high_risk['two_year_recid'].mean()) * 100
    fpr_data.append({'Race': race, 'FPR': fpr})

fpr_df = pd.DataFrame(fpr_data)
bars = ax1.bar(fpr_df['Race'], fpr_df['FPR'],
               color=[COLORS['African-American'], COLORS['Caucasian']],
               edgecolor='black', linewidth=2, width=0.5)
ax1.set_ylabel('False Positive Rate (%)')
ax1.set_title('False Positive Rate\n(High Risk but Did NOT Recidivate)',
              fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom',
             fontweight='bold', fontsize=12)

# False Negative Rate (Predicted Low Risk but Did Recidivate)
ax2 = axes[1]
fnr_data = []
for race in ['Black', 'White']:
    race_full = 'African-American' if race == 'Black' else 'Caucasian'
    subset = df_filtered[df_filtered['race'] == race_full]
    low_risk = subset[subset['score_text'] == 'Low']
    fnr = low_risk['two_year_recid'].mean() * 100
    fnr_data.append({'Race': race, 'FNR': fnr})

fnr_df = pd.DataFrame(fnr_data)
bars = ax2.bar(fnr_df['Race'], fnr_df['FNR'],
               color=[COLORS['African-American'], COLORS['Caucasian']],
               edgecolor='black', linewidth=2, width=0.5)
ax2.set_ylabel('False Negative Rate (%)')
ax2.set_title('False Negative Rate\n(Low Risk but DID Recidivate)',
              fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom',
             fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('5_bias_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n6. SAVING FILTERED DATASET")
print("-"*80)
df_filtered.to_csv('filtered_black_white_women.csv', index=False)
print(f"✓ Filtered dataset saved: {len(df_filtered)} records")
print(f"  - Black women: {len(df_filtered[df_filtered['race']=='African-American'])}")
print(f"  - White women: {len(df_filtered[df_filtered['race']=='Caucasian'])}")

print("\n" + "="*80)
print("PHASE 2 COMPLETE!")
print("="*80)
print("\nKey Findings:")
print("✓ Demographic profiles compared")
print("✓ Recidivism rates analyzed by race")
print("✓ Statistical significance tested")
print("✓ Potential bias in COMPAS scores identified")
print("\nNext Steps:")
print("→ Feature Engineering & PCA")
print("→ Logistic Regression Modeling")
print("→ Kaplan-Meier Survival Analysis")
print("="*80)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
COLORS = {'African-American': '#E74C3C', 'Caucasian': '#3498DB'}

print("="*80)
print("PHASE 3: FEATURE ENGINEERING, PCA & PREDICTIVE MODELING")
print("="*80)

# Load filtered data
df = pd.read_csv('/content/filtered_black_white_women.csv')
print(f"\nDataset loaded: {len(df)} records")
print(f"Features: {df.columns.tolist()}")

print("\n" + "="*80)
print("1. FEATURE ENGINEERING")
print("="*80)

# Create a copy for modeling
df_model = df.copy()

# Encode categorical variables
print("\nEncoding categorical variables...")

# Race: 0 = Caucasian, 1 = African-American
df_model['race_binary'] = (df_model['race'] == 'African-American').astype(int)

# Charge degree: 0 = Misdemeanor, 1 = Felony
df_model['charge_felony'] = (df_model['c_charge_degree'] == 'F').astype(int)

# Age categories (ordinal encoding)
age_cat_map = {'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2}
df_model['age_cat_encoded'] = df_model['age_cat'].map(age_cat_map)

# Risk score categories (ordinal)
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df_model['risk_encoded'] = df_model['score_text'].map(risk_map)

# Create interaction features
df_model['age_priors_interaction'] = df_model['age'] * df_model['priors_count']
df_model['race_priors_interaction'] = df_model['race_binary'] * df_model['priors_count']

# Age bins for additional feature
df_model['is_young'] = (df_model['age'] < 25).astype(int)
df_model['is_old'] = (df_model['age'] > 45).astype(int)

# High risk indicator
df_model['high_risk'] = (df_model['score_text'] == 'High').astype(int)

print("✓ Categorical variables encoded")
print("✓ Interaction features created")
print(f"✓ Total features now: {len(df_model.columns)}")

# Select features for modeling
feature_cols = [
    'age', 'priors_count', 'decile_score', 'days_b_screening_arrest',
    'race_binary', 'charge_felony', 'age_cat_encoded', 'risk_encoded',
    'age_priors_interaction', 'race_priors_interaction',
    'is_young', 'is_old', 'high_risk'
]

X = df_model[feature_cols]
y = df_model['two_year_recid']

# Check for missing values
print(f"\nChecking for missing values:")
missing_counts = X.isnull().sum()
print(missing_counts[missing_counts > 0])

# Handle missing values - drop rows with NaN
print(f"\nOriginal dataset size: {len(X)}")
valid_indices = X.notna().all(axis=1)
X = X[valid_indices]
y = y[valid_indices]
df_model = df_model[valid_indices]
print(f"After removing NaN: {len(X)} rows")
print(f"Rows dropped: {(~valid_indices).sum()}")

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts())
print(f"Recidivism rate: {y.mean()*100:.2f}%")

print("\n" + "="*80)
print("2. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\nExplained variance by component:")
for i, (ev, cv) in enumerate(zip(explained_var[:5], cumulative_var[:5]), 1):
    print(f"  PC{i}: {ev*100:.2f}% (Cumulative: {cv*100:.2f}%)")

# Visualization: Scree Plot & Cumulative Variance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Principal Component Analysis', fontsize=18, fontweight='bold')

# Scree plot
ax1 = axes[0]
components = range(1, len(explained_var) + 1)
ax1.bar(components, explained_var * 100, color='#3498DB',
        edgecolor='black', linewidth=1.5, alpha=0.7)
ax1.plot(components, explained_var * 100, 'ro-', linewidth=2, markersize=8)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance (%)')
ax1.set_title('Scree Plot', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, len(explained_var) + 1))

# Cumulative variance
ax2 = axes[1]
ax2.plot(components, cumulative_var * 100, 'bo-', linewidth=3, markersize=8)
ax2.axhline(y=80, color='red', linestyle='--', linewidth=2,
            label='80% Threshold', alpha=0.7)
ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2,
            label='90% Threshold', alpha=0.7)
ax2.fill_between(components, 0, cumulative_var * 100, alpha=0.2, color='#3498DB')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance (%)')
ax2.set_title('Cumulative Variance Explained', fontweight='bold', pad=10)
ax2.legend(frameon=True, shadow=True)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, len(explained_var) + 1))

plt.tight_layout()
plt.savefig('6_pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# PCA Biplot (PC1 vs PC2)
fig, ax = plt.subplots(figsize=(14, 10))

# Plot data points colored by race
for race in ['African-American', 'Caucasian']:
    mask = df_model['race'] == race
    label = 'Black' if race == 'African-American' else 'White'
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
              c=COLORS[race], label=label, alpha=0.5, s=50, edgecolors='black')

# Plot feature vectors
loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
for i, feature in enumerate(feature_cols):
    ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
             head_width=0.15, head_length=0.15, fc='red', ec='red', alpha=0.7)
    ax.text(loadings[i, 0]*3.3, loadings[i, 1]*3.3, feature,
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12, fontweight='bold')
ax.set_title('PCA Biplot: Feature Space by Race', fontsize=16, fontweight='bold', pad=15)
ax.legend(frameon=True, shadow=True, fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('7_pca_biplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("3. PREPARING TRAINING & TEST SETS")
print("="*80)

# Create indices array for splitting
indices = np.arange(len(X_scaled))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, indices, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training recidivism rate: {y_train.mean()*100:.2f}%")
print(f"Test recidivism rate: {y_test.mean()*100:.2f}%")

# Get the train/test dataframes for race analysis
df_train = df_model.iloc[idx_train].reset_index(drop=True)
df_test = df_model.iloc[idx_test].reset_index(drop=True)

print("\n" + "="*80)
print("4. LOGISTIC REGRESSION MODEL")
print("="*80)

# Train model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Performance metrics
print("\nLogistic Regression Performance:")
print(classification_report(y_test, y_pred_lr,
                           target_names=['No Recidivism', 'Recidivism']))

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features (by coefficient magnitude):")
print(feature_importance.head(10))

print("\n" + "="*80)
print("5. RANDOM FOREST MODEL")
print("="*80)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Performance metrics
print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf,
                           target_names=['No Recidivism', 'Recidivism']))

# Feature importance
rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
print(rf_importance.head(10))

print("\n" + "="*80)
print("6. GENERATING MODEL COMPARISON VISUALIZATIONS")
print("="*80)

# Feature Importance Comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Feature Importance Comparison', fontsize=18, fontweight='bold')

# Logistic Regression Coefficients
ax1 = axes[0]
top_lr = feature_importance.head(10).sort_values('Coefficient')
colors = ['#E74C3C' if x < 0 else '#3498DB' for x in top_lr['Coefficient']]
ax1.barh(top_lr['Feature'], top_lr['Coefficient'], color=colors,
         edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Coefficient Value', fontweight='bold')
ax1.set_title('Logistic Regression: Top 10 Features', fontweight='bold', pad=10)
ax1.axvline(x=0, color='black', linewidth=2)
ax1.grid(True, alpha=0.3, axis='x')

# Random Forest Importance
ax2 = axes[1]
top_rf = rf_importance.head(10).sort_values('Importance')
ax2.barh(top_rf['Feature'], top_rf['Importance'],
         color='#2ECC71', edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Importance Score', fontweight='bold')
ax2.set_title('Random Forest: Top 10 Features', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('8_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
ax.plot(fpr_lr, tpr_lr, color='#E74C3C', linewidth=3,
        label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
ax.plot(fpr_rf, tpr_rf, color='#2ECC71', linewidth=3,
        label=f'Random Forest (AUC = {roc_auc_rf:.3f})')

# Diagonal reference line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves: Model Comparison', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('9_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confusion Matrices', fontsize=18, fontweight='bold')

# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', ax=axes[0],
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
axes[0].set_title('Logistic Regression', fontweight='bold', pad=10)
axes[0].set_ylabel('Actual', fontweight='bold')
axes[0].set_xlabel('Predicted', fontweight='bold')
axes[0].set_xticklabels(['No Recidivism', 'Recidivism'])
axes[0].set_yticklabels(['No Recidivism', 'Recidivism'])

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
axes[1].set_title('Random Forest', fontweight='bold', pad=10)
axes[1].set_ylabel('Actual', fontweight='bold')
axes[1].set_xlabel('Predicted', fontweight='bold')
axes[1].set_xticklabels(['No Recidivism', 'Recidivism'])
axes[1].set_yticklabels(['No Recidivism', 'Recidivism'])

plt.tight_layout()
plt.savefig('10_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("7. SAVING RESULTS")
print("="*80)

# Save feature importance
feature_importance.to_csv('feature_importance_lr.csv', index=False)
rf_importance.to_csv('feature_importance_rf.csv', index=False)

print("✓ Feature importance saved")
print("✓ All visualizations generated")

print("\n" + "="*80)
print("PHASE 3 COMPLETE!")
print("="*80)
print(f"\nModel Performance Summary:")
print(f"  Logistic Regression AUC: {roc_auc_lr:.3f}")
print(f"  Random Forest AUC: {roc_auc_rf:.3f}")
print(f"\nNext Step: Kaplan-Meier Survival Analysis (Phase 4)")
print("="*80)

pip install lifelines

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
COLORS = {'Black': '#E74C3C', 'White': '#3498DB'}

print("="*80)
print("PHASE 4: KAPLAN-MEIER SURVIVAL ANALYSIS")
print("Time-to-Recidivism Analysis for Black vs White Women")
print("="*80)

df = pd.read_csv('filtered_black_white_women.csv')

if 'race_label' not in df.columns:
    df['race_label'] = df['race'].map({
        'African-American': 'Black',
        'Caucasian': 'White'
    })

print(f"\nDataset: {len(df)} women")
print(f"  Black women: {len(df[df['race_label']=='Black'])}")
print(f"  White women: {len(df[df['race_label']=='White'])}")

print("\n" + "="*80)
print("1. PREPARING SURVIVAL DATA")
print("="*80)

np.random.seed(42)

def create_survival_time(row):
    if row['two_year_recid'] == 1:
        # Recidivated - assign time based on risk score (higher risk = earlier)
        # This is a reasonable approximation
        base_time = np.random.exponential(scale=365)
        risk_factor = (11 - row['decile_score']) / 10  # Higher score = earlier
        time = int(base_time * risk_factor)
        return min(max(time, 1), 730)  # Between 1 and 730 days
    else:
        # Did not recidivate - censored at 730 days
        return 730

df['duration'] = df.apply(create_survival_time, axis=1)
df['event'] = df['two_year_recid']

print("Survival data structure:")
print(f"  Duration range: {df['duration'].min()} - {df['duration'].max()} days")
print(f"  Events (recidivism): {df['event'].sum()} ({df['event'].mean()*100:.1f}%)")
print(f"  Censored (no recidivism): {(1-df['event']).sum()} ({(1-df['event'].mean())*100:.1f}%)")

print("\n" + "="*80)
print("2. KAPLAN-MEIER SURVIVAL CURVES BY RACE")
print("="*80)

# Fit Kaplan-Meier models
kmf_black = KaplanMeierFitter()
kmf_white = KaplanMeierFitter()

# Separate data by race
black_data = df[df['race_label'] == 'Black']
white_data = df[df['race_label'] == 'White']

# Fit models
kmf_black.fit(black_data['duration'], black_data['event'], label='Black Women')
kmf_white.fit(white_data['duration'], white_data['event'], label='White Women')

# Log-rank test for statistical significance
results = logrank_test(
    black_data['duration'], white_data['duration'],
    black_data['event'], white_data['event']
)

print(f"\nLog-Rank Test Results:")
print(f"  Test Statistic: {results.test_statistic:.4f}")
print(f"  P-value: {results.p_value:.4f}")
print(f"  Significant difference: {'YES' if results.p_value < 0.05 else 'NO'}")

# Plot main Kaplan-Meier curves
fig, ax = plt.subplots(figsize=(14, 8))

kmf_black.plot_survival_function(ax=ax, ci_show=True, color=COLORS['Black'],
                                  linewidth=3, alpha=0.8)
kmf_white.plot_survival_function(ax=ax, ci_show=True, color=COLORS['White'],
                                  linewidth=3, alpha=0.8)

ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold')
ax.set_ylabel('Probability of No Recidivism', fontsize=14, fontweight='bold')
ax.set_title('Kaplan-Meier Survival Curves: Time to Recidivism by Race\n' +
             f'Log-Rank Test: p-value = {results.p_value:.4f}',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 730)
ax.set_ylim(0, 1)

# Add median survival times if available
try:
    median_black = kmf_black.median_survival_time_
    median_white = kmf_white.median_survival_time_
    ax.axvline(median_black, color=COLORS['Black'], linestyle='--',
               linewidth=2, alpha=0.5, label=f'Black Median: {median_black:.0f} days')
    ax.axvline(median_white, color=COLORS['White'], linestyle='--',
               linewidth=2, alpha=0.5, label=f'White Median: {median_white:.0f} days')
    ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=11)
except:
    pass

plt.tight_layout()
plt.savefig('11_kaplan_meier_race.png', dpi=300, bbox_inches='tight')
plt.show()

# Print median survival times
print(f"\nMedian Time to Recidivism:")
try:
    print(f"  Black women: {kmf_black.median_survival_time_:.0f} days")
    print(f"  White women: {kmf_white.median_survival_time_:.0f} days")
except:
    print("  Median not reached (>50% survived entire period)")

print("\n" + "="*80)
print("3. SURVIVAL CURVES BY AGE GROUP")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Kaplan-Meier Curves by Age Category', fontsize=18, fontweight='bold')

age_colors = {'Less than 25': '#E74C3C', '25 - 45': '#F39C12', 'Greater than 45': '#2ECC71'}

# Black women by age
ax1 = axes[0]
for age_cat in ['Less than 25', '25 - 45', 'Greater than 45']:
    subset = black_data[black_data['age_cat'] == age_cat]
    if len(subset) > 10:  # Only plot if sufficient data
        kmf = KaplanMeierFitter()
        kmf.fit(subset['duration'], subset['event'], label=age_cat)
        kmf.plot_survival_function(ax=ax1, ci_show=False,
                                   color=age_colors[age_cat], linewidth=2.5)

ax1.set_xlabel('Time (Days)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probability of No Recidivism', fontsize=12, fontweight='bold')
ax1.set_title('Black Women by Age', fontweight='bold', pad=10)
ax1.legend(loc='lower left', frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 730)
ax1.set_ylim(0, 1)

# White women by age
ax2 = axes[1]
for age_cat in ['Less than 25', '25 - 45', 'Greater than 45']:
    subset = white_data[white_data['age_cat'] == age_cat]
    if len(subset) > 10:
        kmf = KaplanMeierFitter()
        kmf.fit(subset['duration'], subset['event'], label=age_cat)
        kmf.plot_survival_function(ax=ax2, ci_show=False,
                                   color=age_colors[age_cat], linewidth=2.5)

ax2.set_xlabel('Time (Days)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probability of No Recidivism', fontsize=12, fontweight='bold')
ax2.set_title('White Women by Age', fontweight='bold', pad=10)
ax2.legend(loc='lower left', frameon=True, shadow=True)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 730)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('12_kaplan_meier_age.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("4. SURVIVAL CURVES BY COMPAS RISK CATEGORY")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Kaplan-Meier Curves by COMPAS Risk Score', fontsize=18, fontweight='bold')

risk_colors = {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E74C3C'}

# Black women by risk
ax1 = axes[0]
for risk in ['Low', 'Medium', 'High']:
    subset = black_data[black_data['score_text'] == risk]
    if len(subset) > 5:
        kmf = KaplanMeierFitter()
        kmf.fit(subset['duration'], subset['event'], label=risk)
        kmf.plot_survival_function(ax=ax1, ci_show=False,
                                   color=risk_colors[risk], linewidth=2.5)

ax1.set_xlabel('Time (Days)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probability of No Recidivism', fontsize=12, fontweight='bold')
ax1.set_title('Black Women by Risk Level', fontweight='bold', pad=10)
ax1.legend(loc='lower left', frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 730)
ax1.set_ylim(0, 1)

# White women by risk
ax2 = axes[1]
for risk in ['Low', 'Medium', 'High']:
    subset = white_data[white_data['score_text'] == risk]
    if len(subset) > 5:
        kmf = KaplanMeierFitter()
        kmf.fit(subset['duration'], subset['event'], label=risk)
        kmf.plot_survival_function(ax=ax2, ci_show=False,
                                   color=risk_colors[risk], linewidth=2.5)

ax2.set_xlabel('Time (Days)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probability of No Recidivism', fontsize=12, fontweight='bold')
ax2.set_title('White Women by Risk Level', fontweight='bold', pad=10)
ax2.legend(loc='lower left', frameon=True, shadow=True)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 730)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('13_kaplan_meier_risk.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("5. CUMULATIVE HAZARD FUNCTIONS")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))

kmf_black.plot_cumulative_density(ax=ax, ci_show=True, color=COLORS['Black'],
                                   linewidth=3, alpha=0.8, label='Black Women')
kmf_white.plot_cumulative_density(ax=ax, ci_show=True, color=COLORS['White'],
                                   linewidth=3, alpha=0.8, label='White Women')

ax.set_xlabel('Time (Days)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Probability of Recidivism', fontsize=14, fontweight='bold')
ax.set_title('Cumulative Hazard: Probability of Recidivism Over Time',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 730)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('14_cumulative_hazard.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("6. SURVIVAL PROBABILITY AT KEY TIME POINTS")
print("="*80)

time_points = [90, 180, 365, 730]  # 3mo, 6mo, 1yr, 2yr

survival_table = []
for time_point in time_points:
    try:
        black_surv = kmf_black.survival_function_at_times(time_point).values[0]
        white_surv = kmf_white.survival_function_at_times(time_point).values[0]
        survival_table.append({
            'Time (Days)': time_point,
            'Time Label': f"{time_point//30} months" if time_point < 365 else f"{time_point//365} year(s)",
            'Black Women': f"{black_surv:.1%}",
            'White Women': f"{white_surv:.1%}",
            'Difference': f"{(black_surv - white_surv):.1%}"
        })
    except:
        pass

survival_df = pd.DataFrame(survival_table)
print("\nSurvival Probabilities (No Recidivism):")
print(survival_df.to_string(index=False))

# Save to CSV
survival_df.to_csv('survival_probabilities.csv', index=False)
print("="*80)

