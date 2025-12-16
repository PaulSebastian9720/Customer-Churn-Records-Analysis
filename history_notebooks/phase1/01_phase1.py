# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## <span id="sec1"></span> **1. Loading the Dataset**

# %%
# Necessary imports librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils_stats import cramers_v
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


# %%
df = pd.read_csv("./_attachments/datasets/Customer-Churn-Records.csv", sep=',')
tarjet = "Exited"

print("Dataset loaded successfully")
df.head()

# %% [markdown]
# ## <span id="sec1"></span> **2 ExploratoryAnalysis**

# %% [markdown]
# ### <span id="sec21"></span> **2.1 Dataset structure**

# %%
print("Dataset structure:", df.shape)
print("Columns: ", df.columns)
print("Obetive variable: ", tarjet)
df[tarjet].value_counts()

# %% [markdown]
# ### <span id="sec22"></span> **2.2 Frequency analysis**

# %%
print("Statistical description of the variables:")
print(df.describe())

# %% [markdown]
# ### <span id="sec23"></span> **2.3 Descriptive statistics**

# %%
df.describe(include='all')

# %%
for col in df.columns:
    print(f"\n========== {col} ==========")
    print(f"Number of unique values: {df[col].nunique()}")
    print(f"Unique values: {df[col].unique()}")

# %% [markdown]
# ### <span id="sec24"></span> **2.4 Correlations**

# %%
num_cols = df.select_dtypes(include=['int64', 'float64'])
cr = num_cols.corr()
mask = np.triu(np.ones_like(cr, dtype=bool))

plt.figure(figsize=(18, 14))
ax = sns.heatmap(
    cr,
    mask=mask,
    cmap="YlGnBu",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    vmin=-1, vmax=1
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "./_attachments/img/correlacion_numericas_piramide.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% [markdown]
# ### <span id="se25"></span> **2.5 Data visualization**

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# CreditScore
sns.histplot( df['CreditScore'], kde=True, bins=20, color='blue', ax=axes[0]
             )
axes[0].set_title("Distribution CreditScore")
axes[0].set_xlabel("CreditScore")
axes[0].set_ylabel("Frecuencia")

# Balance
sns.histplot(df['Balance'], kde=True, bins=20, color='green', ax=axes[1])
axes[1].set_title("Distribution Balance")
axes[1].set_xlabel("Balance")
axes[1].set_ylabel("Frecuencia")

# EstimatedSalary
sns.histplot( df['EstimatedSalary'], kde=True, bins=20, color='purple', ax=axes[2])
axes[2].set_title("Distribution EstimatedSalary")
axes[2].set_xlabel("EstimatedSalary")
axes[2].set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# %%
sns.scatterplot(x='Balance', y='EstimatedSalary', hue='Exited', data=df, palette='coolwarm')
plt.xlabel("Balance")
plt.ylabel("Estimated Salary")
plt.legend(title="Exited")
plt.show()

# %% [markdown]
# ### <span id="sec26"></span> **2.6 Analysis of categorical variables**

# %%
categorical_cols = [
    'Geography',
    'Gender',
    'HasCrCard',
    'IsActiveMember',
    'Complain',
    'Card Type',
    'Satisfaction Score'
]

# %%
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(
        data=df,
        x=col,
        ax=axes[i]
    )
    axes[i].set_title(f'Distribution {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuency')
    axes[i].tick_params(axis='x', rotation=45)

# Eliminar subplots vacíos
for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %%
cat_df = df[categorical_cols]
assoc_cat = pd.DataFrame(
    np.zeros((len(categorical_cols), len(categorical_cols))),
    index=categorical_cols,
    columns=categorical_cols
)

for col1 in categorical_cols:
    for col2 in categorical_cols:
        assoc_cat.loc[col1, col2] = cramers_v(cat_df[col1], cat_df[col2])

mask = np.triu(np.ones_like(assoc_cat, dtype=bool))

sns.set(style="white")

plt.figure(figsize=(18, 14))
ax = sns.heatmap(
    assoc_cat,
    mask=mask,
    cmap="YlGnBu",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    vmin=0, vmax=1
)


plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "./_attachments/img/asociacion_categoricas_piramide.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% [markdown]
# ## <span id="sec3"></span> **3. Preparation of the Data Set and Variables**

# %% [markdown]
# ### <span id="sec31"></span> **3.1 Elimination of Non-Relevant Variables**
# In order to avoid bias and overfitting, a new data set called df_transformed was created, eliminating the variables RowNumber, CustomerId and Surname, which correspond to identifiers without predictive value. This transformation allows the original dataset to be preserved and facilitates the reproducibility of the analysis.

# %%
cols_to_drop = [
    'RowNumber',
    'CustomerId',
    'Surname'
]
df_transformed = df.drop(columns=cols_to_drop)

print("Original:", df.shape)
print("Transformed:", df_transformed.shape)
print(df_transformed.columns)


# %% [markdown]
# ### <span id="sec32"></span> **3.2 Handling Missing Data**
# The dataset does not present missing values, therefore no technique is applied to correct

# %%
df_transformed.isnull().any()


# %% [markdown]
# ## <span id="sec4"></span> **4. Variable transformations**

# %% [markdown]
# ### <span id="sec41"></span> **4.1 Transformations Table**

# %% [markdown]
# ### <span id="sec42"></span> **4.2 Pipeline Design**
# Global standardization using StandardScaler was applied subsequent to all coding transformations, so that all variables including those derived from one-hot coding were centered at zero mean and unit standard deviation. This approach guarantees homogeneous scaling, especially suitable for distance-based models and neural networks.

# %%
numeric_features = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'EstimatedSalary', 'Point Earned'
]
ordinal_features = ['Satisfaction Score']
nominal_features = ['Geography', 'Gender', 'Card Type']
binary_features = ['HasCrCard', 'IsActiveMember', 'Complain']

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('ord', OrdinalEncoder(categories=[[1, 2, 3, 4, 5]]), ordinal_features),
        ('bin', 'passthrough', binary_features),
        ('nom', OneHotEncoder( handle_unknown='ignore'), nominal_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('scaler', StandardScaler())
])


pipeline

# %% [markdown]
# ### <span id="sec43"></span> **4.3 Variable transformation stage**

# %%

X = df_transformed.drop(columns=[
    'Exited'
])

y = df_transformed['Exited']
X_transformed = pipeline.fit_transform(X)
feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
feature_names = [
    name.split('__')[-1] for name in feature_names
]



# %% [markdown]
# ### <span id="sec44"></span> **4.4 Downloads**

# %%
#df_without_tags = pd.DataFrame(
#    X_transformed,
#)
#df_without_tags.to_csv(
#    "_attachments/datasets/customer_churn_transformed_without_tags.csv",
#    index=False
#)
#df_without_tags.head()



# %%
df_with_tags = pd.DataFrame(
    X_transformed,
    columns=feature_names
)
df_with_tags['Exited'] = y.values

df_with_tags.head()


# %%
corr = df_with_tags.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.set(style="white")

plt.figure(figsize=(18, 14))
sns.heatmap(
    corr,
    mask=mask,
    cmap="YlGnBu",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    vmin=-1, vmax=1
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "_attachments/img/correlacion_completa_con_tags_piramide.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% [markdown]
# ### <span id="sec45"></span> **4.5 Downloads**

# %%
df_with_tags.to_csv(
    "_attachments/datasets/customer_churn_transformed_with_tags.csv",
    index=False
)
