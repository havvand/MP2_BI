import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

print(plt.get_backend(), "backend")

red_wine = pd.read_excel("data/winequality-red.xlsx", header=1)
white_wine = pd.read_excel("data/winequality-white.xlsx", header=1)
#%%
print(red_wine.shape)
print(white_wine.shape)
#%%
print(list(red_wine))
print(list(white_wine))

print("------Getting the sum of the missing values in each column---------")
print("------Red Wine---------")
print(red_wine.isnull().sum())
print("------White Wine---------")
print(white_wine.isnull().sum())

#%%
print("-------Add wine_type and concat DataFrames--------")
red_wine['red_wine'] = "1"
red_wine['white_wine'] = "0"
white_wine['white_wine'] = "1"
white_wine['red_wine'] = "0"

#%%
print("------Explore the data. Original and new columns, type of data in each---------")
print("Rows and columns in red_wine: ", red_wine.shape)
print("DUPLICATES: ", red_wine.duplicated().sum())
red_wine = red_wine.drop_duplicates()
print("DUPLICATES: ", red_wine.duplicated().sum())

print("Rows and columns in white_wine: ", white_wine.shape)
print("DUPLICATES: ", white_wine.duplicated().sum())
white_wine = white_wine.drop_duplicates()
print("DUPLICATES: ", white_wine.duplicated().sum())

print("Datatypes in red_wine: \n", red_wine.dtypes)
print("Datatypes in white_wine: \n", white_wine.dtypes)

#%%
print("------Calculate the descriptive statistics for the numeric data---------")
red_described = red_wine.describe()
white_described = white_wine.describe()

print(red_wine.mode())
outliers_red = red_wine.boxplot(column=['fixed acidity', 'volatile acidity', 'alcohol', 'residual sugar', 'pH', 'sulphates', 'quality'])

outliers_white = white_wine.boxplot(column=['fixed acidity', 'volatile acidity', 'alcohol', 'residual sugar', 'pH', 'sulphates', 'quality'])
plt.show()
print(white_wine.mode())


#%%
wine_data_copy = pd.concat([red_wine, white_wine])
#%%
wine_counts = wine_data_copy['red_wine'].value_counts()

wine_counts.plot(kind='bar', color=['red', 'black'], alpha=0.7)
plt.ylabel('Count')
plt.show()

#%% which type of wine has higher average quality? White wine has a higher average quality
plt_average_quality = sns.boxplot(x='red_wine', y='quality', hue='red_wine', data=wine_data_copy, showmeans=True)
plt.show()

#%% which type of wine has higher average level of alcohol? Red wine has a higher average level of alcohol
plt_average_alc_level = sns.boxplot(x='red_wine', y='alcohol', hue='red_wine', data=wine_data_copy, color='beige', showmeans=True)
plt.show()

#%% which one has higher average quantity of residual sugar? White wine.
plt_average_residual_sugar_level = sns.boxplot(x='red_wine', y='residual sugar', hue='red_wine', data=wine_data_copy, color='beige', showmeans=True)
plt.show()
#%% Bin the attribute pH in 5 subsets and find subset with the highest density
min_val = wine_data_copy['pH'].min()
max_val = wine_data_copy['pH'].max()

bin_width = (max_val - min_val) / 6 #The width is the range divided by the number of bins, so that the bins are equally spaced

bins = [min_val + i * bin_width for i in range(6)] # This is the list of the bin edges
print(bins)

wine_data_copy['pH_bin'] = pd.cut(wine_data_copy['pH'], bins=bins, include_lowest=True)
print(wine_data_copy['pH_bin'].dtypes)
print(wine_data_copy['pH_bin'])

wine_data_copy['pH_bin'] = wine_data_copy['pH_bin'].apply(lambda x: float(x.mid))
sns.boxplot(x='pH_bin', y='density', data=wine_data_copy, showmeans=True)
print(wine_data_copy.head())
mean_density = wine_data_copy.groupby('pH_bin')['density'].mean()
print(mean_density)
largetst_value = mean_density.max()
print('Highest mean density: ', largetst_value, 'for entry: ', mean_density.idxmax())

#%% Bin the attribute pH in 10 subsets and find subset with the highest density
min_val = wine_data_copy['pH'].min()
max_val = wine_data_copy['pH'].max()

bin_width = (max_val - min_val) / 10 #The width is the range divided by the number of bins, so that the bins are equally spaced

bins = [min_val + i * bin_width for i in range(11)] # This is the list of the bin edges
print(bins)

wine_data_copy['pH_bin_10'] = pd.cut(wine_data_copy['pH'], bins=bins, include_lowest=True)
print(wine_data_copy['pH_bin_10'].dtypes)
print(wine_data_copy['pH_bin_10'])

wine_data_copy['pH_bin_10'] = wine_data_copy['pH_bin_10'].apply(lambda x: float(x.mid))
sns.boxplot(x='pH_bin_10', y='density', data=wine_data_copy, showmeans=True)
print(wine_data_copy.head())
mean_density = wine_data_copy.groupby('pH_bin_10')['density'].mean()
print(mean_density)
largetst_value = mean_density.max()
print('Highest mean density: ', largetst_value, 'for entry: ', mean_density.idxmax())

 #%% Correlation matrix
correlation_matrix = wine_data_copy.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix, annot=True, annot_kws={"size": 12})
#%%
sns.clustermap(wine_data_copy.corr(), annot=True, cmap='coolwarm', )
#%% 10 Residual sugar outliers
sns.boxplot(x='red_wine', y='residual sugar', data=wine_data_copy, showmeans=True)
residual_sugar_column = wine_data_copy['residual sugar']

Q1 = residual_sugar_column.quantile(0.25)
Q3 = residual_sugar_column.quantile(0.75)
# IQR Method. Everything that falles below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is an outlier
IQR = Q3 - Q1

outlier_range = (residual_sugar_column < (Q1 - 1.5 * IQR)) | (residual_sugar_column > (Q3 + 1.5 * IQR))

outlier_rows = residual_sugar_column[outlier_range]
print(outlier_rows)
#%%
residual_sugar_outliers_removed = wine_data_copy[~outlier_range]
sns.boxplot(x='red_wine', y='residual sugar', data=residual_sugar_outliers_removed, showmeans=True)
print(wine_data_copy.shape)
print(residual_sugar_outliers_removed.shape)

#%% 11 Identify the attribute with the lowest correlation to the wine quality and remove it
sns.clustermap(wine_data_copy.corr(), annot=True, cmap='coolwarm', )
# Remove density??

#%% 12 Transform categorical data to numerical data
# Already done earlier, when we added the red_wine and white_wine columns as 1 and 0

# Dropping NaN values from pH_bin
print(wine_data_copy.isna().sum())
print(wine_data_copy.shape)
wine_data_copy = wine_data_copy[wine_data_copy['pH_bin'].notna()]
print(wine_data_copy.shape)

#%%
# Select only the numerical columns (i only have numerical columns
numerical_data = wine_data_copy

# Strandardize the numerical data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Create a PCA instance
pca = PCA()

# Fit and transform the data
principal_components = pca.fit_transform(scaled_data)

# Convert to DF
principal_df = pd.DataFrame(data=principal_components)

print('Explained variance: ', pca.explained_variance_ratio_)

explained_variance_percent = pca.explained_variance_ratio_ * 100
for i, variance in enumerate(explained_variance_percent):
    print(f"Principal Component {i+1}: {variance:.2f}%")

#%% Try to reduce the number of features of the aggregated data set by applying principal
# component analysis (PCA). What is the optimal number of components? 4 according to the explained variance - 70 + percent
# Get the loadings
loadings = pca.components_

# Create a DataFrame that will have the PCA loadings for all the features
loadings_df = pd.DataFrame(loadings, columns=numerical_data.columns)

# Print the loadings
print(loadings_df)

# Sum the loadings for each feature
loadings_sum = loadings_df.sum(axis=0)

# Get the features to drop
features_to_drop = loadings_sum[loadings_sum <= 0].index

# Drop the features
copsssy_copy = wine_data_copy.drop(features_to_drop, axis=1)
#%%
reduced_df = principal_df.iloc[:, :4]

#%% Print 10 random rows of the reduced data set
print(reduced_df.sample(10))