#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd


# In[53]:


data=pd.read_csv('2024.csv')


# In[54]:


data.columns


# In[55]:


data.describe()


# In[57]:


import matplotlib.pyplot as plt # Import missing module
import seaborn as sns # Also import Seaborn
import numpy as np # Let's add NumPy (can be used for mask)

# Define columns for Correlation Analysis
columns = ['Ladder score',
'Explained by: Log GDP per capita',
'Explained by: Social support',
'Explained by: Healthy life expectancy',
'Explained by: Freedom to make life choices',
'Explained by: Generosity',
'Explained by: Perceptions of corruption']

# Create the correlation matrix
correlation_matrix = data[columns].corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="plasma", fmt=".3f", linewidths=0.5)
plt.title("Correlation Matrix Heat Map")
plt.show()


# In[43]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("2024.csv")

# Set column "Country name" as index
data.set_index("Country name", inplace=True)

# Select variables to use for clustering
X = data.iloc[:, 1:]  # Get all columns except the first column (Country name)

# Fill missing values with column averages
X.fillna(X.mean(), inplace=True)

# Create K-means clustering model
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model to the data
kmeans.fit(X)


# In[44]:


# Get cluster labels
clusters = kmeans.labels_

# Add the extracted cluster labels to the DataFrame based on the number of clusters
data['Cluster'] = clusters

# Show cluster centers
print("Küme Merkezleri:")
print(kmeans.cluster_centers_)

# Show groups by cluster number
print("\nKüme Sayısı ve Üye Sayıları:")
print(data['Cluster'].value_counts())


# In[45]:


# Determining the most appropriate number of clusters with the Elbow method
sse = []
for k in range(1, 5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Graph for determining the most appropriate number of clusters according to the point where the elbow is formed
plt.figure(figsize=(10, 6))
plt.plot(range(1, 5), sse, marker='o')
plt.title('Determining the Number of Clusters with the Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('İnertia')

# Estimate and highlight the most suitable number of clusters
optimal_k = range(1, 5)[sse.index(min(sse, key=lambda x: abs(x - min(sse))))]
plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal Cluster: {optimal_k}')
plt.legend()
plt.show()


# In[46]:


# Create K-means clustering model
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Get cluster labels
clusters = kmeans.labels_

# Add the extracted cluster labels to the DataFrame based on the number of clusters
data['Cluster'] = clusters

# Show cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Show groups by cluster number
print("\nNumber of Clusters and Number of Members:")
print(data['Cluster'].value_counts())

# Visualize clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Ladder score', y='Explained by: Log GDP per capita', hue='Cluster', palette='Set1', s=100)
plt.title('K-means Kümeleme Sonuçları')
plt.xlabel('Ladder score')
plt.ylabel('Explained by: Log GDP per capita')
plt.legend(title='Cluster')
plt.show()


# In[47]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('2024.csv')

# Set 'Country name' as the index
data.set_index("Country name", inplace=True)

# Remove the 'upperwhisker' and 'lowerwhisker' columns from the dataset
X = data.drop(['upperwhisker', 'lowerwhisker'], axis=1)

# Fill missing values with column means
X.fillna(X.mean(), inplace=True)

# Create and fit the KMeans clustering model
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Calculate the mean values for each feature in each cluster, excluding the removed columns
cluster_means = data.drop(['upperwhisker', 'lowerwhisker'], axis=1).groupby('Cluster').mean()

# Plot the average values of features in each cluster with a bar chart and label each bar with its value
fig, ax = plt.subplots(figsize=(15, 10))
bars = cluster_means.T.plot(kind='bar', ax=ax)

# Add labels on each bar
for container in bars.containers:
    bars.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10)

# Set the title and labels
plt.title('Average Values of Features for Each Cluster')
plt.xlabel('Features')
plt.ylabel('Average Value')
plt.legend(title='Cluster')

# Fix the rotation issue
plt.xticks(rotation=45)  # Rotate the X-axis labels by 45 degrees

plt.show()


# In[48]:


cluster_means


# In[49]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('2024.csv')

# Separate properties and target variables
X = df.drop(columns=['Country name', 'Ladder score'])
y = df['Ladder score']

# Fill missing values with average
X_imputed = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Determine the appropriate number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Dirsek Yöntemi')
plt.xlabel('Küme Sayısı')
plt.ylabel('WCSS')
plt.show()

# K-means clustering (e.g. with 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Reduction to two dimensions with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Data Visualization
plt.figure(figsize=(20, 15))
scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100)
plt.title('Clustering of Countries with K-means')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# Add clusters to original dataframe
df['Cluster'] = clusters

# Label each data point
for i in range(len(X_pca)):
    plt.text(X_pca[i, 0], X_pca[i, 1], df['Country name'].iloc[i], fontsize=12)

plt.legend(title='Küme')
plt.show()

# Print the countries in each set
for cluster in range(kmeans.n_clusters):
    print(f"Küme {cluster}:\n", df[df['Cluster'] == cluster]['Country name'].values)


# In[50]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv('2024.csv')

# Separate properties and target variables
X = df.drop(columns=['Country name', 'Ladder score'])
y = df['Ladder score']

# Fill missing values ​​with average
X_imputed = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# K-means clustering to use clusters as target variables
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
y_clusters = kmeans.fit_predict(X_scaled)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clusters, test_size=0.2, random_state=42)

# Define models
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
decision_tree_model = DecisionTreeClassifier(random_state=42)
svm_model = SVC(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)
neural_network_model = MLPClassifier(max_iter=1000, random_state=42)
xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Train models
logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
neural_network_model.fit(X_train, y_train)
xgboost_model.fit(X_train, y_train)

# Make predictions
y_pred_logistic = logistic_model.predict(X_test)
y_pred_decision_tree = decision_tree_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_random_forest = random_forest_model.predict(X_test)
y_pred_neural_network = neural_network_model.predict(X_test)
y_pred_xgboost = xgboost_model.predict(X_test)

# Draw Confusion Matrix for each model
models = {
    "Logistic Regression": y_pred_logistic,
    "Decision Tree": y_pred_decision_tree,
    "SVM": y_pred_svm,
    "Random Forest": y_pred_random_forest,
    "Neural Network": y_pred_neural_network,
    "XGBoost": y_pred_xgboost
}

# Plot Confusion Matrices in a single column
plt.figure(figsize=(8, 30))
for i, (model_name, y_pred) in enumerate(models.items(), 1):
    plt.subplot(len(models), 2, i)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Print the accuracy of each model on the screen
for model_name, y_pred in models.items():
    print(f"{model_name} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))


# In[51]:


import matplotlib.pyplot as plt

# Model names and accuracy values
models = ["Logistic Regression", "Decision Trees", "SVM", "Random Forests", "ANN", "XGBoost"]
accuracies = [0.86, 0.86, 0.86, 0.83, 0.86, 0.79]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=["b", "g", "r", "c", "y", "k"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("K-Means Model Performance")
plt.ylim(0, 1)  # Y-axis limits
plt.grid(axis="y", linestyle="--", alpha=0.4)

# # Label numeric values
for i, acc in enumerate(accuracies):
    plt.text(i, acc, str(acc), ha='center', va='bottom')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




