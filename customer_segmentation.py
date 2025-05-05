# customer_segmentation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Progetto: Customer Segmentation (RFM + KMeans + PCA)
# Dataset mock e-commerce
# -----------------------------

# Step 1: Generazione dati simulati
np.random.seed(42)
n_customers = 500
customer_ids = np.arange(10000, 10000 + n_customers)
recency = np.random.randint(1, 365, size=n_customers)
frequency = np.random.poisson(lam=10, size=n_customers)
monetary = np.random.gamma(shape=2, scale=100, size=n_customers)

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary
})

# Step 2: Normalizzazione
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

# Step 3: Clustering KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 4: PCA per visualizzazione
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)
df['PC1'] = pca_components[:, 0]
df['PC2'] = pca_components[:, 1]

# Step 5: Visualizzazione
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2',
                hue='Cluster', palette='tab10', s=60)
plt.title("📊 Customer Segmentation con PCA + KMeans")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.savefig("customer_segmentation_pca.png")

# Salvataggio
df.to_csv("customer_segmentation_rfm.csv", index=False)
