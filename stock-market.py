import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


data = {
    "Company": ["TCS","Infosys","HDFC_Bank","ICICI_Bank","Wipro","Tata_Motors","Adani_Enterprises","ITC","Sun_Pharma","Zamato","Bharti_Airtel","Swiggy"],
    "AnnualReturn": [15,12,14,9,18,7,25,32,8,16,35,20],
    "Volatility": [18,14,16,10,22,20,30,42,12,19,45,21],
    "TradingVolume": [800,600,550,700,750,400,900,1100,500,450,1200,850],
    "PERatio": [25,30,28,18,20,22,15,40,24,32,55,26]
}
df = pd.DataFrame(data)

X = df[[
    "AnnualReturn",
    "Volatility",
    "TradingVolume",
    "PERatio"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertia = []

for k in range(1,11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit_predict(X_scaled)
    inertia.append(model.inertia_)

plt.plot(range(1,11),inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(alpha=0.3)
plt.savefig("elbow.png")
plt.show()

# Silhouette Score
silhouette_scores = []

for k in range(2,11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print("==========================================")
    print("K: ",k, "Silhouette Score: ", round(score,3))

plt.plot(range(2,11),silhouette_scores, marker="o",c="green")
plt.xlabel("Numer of Clusters(K)")
plt.ylabel("Silhouette Core")
plt.title("Silhouette Graph")
plt.grid(alpha=0.3)
plt.show()

# KMeans Model 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
print("=========================================")
print("KMeans Cluster: ", df)

# Cluster Centers 
centroids = kmeans.cluster_centers_
original_centroids = scaler.inverse_transform(centroids)
print("==========================================")
print("Cluster Centroids: ", original_centroids)

# KMeans Graph 
plt.scatter(
    df["AnnualReturn"],
    df["Volatility"],
    c=df["Cluster"],
    s=100,
    alpha=0.8
)
plt.scatter(
    original_centroids[:,0],
    original_centroids[:,1],
    s=250,
    marker="*",
    color="red",
    edgecolor="black"
)
plt.xlabel("Annual Return")
plt.ylabel("Volatility")
plt.title("KMeans with Centroids")
plt.grid()
plt.show()

# KMEANS Cluster with color centroids
for cluster in range(3):
    subset = df[df["Cluster"] == cluster]
    plt.scatter(
        subset["AnnualReturn"],
        subset["Volatility"],
        s=100,
        alpha=0.8,
        label=f"Cluster {cluster}"
    )
plt.scatter(
    original_centroids[:,0],
    original_centroids[:,1],
    s=200,
    marker="*",
    color="black",
    edgecolor="red",
    label="Centroids"
)
plt.xlabel("Annual Return")
plt.ylabel("Volatility")
plt.title("KMeans Graph with Color Cluster")
plt.legend()
plt.grid(True)
plt.show()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_centroids = pca.fit_transform(kmeans.cluster_centers_)

#PCA Scatter Plot
for cluster in range(3):
    plt.scatter(
        X_pca[df["Cluster"].values == cluster, 0],
        X_pca[df["Cluster"].values == cluster, 1],
        s=100,
        alpha=0.8,
        label=f"Cluster {cluster}"
    )

plt.scatter(
    pca_centroids[:,0],
    pca_centroids[:,1],
    marker="D",
    s=100,
    color="black",
    edgecolor="white",
    linewidth=1.5,
    label="Cntroids"
)
plt.xlabel("PC1"),
plt.ylabel("PC2"),
plt.title("Stock Market Clusters Using PCA")
plt.legend()
plt.grid(True)
plt.savefig("pca_clusters.png")
plt.show()

print("===================================")
print(sum(pca.explained_variance_ratio_))

print("***********************************")
print(df.groupby("Cluster").mean(numeric_only=True))
