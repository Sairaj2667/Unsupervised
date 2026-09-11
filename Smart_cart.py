import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install kneed

df=pd.read_csv("smartcart_customers.csv")

df.head()

print(df.shape)
df.isnull().sum()

df["Income"]=df["Income"].fillna(df["Income"].median())

df.isnull().sum()

df.info

df.columns

df["Age"]=2026-df["Year_Birth"]

df["Dt_Customer"]=pd.to_datetime(df["Dt_Customer"],dayfirst=True)
ref_date=df["Dt_Customer"].max()
df["Custome_tenure_day"]=(ref_date-df["Dt_Customer"]).dt.days #dt.days = total days 

df.head()

 df["Total_spending"]=df["MntWines"]+df["MntFruits"]+df["MntMeatProducts"]+df["MntFishProducts"]+df["MntSweetProducts"]+df["MntGoldProds"]

df["Children"]=df["Teenhome"]+df["Kidhome"]

df["Education"].value_counts()

df["Education"]=df["Education"].replace({
    "Basic":"UnderGraduate","2n Cycle":"UnderGraduate",
    "Graduation":"Graduate",
    "Master":"PostGraduate","PhD":"PostGraduate"
})


df["Marital_Status"].value_counts()

df["Living_with"]=df["Marital_Status"].replace({
    "Married":"Partner","Together":"Partner",
    "Single":"Alone", "Alone":"Alone", "Absurd":"Alone", "YOLO":"Alone",
    "Divorced":"Alone","Widow":"Alone"
})

df["Living_with"].value_counts()


df.columns

cols=["ID","Kidhome","Marital_Status","Teenhome","Year_Birth","Dt_Customer"]
spending_cols=["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
cols_to_drop=cols+spending_cols
df_clean=df.drop(columns=cols_to_drop)

df_clean.head()

colss=["Income","Recency","Response","Age","Total_spending","Children"]
sns.pairplot(df_clean[colss])

df_clean=df_clean[(df_clean["Age"]<90)]
df_clean=df_clean[(df_clean["Income"]<600_000)]

corr=df_clean.corr(numeric_only=True)
corr

plt.figure(figsize=(20,8))
sns.heatmap(data=corr,cmap="coolwarm",annot=True)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
cat_cols=["Living_with","Education"]
enc_cols=ohe.fit_transform(df_clean[cat_cols])

enc_df=pd.DataFrame(enc_cols.toarray(),columns=ohe.get_feature_names_out(cat_cols),index=df_clean.index)

df_enc=pd.concat([df_clean.drop(columns=cat_cols),enc_df],axis=1)

df_enc.shape

from sklearn.preprocessing import StandardScaler
X=df_enc
sc=StandardScaler()
X_scaled=sc.fit_transform(X)

from sklearn.decomposition import PCA
pca=PCA(n_components=3)
X_pca=pca.fit_transform(X_scaled)

fig=plt.figure(figsize=(8,6))
ax= fig.add_subplot(111,projection="3d")
ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2])
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
ax.set_title("3d")

from sklearn.cluster import KMeans
from kneed import KneeLocator 
wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)

knee=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing")
optimal_k=knee.elbow
print(f"best K ={optimal_k}")

plt.plot(range(1,11),wcss,marker='o')

from sklearn.metrics import silhouette_score
scores=[]
for k in range(2,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    labels=kmeans.fit_predict(X_pca)
    score=silhouette_score(X_pca,labels)
    scores.append(score)
plt.plot(range(1,11),wcss,marker='o')
plt.xlabel("k")
plt.ylabel("silhouette_score")

#combine
k_range=range(2,11)
fig,ax1=plt.subplots(figsize=(8,6))
ax1.plot(k_range,wcss[:len(k_range)],marker='o',color="blue")
ax1.set_xlabel("K")
ax1.set_ylabel("WCSS")
ax2=ax1.twinx()
ax2.plot(k_range,scores[:len(k_range)],marker="X",color="red",linestyle="--")
ax2.set_ylabel("SS")

kmeans=KMeans(n_clusters=4,random_state=42)
label_kmeans=kmeans.fit_predict(X_pca)

fig=plt.figure(figsize=(8,6))
ax= fig.add_subplot(111,projection="3d")
ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=label_kmeans)

from sklearn.cluster import AgglomerativeClustering
agg_clf=AgglomerativeClustering(n_clusters=4,linkage="ward")
label_agg=agg_clf.fit_predict(X_pca)


fig=plt.figure(figsize=(8,6))
ax= fig.add_subplot(111,projection="3d")
ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=label_agg)

df_clean=df_clean.drop("Labels",axis=1,errors="ignore")
df_clean["Clusters"]=label_agg
df_clean.head()

pal=["red","blue","yellow","green"]
sns.countplot(x=df_clean["Clusters"],palette=pal,hue=df_clean["Clusters"])

sns.scatterplot(x=df_clean["Total_spending"],y=df_clean["Income"],hue=df_clean["Clusters"],palette=pal)

#Summary
cluster_summary=X.groupby(df_clean["Clusters"]).mean()
print(cluster_summary)

#From The Above Data We Can Form $ Group Where We can Give Discount or Special Services To Regular Customers

"""
[C0] - Family Shoppers (Discounts & Coupons):
    - More children
    - Poor response
    - Partners
    - Web visits: High (↑)
    - Web, catalog, store purchases: Very low (↓↓)
----------------------------------------------------
[C1] - Loyalty Programs (Price Sensitive):
    - Fewer children
    - Slightly higher age (↑)
    - Average response
    - Partners
    - Web visits: Low (↓)
    - Store, catalog, and web purchases: High (↑)
-----------------------------------------------------
[C2] - Digital Browsers (Sales & Heavy Discount Coupons):
    - More children
    - Average response
    - Alone
    - Web visits: High (↑)
    - Web, catalog, store purchases: Extremely low (↓↓↓)
-------------------------------------------------------
[C3] - Best ROI / High Value Singles (Premium Services):
    - Fewer children
    - Slightly higher age (↑)
    - Best response
    - Alone
    - Web visits: Low (↓)
    - Store, catalog, and web purchases: High (↑) 
"""