import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
  
datos=pd.read_csv('Cruce 4.csv')
df=pd.DataFrame(datos)
x=df['RentaPromMes'].values
y=df['TicketPromMes'].values

print("Valor promedio de TC: ",df['RentaPromMes'].mean())
info=df[['RentaPromMes','TicketPromMes']].as_matrix()
print(info)

X=np.array(list(zip(x,y)))
print(X)
  
no_of_clusters = range(2,20)
  
for n_clusters in no_of_clusters: 
  
    cluster = KMeans(n_clusters = n_clusters) 
    cluster_labels = cluster.fit_predict(X) 
    silhouette_avg = silhouette_score(X, cluster_labels) 
  
    print("For no of clusters =", n_clusters, 
          " The average silhouette_score is :", silhouette_avg)