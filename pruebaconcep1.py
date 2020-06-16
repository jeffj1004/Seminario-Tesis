import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datos=pd.read_csv('Cruce 4.csv')
df=pd.DataFrame(datos)
x=df['RentaPromMes'].values
y=df['TicketPromMes'].values

print("Valor promedio de TC: ",df['RentaPromMes'].mean())
info=df[['RentaPromMes','TicketPromMes']].as_matrix()
print(info)

X=np.array(list(zip(x,y)))
print(X)

kmeans=KMeans(n_clusters=7)
kmeans=kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

colors=['r.',
"b.",
"g.",
"c.",
"k.",
"y.",
"m.",
'r.',
"b.",
"g.",
"c.",
"k.",
"y.",
"m.",
"b."]

for i in range(len(X)):
    print("Coordenada: ",X[i]," Label: ",labels[i])
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=5,zorder=10)
plt.show()