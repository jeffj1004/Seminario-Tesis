import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

np.random.seed(2017)

educ_data = pd.read_csv("Cruce 4.csv")

educ_data.head()
educ_data.describe()

educ_data["TicketPromMes"] = (educ_data["TicketPromMes"]-np.mean(educ_data["TicketPromMes"]))/np.std(educ_data["TicketPromMes"])
educ_data["RentaPromMes"] = (educ_data["RentaPromMes"]-np.mean(educ_data["RentaPromMes"]))/np.std(educ_data["RentaPromMes"])

total_itter = 3*len(educ_data.index)

nodes_num = 7

input_dim = len(educ_data.columns)

learn_init = 0.1

Weight_mat = 4*np.random.rand(input_dim,nodes_num)-2

print("Initialized weight matrix,", Weight_mat)

for itter in range(total_itter):
    
    dist_bmu = float("inf")

    row_index = np.random.randint(len(educ_data.index))
    
    data_chosen = educ_data.loc[[row_index]]
    
    for node in range(nodes_num):
        
        dist_neuron = np.linalg.norm(data_chosen-Weight_mat[:,node])
        
        if dist_neuron < dist_bmu:
            
            dist_bmu = dist_neuron
            
            weight_bmu = Weight_mat[:,node]
            index_bmu = node
            
    learn_rate = learn_init*np.exp(-itter/total_itter)

    Weight_mat[:,index_bmu] = np.add(weight_bmu,learn_rate*(np.subtract(data_chosen,weight_bmu)))

print("Trained weights from SOM,", Weight_mat)

group = np.zeros(len(educ_data.index))
    
for index, data in educ_data.iterrows():
    
    dist_cluster = float("inf")
    
    for centroid in range(nodes_num):
        
        dist_centroid = np.linalg.norm(data-Weight_mat[:,centroid])

        if dist_centroid < dist_cluster:

                dist_cluster = dist_centroid

                group[index] = centroid+1
            
educ_data["group"] = group
educ_data.head()

educ_data[educ_data.group == 1].describe()
educ_data[educ_data.group == 2].describe()
educ_data[educ_data.group == 3].describe()
educ_data[educ_data.group == 4].describe()
educ_data[educ_data.group == 5].describe()
educ_data[educ_data.group == 6].describe()
educ_data[educ_data.group == 7].describe()