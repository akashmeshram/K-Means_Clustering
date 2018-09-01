# -*- coding: utf-8 -*-
"""
@author: Akash Meshram (16MT10005)

MIES(EC60091) Assignment
Computer Assignment on Clustering
"""
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

#import dataset as numpy array
df = pd.read_excel('dataset.xlsx', header=None).values

f = open('result.txt', 'w')

#setup
k_val = 5 # range of k-centroids
err = [] #list of error for each k value
itern = 20 #no. of iteration on each k value
print('k', 'error')
f.write('k'+" "+ 'error' +"\n")

# iterate over each k
for k in range(1,k_val+1):
	#initialize the centroids, the random 'k' elements in the dataset will be initial centroids
    clusters = df[np.random.randint(df.shape[0], size=k), :]

    #List for bucket for each centroid, where nearest data point will be added
    clusters_group = []
    #Create k bucket for k centroids resp.
    for _ in range(k):
        clusters_group.append([])

    #begin iterations
    for _ in range(itern):
    	#for each point in df
        for row in df:
        	#array for diststace from each centroid
            dist = []
            for c in clusters:
            	#the distance between the point and cluster centers
                dist.append(np.linalg.norm(row-c))
            #choose the nearest centroid, using index
            min_index, min_value = min(enumerate(dist), key=operator.itemgetter(1))
            #adding point to bucket of nearest centroid
            clusters_group[min_index].append(row)
        
        #average the cluster datapoints to re-calculate the centroids
        for i in range(k):
            temp = np.asarray(clusters_group[i])
            x=np.sum(temp[:,0])/len(temp)
            y=np.sum(temp[:,1])/len(temp)
            clusters[i] = [x,y]
            #for each itration buckets are emptied,
            #except for last, so as to calculate error later
            if (_ != itern-1):
                clusters_group[i] = []
    
    #claculate error
    error = 0
    for i in range(k):
        temp = np.asarray(clusters_group[i])
        error += np.sum(np.linalg.norm(temp-clusters[i]))
    
    print(k, error)
    f.write(str(k)+" "+str(error)+'\n')
    #append error to list, so as to plot later
    err.append(error)

#begin plot            
fig = plt.figure()
plt.xlabel("K-value")
plt.ylabel("error")
plt.plot(range(1,k_val+1),err,'-r')
fig.savefig('result_plot.png')
f.close()