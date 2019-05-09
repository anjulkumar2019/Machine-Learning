import numpy as np
class kmean:
    
    def distEclud(vecA, vecB):
        return np.sqrt(sum(np.power(vecA - vecB, 2)))
    
    def randCent(dataSet, k):
        n = np.shape(dataSet)[1]
        centroids = np.mat(np.zeros((k,n)))
        for j in range(n):
            minJ = min(dataSet[:,j])
            rangeJ = float(max(dataSet[:,j]) - minJ)
            centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
        return centroids
    
    def kMeans(dataSet, k):
        m = np.shape(dataSet)[0]
        clusterAssment = np.mat(np.zeros((m,2)))
        centroids = self.randCent(dataSet, k)
        clusterChanged = True
        while(clusterChanged):
            clusterChanged = False
            for i in range(m):
                minDist = np.inf; minIndex = -1
                for j in range(k):
                    distJI = self.distEclud(centroids[j,:],dataSet[i,:])
                    if distJI < minDist:
                        minDist = distJI; minIndex = j
                if clusterAssment[i,0] != minIndex: 
                    clusterChanged = True
                clusterAssment[i,:] = minIndex,minDist**2
            print(centroids)
            for cent in range(k):
                ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
                centroids[cent,:] = np.mean(ptsInClust, axis=0)
        return(centroids, clusterAssment)
