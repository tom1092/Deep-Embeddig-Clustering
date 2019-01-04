from sklearn.cluster import KMeans

def create_K_means(numClusters):
    '''
        create a k means instance
        :param numClusters: the number of clusters extracted
        :return: k means object
    '''
    km = KMeans(n_clusters=numClusters, n_jobs=8)
    return km


def fit_k_means(featureDataset, km):

    '''
    create clusters in clusterFolder folder

    :param featureDataset: the encoded dataset
    :param km: the k means instantiated object
    :return: a list with the reference label of each feature
    '''

    print "...Clustering..."
    km.fit(featureDataset)

    #centroids = km.cluster_centers_  uncomment this line to view centroids
    return km.labels_ , km.cluster_centers_


