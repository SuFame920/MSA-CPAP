from sklearn.cluster import KMeans
import numpy as np

def gen_npy(data, dataset="mosi", n_clusters=50, model="text_bert_mean"): 
    #data = data.detach().numpy()
    print(data.shape)
    print(data.dtype)
    reshaped_data = data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reshaped_data)
    centers = kmeans.cluster_centers_
    centers = centers.astype(np.float32)
    print(centers.shape)
    print(centers.dtype) 
    np.save(f'npy_folder/kmeans_{dataset}-{n_clusters}_{model}.npy', centers)