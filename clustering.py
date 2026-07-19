import faiss
import numpy as np

class ClusterIndex:
    def __init__(self, n_clusters=1024, dim=256, nredo=5, niter=20):
        self.kmeans = faiss.Kmeans(dim, n_clusters, niter=niter, nredo=nredo, verbose=False)

    def fit(self, embeddings, init_centroids=None):
        self.kmeans.train(embeddings, init_centroids=init_centroids)

    def assign(self, embeddings):
        _, I = self.kmeans.index.search(embeddings, 1)
        return I.flatten()

    def nearest_clusters(self, embedding, k=3):
        _, I = self.kmeans.index.search(embedding.reshape(1, -1), k)
        return I[0]


def assign_fixed_centroids(embeddings, centroids):
    """Assign embeddings to the nearest centroid without running K-means.

    Used when the cluster head is frozen (Stage 2 epoch 1) and Stage 1 centroids
    are available. Skipping K-means iteration preserves the same-cluster assignments
    from Stage 1 exactly.
    """
    centroids_f32 = centroids.astype("float32")
    index = faiss.IndexFlatL2(centroids_f32.shape[1])
    index.add(centroids_f32)
    _, I = index.search(embeddings.astype("float32"), 1)
    return I.flatten()
