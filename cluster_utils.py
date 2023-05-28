from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score


def score_clusters(X, predictions, silhouette = True, verbose=False):
    """Evaluate how good our cluster label predictions are"""

    db_score = davies_bouldin_score(X=X, labels=predictions)

    ch_score = calinski_harabasz_score(X=X, labels=predictions)
    #the silhouette score is the slowest to compute ~90 secs
    s_score = silhouette_score(X=X, labels=predictions, metric='euclidean')

    if verbose:
        print("David Bouldin score: {0:0.4f}".format(db_score))
        print("Calinski Harabasz score: {0:0.3f}".format(ch_score))
        print("Silhouette score: {0:0.4f}".format(s_score))

    return db_score, ch_score, s_score