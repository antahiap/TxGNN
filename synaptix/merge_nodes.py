

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import pandas as pd

def cluster_words_and_assign(words, similarity_threshold=0.7):
    # Vectorize words with char n-gram TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
    X = vectorizer.fit_transform(words)
    
    # DBSCAN clustering with cosine distance
    distance_threshold = 1 - similarity_threshold
    clustering = DBSCAN(eps=distance_threshold, min_samples=1, metric='cosine')
    labels = clustering.fit_predict(X)
    
    # Group words by cluster label
    clusters = {}
    for word, label in zip(words, labels):
        clusters.setdefault(label, []).append(word)
    
    # Choose a representative for each cluster (here: first word)
    cluster_representative = {label: cluster[0] for label, cluster in clusters.items()}
    
    # Map each word to its cluster representative
    word_to_rep = {}
    for label, cluster in clusters.items():
        rep = cluster_representative[label]
        for w in cluster:
            word_to_rep[w] = rep
            
    return labels, word_to_rep



if __name__ == '__main__':

    print(sys.stdout) 
    print("Your message", flush=True)

    type_m = 'disease'

    synaptix_data = '/home/apakiman/Repo/merck_gds_explr/.images/neo4j/data_synaptix/'
    node_s = pd.read_csv(synaptix_data + 'node_SYNAPTIX.csv', delimiter=',', nrows=100000)


    nodes_m = node_s[node_s['node_type'] == type_m]
    words = nodes_m['node_name'].to_list()
    words = [w.lower() for w in words]

    words = nodes_m['node_name'].str.lower().to_list()

    labels, word_to_rep = cluster_words_and_assign(words, similarity_threshold=0.7)

    nodes_m = nodes_m.copy()  
    nodes_m['merged_name'] = nodes_m['node_name'].str.lower().map(word_to_rep)


    nodes_m.to_csv('name_matching.csv', sep=',')
