import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def intro():
    """
    LouvainDP(G,s):
    -------------------------------------------------------------
    Input:
    -------------------------------------------------------------
        G: Input Graph
        k: group side to create supernodes.
        epsilon: privacy budget
    
    Output:
    -------------------------------------------------------------
        C: the noisy partition of the graph into communities.
    
    Algorithm:
    -------------------------------------------------------------
        1. Import Graph
        2. Initialize new graph G1 with n1 nodes v1 edges.
        3. Initialize epsilon, epsilon1, alpha.
        4. Get a random permutation of edges.
        6. Calculate m1, m0, theta, s.
        7. Edge Noise Addition.
        8. run Louvain method.

    """


def graph_info(G, title):
    """
        Prints the summary of a graph.

    """
    average_degree = np.floor(sum(deg for _, deg in G.degree()) / G.number_of_nodes())

    print(title)
    print("------------------------------------")
    print("number of nodes:", G.number_of_nodes())
    print("number of edges:", G.number_of_edges())
    print("directed graph:", nx.is_directed(G))
    print("average degree:", average_degree)
    print("number of connected components:", nx.number_connected_components(G))
    print("density:", nx.density(G))
    print("average clusting:", round(nx.average_clustering(G), 4))


def cluster_analysis(community, title, k=None, epsilon=None ):
    clusters = []

    [clusters.append(len(community)) for community in community]
    if k:
        print("k:", k)
    if epsilon:
        print("epsilon", epsilon)
    print("number of clusters:", len(community))
    print("largest:", max(clusters))
    print("smallest:", min(clusters))
    print("mean cluster size:", round(np.mean(clusters),2))
    print("standard deviation:", round(np.std(clusters),2))

    plt.hist(clusters)
    plt.title(title)
    plt.xlabel('cluster size')
    plt.ylabel('# of clusters')
    plt.show()