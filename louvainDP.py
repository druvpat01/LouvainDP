import numpy as np
import networkx as nx

class LouvainDP():
    def __init__(self,G):
        super(LouvainDP,self).__init__()
        self.G = G
        self.node_list = list(G.nodes())
        self.num_of_nodes = G.number_of_nodes()
        self.num_of_edges = G.number_of_edges()
        self.epsilon = 0.5 * np.log(self.num_of_nodes)
        self.epsilon2 = 0.01
        self.noise_level = 1/self.epsilon2
        self.epsilon1 = self.epsilon - self.epsilon2
        self.alpha = np.exp(-self.epsilon1)

    def supergraph_mapping(self,n1,k):
        """
            Creates mapping from graph nodes to supergraph node, and also the other way around.

            Parameters:
                n1 (int): number of nodes in G1 graph.
                k  (int): number of nodes in each supernode
            
            Returns:
                dict: mapping from node -> supernode | (key,value) -> (str, str)
                dict: mapping from supernode -> node | (key,value) -> (str, list)
        """

        node_to_supernode = {}      # (key,value) -> (str, str) : node value -> supernode value
        supernode_to_node = {}      # (key,value) -> (str, list) : supernode valur -> all nodes in the supernode

        np.random.shuffle(self.node_list)       # random permutation of nodes

        for supernode in range(n1):
            nodes = self.node_list[supernode * k: (supernode+1) * k]      # grouping node into supernodes. 
            supernode_to_node[supernode+1] = nodes

            for node in nodes:
                node_to_supernode[node] = supernode+1

        return node_to_supernode, supernode_to_node
    
    def create_supergraph(self, node_to_supernode, supernode_to_node):
        """
            Creates supergraph G1.

            Parameters:
                node to supernode (dict): mapping from node -> supernode | (key,value) -> (str, str)
                supernode to node (dict): mapping from supernode -> node | (key,value) -> (str, list)

            Returns:
                nx.graph: supergraph G1
        """

        graph = nx.Graph()      # empty graph

        # if any node within a supernode is connected to a node in another supernode, we create an edge between the supernodes.
        # weight of an edge in supergraph: number of out connection of between two supernodes.
        for nodes in supernode_to_node.values():
            for u in nodes:
                for v in self.G.neighbors(u):                                                   # Conditions for nodes:
                    if(u != v                                                                      # no self edge
                       and (u in node_to_supernode.keys()) and (v in node_to_supernode.keys())     # exists in a supernode
                       and node_to_supernode[u] != node_to_supernode[v]):                          # not in the same supernode.
                        
                        supernode1 = node_to_supernode[u]
                        supernode2 = node_to_supernode[v]

                        if graph.has_edge(supernode1,supernode2):
                            graph[supernode1][supernode2]['weight'] += 1.0    # increment weight if edge already exists between supernodes
                        else:
                            graph.add_edge(supernode1, supernode2, weight = 1.0)  # creates new edge between supernodes

        return graph
    
    # MAKE CHANGES IN THIS FUNCTION.
    def sample_zero_edge_weight(self,theta, s):
        """
            Draw samples from the distribution Pr[X <= x] = 1 - alpha^(x - theta + 1).

            Parameters:
                alpha (float): Parameter of the distribution (0 < alpha < 1).
                theta (float): Shift parameter of the distribution.
                size (int): Number of samples to draw.

            Returns:
                np.ndarray: Array of sampled values.
        """
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be in the range (0, 1).")
        
        # Generate uniform random values
        U = np.random.uniform(0, 1, s)
        
        # Apply the inverse CDF to get samples
        weights = theta - 1 + np.log(1 - U) / np.log(self.alpha)
        
        return weights
    
    def supergraph_commuinties_to_graph_communities(self, supergraph_communities, supernode_to_node):
        """
            Obtains dp commuinities based on the supergraph communities.

            Parameters:
                supergraph communities (list[set()]): communities formed for supergraph G1
                supernode to node   (dict): mapping from supernode -> node | (key,value) -> (str, list)
            
            Returns:
            list[set()]: DP communities form for graph G
        """
        dp_communities = []     # map supergraph communities to graph communities.

        for community in supergraph_communities:
            group = set()
            for supernode in community:
                temp = supernode_to_node[supernode]
                for node in temp:
                    group.add(node)
            dp_communities.append(group)

        # check if the partition is valid or not, if not make it a valid partition.
        total_nodes = len(self.node_list)
        dp_total_nodes = sum(len(community) for community in dp_communities)

        # Checks if all nodes are assigned.
        if total_nodes != dp_total_nodes:
            t1 = set(dp_communities[0])

            for i in range(1,len(dp_communities)):
                t1 = t1 | dp_communities[i]

            missing_nodes = list(set(self.node_list) - t1)      # nodes which are not assigned to any community.

            limit = len(dp_communities)

            # Add missing nodes to dp communities at random.
            for node in missing_nodes:
                index = np.random.randint(0,limit)
                dp_communities[index].add(node)


        # Check for overlaps i.e node must belong to atmost one community.
        for i, community1 in enumerate(dp_communities):
            for j, community2 in enumerate(dp_communities):
                if i != j and community1 & community2:
                    raise ValueError(f"Partition overlaps detected between community {i} and {j}.")

        return dp_communities

    def louvain_dp(self,k):
        """
            Finds communities from a noisy graph.
            
            Parameters: 
                G (nx.graph): input graph
                k (int): number of node in each supernode
                epsilon (float): the privacy budget
            
            Returns:
                list[set()]:  the noisy partition of the graph into communities

        """
        # creating a supergraph
        num_of_nodes_G1 = self.num_of_nodes // k
        node_to_supernode, supernode_to_node = self.supergraph_mapping(num_of_nodes_G1, k)      # mapping graph to supergraph
        G1 = self.create_supergraph(node_to_supernode, supernode_to_node)               # G1: supergraph

        num_of_edges_G1 = G1.number_of_edges()
        m1 = num_of_edges_G1 + np.random.laplace(0, self.noise_level)
        m0 = num_of_edges_G1* (num_of_edges_G1+1)/2
        theta = np.ceil(np.emath.logn(self.alpha, (1+self.alpha)*m1/(m0-m1)))
        s = int(np.ceil((m0-m1)*(self.alpha**theta)/(1+self.alpha)))

        # make this code more generic by thersholding and sampling edges
        weak_edges = []
        for (u,v) in list(G1.edges()):
            G1[u][v]['weight'] += np.random.geometric(self.alpha)

            if G1[u][v]['weight'] < theta:
                weak_edges.append((u,v))
                G1.remove_edge(u,v)
        
        # adds the zero-edge weigths s to the graph.
        zero_edge_weights = self.sample_zero_edge_weight(theta, s)
        for i,(u,v) in enumerate(weak_edges[:s]):
            G1.add_edge(u,v, weight = zero_edge_weights[i]) 
    
        # run louvain on G1 to obtain C1. where C1 is the noisy partition of supergraph.
        supergraph_communities = nx.community.louvain_communities(G1,weight='weight',resolution=1, seed=123)

        dp_communities  = self.supergraph_commuinties_to_graph_communities(supergraph_communities, supernode_to_node)

        return dp_communities, self.epsilon
