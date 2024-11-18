import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

def generate_scale_free_network(n, m, pct_0):
    """
    Generate a scale-free network using the Barabási-Albert model.
    The 'op' attribute is initialized after the network structure is created,
    with probability of op=1 inversely proportional to node degree.
    
    Parameters:
    n (int): Number of nodes
    m (int): Number of edges to attach from a new node to existing nodes
    
    Returns:
    G (networkx.Graph): A scale-free network
    """
    G = nx.barabasi_albert_graph(n, m)

    # Get degrees of all nodes and sort them in descending order
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)

    # Calculate the number of nodes to set to 0
    num_nodes_0 = int(n * pct_0)

    # Set op values
    for i, (node, _) in enumerate(degrees):
        if i < num_nodes_0:
            G.nodes[node]['op'] = 0
        else:
            G.nodes[node]['op'] = 1

    return G

def average_neighbor_op(G, node):
    """
    Calculate the average 'op' value of all nodes connected to the given node.
    
    Parameters:
    G (networkx.Graph): The network
    node: The node to analyze
    
    Returns:
    float: Average 'op' value of neighboring nodes
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return None  # Return None if the node has no neighbors
    neighbor_ops = [G.nodes[neighbor]['op'] for neighbor in neighbors]
    return sum(neighbor_ops) / len(neighbor_ops)

def global_average_neighbor_op(G):
    """
    Calculate the average of average neighbor 'op' values for all nodes in the graph.
    
    Parameters:
    G (networkx.Graph): The network
    
    Returns:
    float: Global average of average neighbor 'op' values
    """
    avg_neighbor_ops = [average_neighbor_op(G, node) for node in G.nodes() if average_neighbor_op(G, node) is not None]
    return sum(avg_neighbor_ops) / len(avg_neighbor_ops)

def print_stats(G, print_all=True):
    """
    Print various statistics about the network
    Args:
        G (networkx.Graph): The network

    Returns:
        Nothing
    """
    if print_all:
        # Print network statistics
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.4f}")

    # Count nodes with op=1 and op=0
    op_1_count = sum(1 for _, data in G.nodes(data=True) if data['op'] == 1)
    op_0_count = G.number_of_nodes() - op_1_count
    print(f"Nodes with op=1: {op_1_count} ({op_1_count/G.number_of_nodes()*100:.2f}%)")
    print(f"Nodes with op=0: {op_0_count} ({op_0_count/G.number_of_nodes()*100:.2f}%)")

    # Analyze relationship between degree and op value
    degrees = dict(G.degree())
    op_values = nx.get_node_attributes(G, 'op')
    degree_op_1 = [degree for node, degree in degrees.items() if op_values[node] == 1]
    degree_op_0 = [degree for node, degree in degrees.items() if op_values[node] == 0]

    print(f"\nAverage degree of nodes with op=1: {np.mean(degree_op_1):.2f}")
    print(f"Average degree of nodes with op=0: {np.mean(degree_op_0):.2f}")

    # Demonstrate the use of average_neighbor_op function
    # print("\nDemonstrating average_neighbor_op function:")
    # for i in range(5):  # Show results for first 5 nodes
    #     avg_op = average_neighbor_op(G, i)
    #     print(f"Node {i}: op = {G.nodes[i]['op']}, average neighbor op = {avg_op:.2f}")

    # Calculate and print global average of neighbor op values using the new function
    global_avg = global_average_neighbor_op(G)
    print(f"\nGlobal average of average neighbor op values: {global_avg:.4f}")

    # Analyze correlation between a node's op and its average neighbor op
    node_ops = [G.nodes[node]['op'] for node in G.nodes()]
    avg_neighbor_ops = [average_neighbor_op(G, node) for node in G.nodes()]
    correlation = np.corrcoef(node_ops, avg_neighbor_ops)[0, 1]
    print(f"Correlation between node op and average neighbor op: {correlation:.4f}")

def collect_data(n, m, pct_0):
    G = generate_scale_free_network(n, m, pct_0)
    op_1_count = sum(1 for _, data in G.nodes(data=True) if data['op'] == 1)
    op_0_count = G.number_of_nodes() - op_1_count
    op_1_pct = op_1_count / G.number_of_nodes() * 100
    op_0_pct = op_0_count / G.number_of_nodes() * 100

    # Analyze relationship between degree and op value
    degrees = dict(G.degree())
    op_values = nx.get_node_attributes(G, 'op')
    degree_op_1 = [degree for node, degree in degrees.items() if op_values[node] == 1]
    degree_op_0 = [degree for node, degree in degrees.items() if op_values[node] == 0]

    op_1_avg_degree = np.mean(degree_op_1)
    op_0_avg_degree = np.mean(degree_op_0)

    # Calculate and print global average of neighbor op values using the new function
    global_avg = global_average_neighbor_op(G)

    # calculate discrepancy of observed vs actual opinion
    discrepancy = global_avg - (op_1_pct / 100)

    return {
        "n": n,
        "m": m,
        "pct_0": pct_0,
        "op_1_count": op_1_count,
        "op_0_count": op_0_count,
        "op_1_pct": op_1_pct,
        "op_0_pct": op_0_pct,
        "op_1_avg_degree": op_1_avg_degree,
        "op_0_avg_degree": op_0_avg_degree,
        "global_avg": global_avg,
        "discrepancy": discrepancy,
    }

def run_sweep(n, pct_0_max, m_max=10, reps=10, plot=False, write_output=False):
    pct_0_max_int = int(pct_0_max * 100)
    params = [(m, pct_0) for m in range(1, m_max + 1) for pct_0 in [x/100 for x in range(5, pct_0_max_int+5, 5)]]
    data = [collect_data(n, m, pct_0) for m, pct_0 in params*reps]
    df = pd.DataFrame.from_records(data)

    df = df.groupby(by=["m", "pct_0"]).agg("mean")
    df = df.reset_index()
    if write_output:
        df.to_csv("scale-free-network.csv", index=False)

    if plot:
        df.plot.scatter(x="m", y="global_avg")
        plt.xlabel("# of edges to attach from new nodes")
        plt.ylabel("Average neighbor opinion")
        plt.title("Average neighbor opinion over 10 runs per m")
        plt.show()

# Generate a scale-free network
n = 1000  # Total number of nodes
full_sweep = True
sweep_repetitions = 100
m_max = 100
pct_0_max = 0.45

if full_sweep:
    run_sweep(n, pct_0_max, m_max=m_max, reps = sweep_repetitions, plot=True, write_output=True)

else:
    m = 2     # Number of edges to attach from a new node to existing nodes
    G = generate_scale_free_network(n, m)
    print_stats(G)

    # Plot the network
    plt.figure(figsize=(12, 8))
    node_colors = ['red' if G.nodes[node]['op'] == 1 else 'blue' for node in G.nodes()]
    nx.draw(G, node_size=20, node_color=node_colors, edge_color='gray', with_labels=False)
    plt.title("Scale-Free Network with Degree-Dependent Node Operations")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


