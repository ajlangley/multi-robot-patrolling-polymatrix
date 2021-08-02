import networkx as nx
import numpy as np

def generate_grid_graph(m, n, edge_length=1.0):
    """
    Generates a grid graph.

    Parameters
    ----------
    m : int
        The height of the grid, in number of vertices.
    n: int
        The with of the grid, in number of vertices.
    """

    grid_undirected = nx.generators.grid_2d_graph(m, n)
    grid_digraph = nx.DiGraph(grid_undirected)

    for i, e in enumerate(grid_digraph.edges()):
        grid_digraph.add_edge(*e, length=edge_length, eid=i)

    return grid_digraph

def generate_random_grid_with_holes(m, n, p, seed=105):
    """
    Generates a random, connected grid with holes. This is done by starting with
    a m x n grid and iterating through each vertex, randomly choosing to remove
    each one.

    Parameters
    ----------
        m : int
            The height of the base grid.
        n : int
            The width of the base grid.
        p : float
            The probability of removing a vertex.
        seed : int (optional)
            The random seed.

    Returns
    -------
        G : networkx.digraph.DiGraph
            A grid graph with randomly created holes.
    """

    G = generate_grid_graph(m, n)
    np.random.seed(seed)

    for v in list(G.nodes):
        if v == ((0, 0)):
            continue
        r = np.random.uniform(0, 1)
        if r <= p:
            G.remove_node(v)
    for v in list(G.nodes):
        if not nx.has_path(G, v, (0, 0)):
            G.remove_node(v)

    eids = dict([(e, {'eid': i}) for i, e in enumerate(G.edges)])
    nx.set_edge_attributes(G, eids)

    return G

# def load_nx_digraph_from_json(node_fp, edge_fp):
#     with open(node_fp, 'r') as f_node:
#         nodes = json.load(f_node)
#         nodes = [tuple(v) for v in nodes]
#     with open(edge_fp, 'r') as f_edge:
#         edges = json.load(f_edge)
#         edges = [(tuple(e[0]), tuple(e[1])) for e in edges]
#
#     G = nx.Graph()
#     G.add_nodes_from(nodes)
#     G.add_edges_from(edges)
#     G_digraph = nx.digraph.DiGraph(G)
#
#     for i, e in enumerate(G_digraph.edges()):
#         G_digraph.add_edge(*e, eid=i)
#
#     return G_digraph

def generate_gridworld_vis_graph(G, sensor_range):
    """
    Given a sensor range, compute the visibility graph given the environment
    graph G. Vertices are considered visibile from a given node (x,y) when their
    x or y coordinate is the same and they are within sensor_range.

    G : networkx.digraph.DiGraph
        The graph representation of the environment.
    sensor_range:
        The sensing range of the agent.

    Returns
    -------
    G_vis : networkx.digraph.DiGraph
        The visibility graph. An edge from v1 to v2 means that v2 is visible
        from v2.
    """
    
    m = np.max([v[1] for v in G.nodes])
    n = np.max([v[0] for v in G.nodes])
    G_vis = nx.digraph.DiGraph()
    G_vis.add_nodes_from(G.nodes)

    for i in range(n + 1):
        for j in range(m + 1):
            v1 = (i, j)
            if v1 in G.nodes:
                for v2 in G.nodes:
                    if np.linalg.norm(np.array(v1) - np.array(v2)) <= sensor_range:
                        if nx.shortest_path_length(G, v1, v2) <= sensor_range:
                            if v1[0] == v2[0] or v1[1] == v2[1]:
                                G_vis.add_edge(v1, v2)

    return G_vis
