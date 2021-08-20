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

def load_graph_from_npz(graph_fp):
    graph_archive = np.load(graph_fp)
    nodes = [tuple(v) for v in graph_archive['nodes']]
    edges = [(tuple(source), tuple(dest)) for source, dest in graph_archive['edges']]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i, e in enumerate(edges):
        source, dest = e
        length = np.linalg.norm(np.array(source) - np.array(dest), 2)
        G.add_edge(*e, eid=i, length=length)

    G_directed = nx.digraph.DiGraph(G)

    return G_directed

def shift_scale_graph(G, origin, res):
    G_new = nx.digraph.DiGraph()
    old_to_new = dict()

    for v in G.nodes:
        x_new = (v[0] - origin[1]) / res
        y_new = (v[1] - origin[0]) / res
        v_new = (x_new, y_new)
        G_new.add_node(v_new)
        old_to_new[v] = v_new

    for e in G.edges(data=True):
        e_new = (old_to_new[e[0]], old_to_new[e[1]])
        G_new.add_edge(old_to_new[e[0]], old_to_new[e[1]], eid=e[-1]['eid'],
                       length=e[-1]['length'])

    return G_new

def build_graph_with_speed(G, mov_budget):
    eid = 0
    G_ws = nx.digraph.DiGraph()
    for v1 in G.nodes:
        distances, _ = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(G, v1,
                                                                                    weight='length',
                                                                                    cutoff=mov_budget)
        for v2 in distances.keys():
            length = np.linalg.norm(np.array(v1) - np.array(v2), 2)
            G_ws.add_edge(v1, v2, eid=eid, length=length)
            eid += 1

    return G_ws

def build_vis_graph(G, vis_pairs):
    nodes = list(G.nodes)
    vis_graph = nx.digraph.DiGraph()
    vis_graph.add_nodes_from(G.nodes)
    for vis_pair in vis_pairs:
        v1 = nodes[vis_pair[0]]
        v2 = nodes[vis_pair[1]]
        vis_graph.add_edge(v1, v2)

    return vis_graph

def build_vis_score_dict_from_vis_map(G, vis_map, kernel_size):
    vis_score_dict = dict()
    for v in G.nodes:
        x_int, y_int = int(v[0]), int(v[1])
        left = max(0, x_int - kernel_size)
        right = min(vis_map.shape[1], x_int + kernel_size)
        top = max(0, y_int - kernel_size)
        bottom = min(vis_map.shape[0], y_int + kernel_size)
        vis_region = vis_map[top:bottom, left:right]
        vis_score_dict[v] = np.mean(vis_region)

    return vis_score_dict
