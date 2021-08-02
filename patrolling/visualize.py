from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import networkx as nx
import os

from .agents import AgentType

def visualize_graphs(G, heatmaps, n_row, n_col, ax_titles=None, figsize=(10, 10),
                     edge_width=2.0, arc_rad=0.1,
                     cmap=plt.cm.get_cmap('viridis'), colorbar=True):
    """
    Visualize a graph with several different edge color maps at once, arranged
    in a grid.

    Parameters
    ----------
    G : networkx.digraph.DiGraph
        The graph to be visualized.
    heatmaps : list of numpy.ndarray
        A list of heatmaps. Each heatmap contains the intensity of each edge,
        index by an edge ID.
    n_row : int
        The desired number of rows in the output figure.
    n_col : int
        The desired number of colums in the output figure.
    ax_titles : list of str, optional
        A list containing a title for each graph drawing.
    figsize : tuple of float
        The figure size.
    edge_width : float
        Edge width (the default is 2.0).
    arc_rad : float
        The arc radius (default is 1.0).
    cmap : matplotlib.colors.ListedColormap
        The colormap for the edges (Default is the 'viridis' color map).
    colorbar : bool
        Specifies whether to add a color bar to the figure (Default is True).
    """

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    edges = []

    if ax_titles is None:
        ax_titles = [''] * len(heatmaps)

    for ax, heatmap, title in zip(axes.ravel(), heatmaps, ax_titles):
        edges += visualize_graph(G, ax, x=heatmap, edge_width=edge_width,
                                 cmap=cmap, arc_rad=arc_rad)
        ax.set_title(title)

    if colorbar:
        fig.colorbar(PatchCollection(edges))

def visualize_graph(G, ax, x=None, edge_width=2.0, node_size=300, arc_rad=0.1,
                    cmap=plt.cm.get_cmap('viridis')):
    """
        Visualize a networkx graph on the specified matplotlib axis, optionally
        with edge weights specified.

        G : networkx.digraph.DiGraph
            The graph to be displayed.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axis on which to visualize the graph.
        x : numpy.ndarray, optional
            The weight associated with each edge, index by edge ID.
        edge_width : float
            The edge width (Default is 2.0).
        node_size : float
            The size of the vertices (Default is 300).
        arc_rad : float
            The arc radius (Default is 0.1).
        cmap : matplotlib.colors.ListedColormap
            The colormap for the edges (Default is the 'viridis' color map).

        Returns
        -------
        edges : list of matplotlib.patches.FancyArrowPatch
            A list of matplotlib patches, each one representing an edge in the
            graph.
    """

    pos = dict([(v, v) for v in G.nodes])

    if x is None:
        nx.draw(G, ax=ax, pos=pos, node_size=node_size, connectionstyle=f'arc3, rad = {arc_rad}')
        edges = []
    else:
        nx.drawing.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=node_size)
        edges = nx.drawing.draw_networkx_edges(G, pos=pos, ax=ax, width=edge_width, edge_color=x,
                                               edge_cmap=cmap, connectionstyle=f'arc3, rad={arc_rad}')

    return edges

def visualize_star_top_patrolling(simulation, G, video_name, edge_width=2.0,
                                  node_size=300, symbol_size=0.1, fps=1,
                                  figsize=(10, 10)):
    """
        Creates a video of a patrolling simulation and saves it in a directory
        titled videos.

        Parameters
        ----------
        simulation : dict
            A dictionary describing a patrolling simulation with one intruder
            and an arbitrary number of patrollers. See documentation for the
            simulate_star_top_patrolling function in patrolling/simulation.py
            for details.
        G : networkx.digraph.DiGraph
            The graph to be displayed in the video.
        video_name : str
            The name of the output video file.
        edge_width : float
            The edge width (default is 2.0).
        node_size : float
            The size of the vertices (default is 300).
        symbol_size : float
            The size of the agent symbols (default is 0.1).
        fps : int
            The video framerate (default is 1).
        figsize : tuple of float
            The size of the figure that will be used to display each frame
            (default is (10, 10)).
    """

    fig = plt.figure(figsize=figsize)
    if not os.path.exists('videos'):
        os.mkdir('videos')

    def make_frame(t):
        t = int(t)
        fig.clf()
        ax = fig.subplots(1, 1)
        ax.axis('equal')
        visualize_graph(G, ax, edge_width=edge_width, node_size=node_size, arc_rad=0)
        for patroller in simulation['patrollers']:
            v = patroller['history'][t]
            draw_agent_symbol(ax, v, patroller['type'], 'green', symbol_size)
        for intruder in simulation['intruders']:
            v = intruder['history'][t]
            if v is not None:
                draw_agent_symbol(ax, v, intruder['type'], 'red', symbol_size)

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=simulation['T'])
    animation.write_videofile(f'videos/{video_name}', fps=fps)

def draw_agent_symbol(ax, pos, agent_type, color, symbol_size):
    """
    Draw a symbol representing an agent on the specified axis.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis on which to draw the symbol.
    pos : Tuple of float
        The position to draw the symbol (specifies the center of the symbol).
    agent_type : <enum 'AgentType'>
        The type of the agent.
    color : str
        The color of the symbol.
    symbol_size : float
        The size of the symbol.
    """

    x, y = pos
    if agent_type == AgentType.QUAD:
        ax.add_patch(mpatches.Circle((x, y), symbol_size / 2, color=color, zorder=2))
    elif agent_type == AgentType.GROUND:
        x_c = x - symbol_size / 2
        y_c = y - symbol_size / 2
        ax.add_patch(mpatches.Rectangle((x_c, y_c), width=symbol_size, height=symbol_size,
                                        color=color, zorder=2))
