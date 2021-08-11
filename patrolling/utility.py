import numpy as np

from .agents import AgentType

class PvIUtilFun:
    """
    A patroller utility function for a game between a patroller and intruder.

    Attributes
    ----------
    pairwise_vis : dict of dict
        A dictionary which specifies the vertices that are visible to an
        agent type from each vertex. The top level dict contains keys of
        type <enum 'AgentType'>, the values of which are dictionaries
        mapping each node to a list of visible nodes.
    occ_utils: dict of dict
        A dictionary which specifies the utility that each type of intruder
        receives for occupying a given node. The top level dict contains
        keys of type <enum 'AgentType'>, the values of which are dictionaries
        mapping each node to an occupation utility for that intruder type.
    util_struct : dict of dict
        A dictionary which specifies the utility gained and lost by each
        agent type for either detecting or being detected by another
        agent, respectively. The top level dict contains keys of type
        <enum 'AgentType'>, the values of which are dictionaries specifying
        utilities for the cases 'detects' and 'is_detected'.

    Methods
    -------
    __call__(e_p, e_i, p_type, i_type)
        Returns the utility obtained by a patroller given the patroller edge,
        intruder edge, patroller type, and intruder type.
    """

    def __init__(self, vis_graphs, occ_utils, util_struct):
        """
        pairwise_vis : dict of dict
            A dictionary which specifies the vertices that are visible to an
            agent type from each vertex. The top level dict contains keys of
            type <enum 'AgentType'>, the values of which are dictionaries
            mapping each node to a list of visible nodes.
        occ_utils: dict of dict
            A dictionary which specifies the utility that each type of intruder
            receives for occupying a given node. The top level dict contains
            keys of type <enum 'AgentType'>, the values of which are dictionaries
            mapping each node to an occupation utility for that intruder type.
        util_struct : dict of dict
            A dictionary which specifies the utility gained and lost by each
            agent type for either detecting or being detected by another
            agent, respectively. The top level dict contains keys of type
            <enum 'AgentType'>, the values of which are dictionaries specifying
            utilities for the cases 'detects' and 'is_detected'.
        """

        self.vis_graphs = vis_graphs
        self.occ_utils = occ_utils
        self.util_struct = util_struct

    def __call__(self, e_p, e_i, p_type, i_type):
        """
        Returns the utility obtained by a patroller given the patroller edge,
        intruder edge, patroller type, and intruder type.

        Parameters
        ----------
        e_p : tuple of nodes
            The edge traversed by the patroller.
        e_i : tuple of nodes
            The edge traversed by the intruder.
        p_type : <enum 'AgentType'>
            The patroller's agent type.
        i_type : <enum 'AgentType'>
            The intruder's agent type.
        """

        p_term, i_term = e_p[1], e_i[1]
        p_detected = p_term in self.vis_graphs[i_type].neighbors(i_term)
        i_detected = i_term in self.vis_graphs[p_type].neighbors(p_term)

        if p_detected and not i_detected:
            u = -self.util_struct[i_type]['detects'] \
                * self.util_struct[p_type]['is_detected'] \
                - self.occ_utils[i_type][i_term]
        elif i_detected and not p_detected:
            u = self.util_struct[p_type]['detects'] \
                * self.util_struct[i_type]['is_detected']
        elif i_detected and p_detected:
            if (p_type, i_type) == (AgentType.QUAD, AgentType.GROUND):
                u = -self.util_struct[i_type]['detects'] \
                * self.util_struct[p_type]['is_detected']
            elif (p_type, i_type) == (AgentType.GROUND, AgentType.QUAD):
                u = self.util_struct[p_type]['detects'] \
                    * self.util_struct[i_type]['is_detected']
            else:
                u = 0
        else:
            u = -self.occ_utils[i_type][i_term]

        if p_detected and not i_detected:
            pass

        return u

def construct_two_player_util_mat(G_dict, p_type, util_fun):
    """
        Constructs a utility matrix for a two player patrolling game.

        Parameters
        ----------
        G_dict : dict of networkx.digraph.DiGraph
            A dictionary of graphs with <enum 'AgentType'> as keys and the graph
            represtations of the environment for the corresponding agent types as
            values.
        p_type : <enum 'AgentType'>
            The type of the patroller agent.
        util_fun : callable
            A callable which takes as argument a patroller edge, an intruder
            edge, the patroller type, and the intruder type, and returns a
            number.

        Returns
        -------
        U : numpy.ndarray
            A |E| X |E|*|AgentType| utility matrix
    """

    U_blocks = []
    G_p = G_dict[p_type]
    for i_type in enumerate(AgentType):
        G_i = G_dict[i_type]
        U_i = np.zeros((G_p.number_of_edges(), G_i.number_of_edges()))
        for e_pat in G_p.edges(data=True):
            for e_int in G_i.edges(data=True):
                p_eid, i_eid = e_pat[-1]['eid'], e_int[-1]['eid']
                i = p_eid
                j = i_eid
                U_i[i, j]  = util_fun(e_pat, e_int, p_type, i_type)
        U_blocks.append(U_i)

    U = np.hstack(U_blocks)

    return U

def construct_star_top_util_mat(G_dict, p_types, util_fun):
    """
    Constructs a utility matrix for a patrolling game played with a single
    intruder and an arbitrary number of patrollers.

    Parameters
    ----------
    G_dict : dict of networkx.digraph.DiGraph
        A dictionary of graphs with <enum 'AgentType'> as keys and the graph
        represtations of the environment for the corresponding agent types as
        values.
    p_types : list of <enum 'AgentType'>
        The types of the patrolling agents.
    """

    U_blocks = []
    for i, p_type in enumerate(p_types):
        U_blocks.append(construct_two_player_util_mat(G_dict, p_type,
                                                      util_fun))

    U = np.vstack(U_blocks)

    return U
