import numpy as np

from .agents import AgentType

def simulate_star_top_patrolling(x_list, y, G, vis_graphs, p_types, intruder_type_dist,
                                 p_intruder, n_simulations, T):
    """
    Parameters
    ----------
    Simulate patrolling using strategies for a game played between many an
    arbitrary number of patrollers and a single intruder. Collects and returns
    statistics about patrolling performance.

    Parameters
    ----------
    x_list : list of numpy.ndarray
        A list of patroller strategies, one for each patroller. Strategies are
        valid probability distributions over the edges in G.
    y : list of np.ndarray
        A list of intruder strategies, one for each type in the AgentType enum.
    G : networkx.digraph.DiGraph
        The graph representation of the environment.
    vis_graphs : dict of networkx.digraph.DiGraph
        A dictionary of graphs where the keys are types from AgentType and the
        values are graphs. Each graph specifies the vertices that are visible
        to that type from any vertex in G.
    p_types : list of <enum 'AgentType'>
        The type of each patroller agent.
    intruder_type_dist : np.ndarray
        A probability distribution over the agent types in the AgentType enum.
        Specifies the probability that the intruder will appear as each agent
        type.
    p_intruder : float
        The probability that a new intruder spawns at each timestep.
    n_simulations : int
        The number of simulations to run.
    T : int
        The length of each simulation.

    Returns
    -------
    simulations : list of dict
        A list of simulations, each of which is a dict. Each simulation dict
        contains the length of the simulation and data about the patrollers and
        intruders in that simulation. For each patroller and intruder, a complete
        history is recorded. For intruders, contains the spawn time, detection
        time, agent type, and type of patroller by which it was detected, if
        applicable. For patrollers, contains the agent type and the times at
        which it is detected by intruders.
    """

    simulations = []

    for n in range(n_simulations):
        patrollers = []
        intruders = []
        for i, p_type in enumerate(p_types):
            patroller = {'type': p_type,
                        'is_detected': [],
                        'history': []}
            patrollers.append(patroller)
        for t in range(T):
            print(f'[{n + 1}/{n_simulations}][{t + 1}/{T}]', end='\r')
            r = np.random.uniform(0, 1)
            if r <= p_intruder:
                new_intruder_type = AgentType(np.random.choice(AgentType, p=intruder_type_dist))
                new_intruder = {'type': new_intruder_type,
                                'spawn_time': t,
                                'detection_time': None,
                                'history': [None] * t}
                intruders.append(new_intruder)

            for i, patroller in enumerate(patrollers):
                v_old = patroller['history'][-1] if len(patroller['history']) else None
                v_new = get_next_pos(x_list[i], G, v=v_old)
                patroller['history'].append(v_new)
                patroller['is_detected'].append(False)
            for intruder in intruders:
                if intruder['detection_time'] is None:
                    v_old = intruder['history'][-1] if len(intruder['history']) else None
                    v_new = get_next_pos(y[intruder['type']], G, v=v_old)
                    intruder['history'].append(v_new)
                else:
                    intruder['history'].append(None)

            for patroller in patrollers:
                for intruder in intruders:
                    if intruder['detection_time'] is None:
                        p_type = patroller['type']
                        i_type = intruder['type']
                        v_p = patroller['history'][-1]
                        v_i = intruder['history'][-1]
                        if v_p in vis_graphs[i_type].neighbors(v_i):
                            patroller['is_detected'][-1] = True
                        if v_i in vis_graphs[p_type].neighbors(v_p):
                            intruder['detection_time'] = t
                            intruder['detected_by'] = p_type


        simulation = {'T': T,
                      'patrollers': patrollers,
                      'intruders': intruders}
        simulations.append(simulation)

    return simulations

def get_next_pos(x, G, v=None):
    """
    Compute the next position of an agent given their strategy, the graph of the
    environment, and their current vertex. If a vertex is not specified, will
    compute a vertex based on the provided strategy.

    Parameters
    ----------
        x : np.ndarray
            The agents strategy. Strategies are valid probability distributions
            over the edges in G.
        G : networkx.digraph.DiGraph
            The graph representation of the envrionment.
        v : object (optional)
            The agent's current vertex in G.

    Returns
    -------
    v_new : object
        The agents next vertex in G
    """
    
    out_edges = list(G.out_edges(v, data=True))
    out_eids = [e[-1]['eid'] for e in out_edges]
    move_probs = x[out_eids]
    move_dist = move_probs / np.sum(move_probs)
    i = np.random.choice(np.arange(len(out_edges)), p=move_dist)
    next_edge = out_edges[i]
    v_new = next_edge[1]

    return v_new
