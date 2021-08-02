import cvxpy as cp
import numpy as np

from .agents import AgentType
from .graph import generate_gridworld_vis_graph, generate_random_grid_with_holes
from .utility  import PvIUtilFun, construct_star_top_util_mat
from .simulation import simulate_star_top_patrolling
from .solvers import compute_ne_star_top, compute_minimax_strategies, make_stable

def run_benchmarks(graphs, p_types, p_intruder=0.3, n_sim=100, T=20):
    """
    Use run_benchmark on several graphs to collect simulation data.

    Parameters
    ----------
    graphs : list of networkx.digraph.DiGraph
        The graphs on which to run the benchmarks.
    p_types : list of <enum 'AgentType'>
        The agent types of the patrollers.
    p_intruder : float
        The probability of spawning a new intruder at each timestep during
        simulation (default is 0.3).
    n_sim : int
        The number of simulations to run on each graph.
    T : int
        The length of each simulation.

    Returns
    -------
    poly_simulations : list of dict
        Simulations using polymatrix patroller strategies. See the documentation
        for the simulate_star_top_patrolling function in patrolling/simulation.py
        for details.
    minimax_simulations : list of dict
        Simulations using independent minimax patroller strategies. See the
        documentation for the simulate_star_top_patrolling function in
        patrolling/simulation.py for details.
    rand_simulations : list of dict
        Simulations using random patroller strategies. See the documentation
        for the simulate_star_top_patrolling function in patrolling/simulation.py
        for details.
    """

    poly_simulations = []
    minimax_simulations = []
    rand_simulations = []

    for i, G in enumerate(graphs):
        print(f'Benchmarking graph {i + 1}/{len(graphs)}...')
        poly_simulations_g, minimax_simulations_g, rand_simulations_g = run_benchmark(G,
                                                                                      p_types,
                                                                                      p_intruder,
                                                                                      n_sim,
                                                                                      T)
        poly_simulations += poly_simulations_g
        minimax_simulations += minimax_simulations_g
        rand_simulations += rand_simulations_g

    return poly_simulations, minimax_simulations, rand_simulations

def run_benchmark(G, p_types, p_intruder=0.3, n_sim=100, T=20):
    """
    Collect simulation data using three solution concepts:
        1. Star topology patrolling game with polymatrix solver.
        2. Star topology patrolling game where patrollers play minimax
           strategies in a game involving only themselves and the intruder.
        3. Patrollers use random strategies.
    In all cases, the intruder optimizes its strategy after the patroller
    strategies are computed, to test the worst case performance.
    """

    vis_graphs = {AgentType.QUAD: generate_gridworld_vis_graph(G, 3),
                AgentType.GROUND: generate_gridworld_vis_graph(G, 1)}
    occ_utils = {AgentType.QUAD: dict([(v, 1) for v in G.nodes]),
                 AgentType.GROUND: dict((v, 1.0) for v in G.nodes)}
    reward_struct = {AgentType.QUAD: {'detects': 1, 'is_detected': 1},
                     AgentType.GROUND: {'detects': 1, 'is_detected': 1}}
    util_fun = PvIUtilFun(vis_graphs, occ_utils, reward_struct)
    U_star = construct_star_top_util_mat(G, p_types, util_fun)

    # Generate patrolling strategies
    x_list_poly, y_list, p = compute_ne_star_top(U_star, G, verbose=False, max_iters=20)
    x_list_minimax, _, _ = compute_minimax_strategies(U_star, G)
    x_list_rand = [np.ones(G.number_of_edges()) / G.number_of_edges() for _ in p_types]

    # Find the intruder best response for each solution
    y_list_opt_poly, p_opt_poly = optimize_intruder_response(U_star, x_list_poly, G)
    y_list_opt_minimax, p_opt_minimax = optimize_intruder_response(U_star, x_list_minimax, G)
    y_list_opt_rand, p_opt_rand = optimize_intruder_response(U_star, x_list_rand, G)

    # Run simulation
    poly_simulations = simulate_star_top_patrolling(x_list_poly, y_list_opt_poly, G, vis_graphs, p_types,
                                                p_opt_poly, p_intruder, n_sim, T)
    minimax_simulations = simulate_star_top_patrolling(x_list_minimax, y_list_opt_minimax, G, vis_graphs, p_types,
                                                   p_opt_minimax, p_intruder, n_sim, T)
    rand_simulations = simulate_star_top_patrolling(x_list_rand, y_list_opt_rand, G, vis_graphs, p_types,
                                                p_opt_rand, p_intruder, n_sim, T)

    return poly_simulations, minimax_simulations, rand_simulations

def compute_star_top_util(x_list, U_star, y_list, p):
    """
    Compute the total theoretical value for the patrolling team in
    patrolling game with a star topology (many patrollers and a single
    intruder).

    Parameters
    ----------
    x_list : list of numpy.ndarray
        The list of patroller strategies. Must be the same length as U_star.
    U_star : list of numpy.ndarray
        A list of utility matrices, one for each patroller.
    y_list : list of numpy.ndarray
        The list of intruder strategies, one for each robot type as specified
        in the AgentType enum.
    p : np.ndarray
        The probability distribution over the robot types specified in the
        AgentType enum for the intruder.

    Returns
    -------
    The total value of the game for the patroller team.
    """
    x = np.concatenate(x_list)
    U = np.vstack(U_star)
    y = np.concatenate([y_list[0] * p[0], y_list[1] * p[1]])

    return x @ U @ y

def compute_stats(simulations):
    """
    Compute the intruder detection rates, patroller detection rates, and average
    time-to-detection for each simulation.

    Parameters
    ----------
    simulation : list of dict
        A list of simulation. See the documentation for the
        simulate_star_top_patrolling in patrolling/simulate.py for details.

    Returns
    -------
    int_det_rates :
    pat_det_rates :
    det_times :
    """
    int_det_rates = calc_int_detection_rates(simulations)
    pat_det_rates = calc_pat_detection_rates(simulations)
    det_times = calc_avg_time_to_detection(simulations)

    return int_det_rates, pat_det_rates, det_times

def calc_int_det_rates(simulations):
    sim_avgs = []

    for simulation in simulations:
        n_detections = 0
        if len(simulation['intruders']) == 0:
            continue
        for intruder in simulation['intruders']:
            n_detections += intruder['detection_time'] is not None
        sim_avgs.append(n_detections / len(simulation['intruders']))

    return sim_avgs

def calc_pat_det_rates(simulations):
    sim_detections = []

    for simulation in simulations:
        n_detections = 0
        if len(simulation['intruders']) == 0:
            continue
        for patroller in simulation['patrollers']:
            n_detections += sum(patroller['is_detected'])

        sim_detections.append(n_detections)

    return sim_detections

def calc_det_times(simulations):
    detection_times = []

    for simulation in simulations:
        for intruder in simulation['intruders']:
            if intruder['detection_time']:
                detection_times.append(intruder['detection_time'] - intruder['spawn_time'])

    return detection_times

def draw_boxplots(x, ax, box_labels=None, title=None, xlabel=None, ylabel=None,
                  showfliers=True):
    ax.boxplot(x, showfliers=False)

    if box_labels is not None:
        ax.set_xticklabels(box_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def draw_bar_chart(data, ax, errs=None, bar_width=0.5, bar_padding=0.3, xlabel=None, ylabel=None,
                   title=None, bar_labels=None):
    n_bars = len(data)
    x_locs = [bar_padding + bar_width / 2 + i * (bar_width + bar_padding) \
              for i in range(n_bars)]
    if errs is None:
        _ = ax.bar(x_locs, data)
    else:
        _ = ax.bar(x_locs, data, yerr=errs)

    ax.set_xticks(x_locs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if bar_labels is not None:
        ax.set_xticklabels(bar_labels)

def draw_grouped_bar_chart(ax, *data, bar_width=0.5, group_padding=0.3, group_labels=None, x_label=None,
                           y_label=None, title=None, legend=False):
    n_bars = len(data)
    n_groups = len(data[0])
    group_width = n_bars * bar_width
    x_locs = [group_padding + group_width / 2 + i * (group_width + group_padding) \
             for i in range(n_groups)]
    left_endpoints = [group_padding + i * (group_padding + group_width) for i in range(n_groups)]

    for i in range(n_bars):
        bar_inds = [left_endpoint + i * bar_width + bar_width / 2 for left_endpoint in left_endpoints]
        _ = ax.bar(bar_inds, data[i], bar_width)

    ax.set_title(title)
    ax.set_xticks(x_locs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def optimize_intruder_response(U_star, x_list, G, use_movement_constraints=True, **solver_args):
    U = np.vstack(U_star)
    y_joint = cp.Variable(U.shape[1])
    constraints = [(cp.sum(y_joint) == 1), (y_joint >= 0)]
    y_vars = []

    for agent_type in AgentType:
        i = agent_type * G.number_of_edges()
        j = i + G.number_of_edges()
        y_vars.append(y_joint[i:j])
        if use_movement_constraints:
            for v in G.nodes:
                incoming_edges = G.in_edges(v, data=True)
                outgoing_edges = G.out_edges(v, data=True)
                in_eids = [e[-1]['eid'] for e in incoming_edges]
                out_eids = [e[-1]['eid'] for e in outgoing_edges]
                in_sum = cp.sum(y_vars[-1][in_eids])
                out_sum = cp.sum(y_vars[-1][out_eids])
                constraints.append((in_sum == out_sum))

    obj = cp.Maximize(-np.concatenate(x_list) @ U @ y_joint)
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_args)

    p = make_stable([np.sum(y_var.value) for y_var in y_vars])
    y_list = [make_stable(y_var.value) for y_var in y_vars]

    return y_list, p
