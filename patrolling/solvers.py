import cvxpy as cp
import numpy as np

from .agents import AgentType

def compute_two_player_minimax(U, G, use_movement_constraints=True,
                               **solver_args):
    """
    Computes the minimax strategy profile for a patrolling game played between
    one patroller and one intruder.

    Parameters
    ----------
    U : numpy.ndarray
        The utility matrix.
    G : networkx.digraph.DiGraph
        The graph representation of the environment
    use_movement_constraints : bool
        Whether to enforce the constraint on the computed policies that the
        probability of entering a node is consistent with the probability of
        leaving it. This ensures that the marginal probability of being at any
        given vertex is a consistent probability distribution.
    solver_args : **kwargs
        Additional arguments to the CVXPY solver.

    Returns
    -------
    x : numpy.ndarray
        The patroller's minimax strategy.
    y_list : list of numpy.ndarray
        The list of intruder strategies, one for each robot type as specified
        in the AgentType enum.
    p : np.ndarray
        The optimal probability distribution over the robot types specified
        in the AgentType enum for the intruder.
    """

    z = cp.Variable()
    x_var = cp.Variable(U.shape[0])
    pat_constraints = [(cp.sum(x_var) == 1), (x_var >= 0), (z <= x_var @ U)]

    if use_movement_constraints:
        for v in G.nodes:
            incoming_edges = G.in_edges(v, data=True)
            outgoing_edges = G.out_edges(v, data=True)
            in_eids = [e[-1]['eid'] for e in incoming_edges]
            out_eids = [e[-1]['eid'] for e in outgoing_edges]
            in_sum = cp.sum(x_var[in_eids])
            out_sum = cp.sum(x_var[out_eids])
            pat_constraints.append((in_sum == out_sum))

    pat_obj = cp.Maximize(z)
    pat_prob = cp.Problem(pat_obj, pat_constraints)
    pat_prob.solve(**solver_args)

    w = cp.Variable()
    y_joint = cp.Variable(U.shape[1])
    int_constraints = [(cp.sum(y_joint) == 1), (y_joint >= 0), (w >= U @ y_joint)]
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
                int_constraints.append((in_sum == out_sum))

    int_obj = cp.Minimize(w)
    int_prob = cp.Problem(int_obj, int_constraints)
    int_prob.solve(**solver_args)

    x = make_stable(x_var.value)
    p = make_stable([np.sum(y_var.value) for y_var in y_vars])
    y_list = [make_stable(y_var.value) for y_var in y_vars]

    return x, y_list, p

def compute_ne_star_top(U_star, G, use_movement_constraints=True, **solver_args):
    """
    Compute an equilibrium strategy profile for a patrolling game played between
    one intruder and an arbitrary number of patrollers.

    Parameters
    ----------
    U_star : list of numpy.ndarray
        A list of utility matrices, one for each patroller.
    G : networkx.digraph.DiGraph
        The graph representation of the environment
    use_movement_constraints : bool
        Whether to enforce the constraint on the computed policies that the
        probability of entering a node is consistent with the probability of
        leaving it. This ensures that the marginal probability of being at any
        given vertex is a consistent probability distribution.
    solver_args : **kwargs
        Additional arguments to the CVXPY solver.

    Returns
    -------
        x_list : list of numpy.ndarray
            The list of equilibrium patroller strategies. Will be the same
            length as U_star.
        y_list : list of numpy.ndarray
            The list of  equilibrium intruder strategies, one for each robot
            type as specified in the AgentType enum.
        p : np.ndarray
            The equilibrium probability distribution over the robot types specified
            in the AgentType enum for the intruder.
    """

    U = np.vstack(U_star)
    z = cp.Variable(len(U_star))
    z_vec = cp.Variable(len(U))
    w = cp.Variable()
    x_cat = cp.Variable(len(U))
    y_joint = cp.Variable(U.shape[1])
    constraints = []
    x_vars = []
    y_vars = []

    i = 0
    for k, U_i in enumerate(U_star):
        j = i + U_i.shape[0]
        x_vars.append(x_cat[i:j])
        constraints.append((z_vec[i:j] == z[k]))
        i = j

    for i in AgentType:
        j = i * G.number_of_edges()
        k = j + G.number_of_edges()
        y_vars.append(y_joint[j:k])

    if use_movement_constraints:
        for v in G.nodes:
            incoming_edges = G.in_edges(v, data=True)
            outgoing_edges = G.out_edges(v, data=True)
            in_eids = [e[-1]['eid'] for e in incoming_edges]
            out_eids = [e[-1]['eid'] for e in outgoing_edges]
            for x_var in x_vars:
                in_sum = cp.sum(x_var[in_eids])
                out_sum = cp.sum(x_var[out_eids])
                constraints.append((in_sum == out_sum))
            for i in AgentType:
                j = i * G.number_of_edges()
                k = j + G.number_of_edges()
                in_sum = cp.sum(y_joint[j:k][in_eids])
                out_sum = cp.sum(y_joint[j:k][out_eids])
                constraints.append((in_sum == out_sum))

    constraints += [(y_joint >= 0), (cp.sum(y_joint) == 1), (x_cat >= 0)] \
                    + [(cp.sum(x_var) == 1) for x_var in x_vars]
    constraints += [(w >= -x_cat @ U), (z_vec >= U @ y_joint)]

    obj = cp.Minimize(cp.sum(z) + w)
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_args)

    x_list = [make_stable(x_var.value) for x_var in x_vars]
    p = make_stable([np.sum(y_var.value) for y_var in y_vars])
    y_list = [make_stable(y_var.value) for y_var in y_vars]

    return x_list, y_list, p

def compute_minimax_star_top(U_star, G, use_movement_constraints=True,
                             **solver_args):
    """
    Computes the minimax strategy for each player in a patrolling game where
    an arbitrary number of patrollers play against a single intruder.

    Parameters
    ----------
    U_star : list of numpy.ndarray
        A list of utility matrices, one for each patroller.
    G : networkx.digraph.DiGraph
        The graph representation of the environment
    use_movement_constraints : bool
        Whether to enforce the constraint on the computed policies that the
        probability of entering a node is consistent with the probability of
        leaving it. This ensures that the marginal probability of being at any
        given vertex is a consistent probability distribution.
    solver_args : **kwargs
        Additional arguments to the CVXPY solver.

    Returns
    -------
        x_list : list of numpy.ndarray
            The list of patroller strategies. Will be the same length as U_star.
        y_list : list of numpy.ndarray
            The list of intruder strategies, one for each robot type as specified
            in the AgentType enum.
        p : np.ndarray
            The optimal probability distribution over the robot types specified
            in the AgentType enum for the intruder.
    """

    x_list = []

    for U in U_star:
        x, _, _ = compute_two_player_minimax(U, G, **solver_args,
                                             use_movement_constraints=use_movement_constraints)
        x_list.append(x)

    _, y_list, p = compute_ne_star_top(U_star, G, **solver_args,
                                       use_movement_constraints=use_movement_constraints)

    return x_list, y_list, p

def make_stable(x):
    """
    Turns a vector into a valid probability distribution by removing negative
    entries with small absolute value and normalize to sum to 1.

    Parameters
    ----------
    x : numpy.ndarray
        The vector to be stabilized

    Returns
    -------
    x_copy : numpy.ndarray
        A numerically stable copy of x.
    """

    x_copy = np.copy(x)
    x_copy[np.where(np.isclose(x_copy, 0))] = 0
    if np.sum(x_copy):
        x_copy /= np.sum(x_copy)

    return x_copy
