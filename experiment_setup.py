import numpy as np
from backend.dynamic_vehicles import CircularSingleIntegrator, \
    CircularDoubleIntegrator                                # Circular Single/Double integrator vehicles
from backend.safe_halfspaces import DRCVaRHalfspace, \
    CVaRHalfspace, MeanHalfspace                            # safe halfspace
from backend.safety_filters import MPCFilter, MPCFilterWithSlack    # MPC Filter
from backend.ref_traj_generation import MPCReferenceTrajectory      # ref trajectory generator


def drone_exp_setup(exp_type, metric, filter_slack):
    """
    Drone experiment setup
    :param exp_type: ['ego_intersect', 'ego_headon', 'ego_overtaking', 'ego_and_3_vehicles']
    :param metric: ['mean', 'cvar', 'drcvar']
    :param filter_slack: True/False
    :return:
    """
    ENV_LIM = 5

    # optimization problem solver
    SOLVER = 'ECOS'

    # Experiment settings
    t0 = 0  # start time
    dt = 0.2  # discrete time step
    rad = 0.3  # radius of vehicles

    # MPC filter settings
    horizon = 10  # MPC horizon
    mpc_Q, mpc_QT, mpc_R = 2., 5., 1.
    filter_solver = SOLVER

    # reference Trajectory settings
    ego_ref_Q, ego_ref_QT, ego_ref_R = 1., 3., 1.
    traj_gen_solver = SOLVER

    # safe halfspace params
    num_samp = 20  # number of samples
    delta = 0.1  # loss bound
    alpha = 0.2  # alpha-worst cases (when applicable)
    eps = 0.05  # wasserstein ball (when applicable)

    # General optimization problem settings
    accel_lim_x = 100
    accel_lim_y = 100
    measurement_Ax_leq_b = {'A': np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]),
                            'b': np.array([ENV_LIM, ENV_LIM, ENV_LIM, ENV_LIM])}
    control_Ax_leq_b = {'A': np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]),
                        'b': np.array([accel_lim_x, accel_lim_x, accel_lim_y, accel_lim_y])}

    noise_std_dev = np.array([0.1, 0.1])  # noise std dev to sample data from

    # save data
    fig_name = metric + '_' + exp_type
    if filter_slack:
        fig_name += '_with_slack'
    EGO_COLOR, EGO_ALPHA = 'tab:blue', 0.2
    OBST_COLORS, OBST_ALPHA = ['tab:red', 'tab:orange', 'tab:pink'], 0.2

    # # #
    if exp_type == 'ego_overtaking':
        total_time = 3  # total time in seconds
        sim_steps = int(np.ceil(total_time / dt))

        ego_radius = rad
        ego_init_pos = np.array([-ENV_LIM + ego_radius, 0])
        ego_init_vel = np.array([1.5, 0])
        ego_init_state = np.hstack([ego_init_pos, ego_init_vel])
        ego_goal_state = np.array([ENV_LIM - ego_radius, 0, 0, 0])
        ego_veh = CircularDoubleIntegrator(ego_init_state, dt, t0, ego_radius)

        obst_radii = [rad]
        obst_init_positions = [np.array([-2, -0.05])]
        obst_init_states = obst_init_positions
        obst_ref_vels = [np.array([1, 0])]
    elif exp_type == 'ego_headon':
        total_time = 3  # total time in seconds
        sim_steps = int(np.ceil(total_time / dt))

        ego_radius = rad
        ego_init_pos = np.array([-ENV_LIM + ego_radius, 0])
        ego_init_vel = np.array([1.5, 0])
        ego_init_state = np.hstack([ego_init_pos, ego_init_vel])
        ego_goal_state = np.array([ENV_LIM - ego_radius, 0, 0, 0])
        ego_veh = CircularDoubleIntegrator(ego_init_state, dt, t0, ego_radius)

        obst_radii = [rad]
        obst_init_positions = [np.array([2, -0.01])]
        obst_init_states = obst_init_positions
        obst_ref_vels = [np.array([-1, 0])]
    elif exp_type == 'ego_intersect':
        total_time = 3  # total time in seconds
        sim_steps = int(np.ceil(total_time / dt))

        # ego vehicle setup
        ego_radius = rad
        ego_init_pos = np.array([-3.5, 1])
        ego_init_vel = np.array([1.5, 0])
        ego_init_state = np.hstack([ego_init_pos, ego_init_vel])
        ego_goal_state = np.array([1, -3, 0, 0])
        ego_veh = CircularDoubleIntegrator(ego_init_state, dt, t0, ego_radius)

        # obstacle vehicle setup
        obst_radii = [rad]
        obst_init_positions = [np.array([-2.5, -1])]
        obst_init_states = obst_init_positions
        obst_ref_vels = [np.array([1.5, 0])]
    elif exp_type == 'ego_and_3_vehicles':
        total_time = 5  # total time in seconds
        sim_steps = int(np.ceil(total_time / dt))

        # ego vehicle setup
        ego_radius = rad
        ego_init_pos = np.array([-ENV_LIM + ego_radius, -1.])
        ego_init_vel = np.array([1.5, 0])
        ego_init_state = np.hstack([ego_init_pos, ego_init_vel])
        ego_goal_state = np.array([ENV_LIM - ego_radius, 0, 0, 0])
        ego_veh = CircularDoubleIntegrator(ego_init_state, dt, t0, ego_radius)

        # obstacle vehicle setup
        obst_radii = [rad, rad, rad]
        obst_init_positions = [np.array([-1.1, 1.01]), np.array([-2, -1.01]), np.array([-1, -2.01])]
        obst_init_states = obst_init_positions
        obst_ref_vels = [np.array([0.7, 0]), np.array([1, 0]), np.array([0.7, 0])]
    else:
        raise NotImplementedError('Experiment type not supported')

    obst_vehs = [CircularSingleIntegrator(obst_init_state, dt, t0, obst_radius)
                     for obst_init_state, obst_radius in zip(obst_init_states, obst_radii)]
    num_obst = len(obst_vehs)

    # Safe halfspace
    if metric == 'drcvar':
        safe_hs = DRCVaRHalfspace(alpha, eps, delta, num_samp, solver=SOLVER)
    elif metric == 'cvar':
        safe_hs = CVaRHalfspace(alpha, delta, num_samp, loss_type='continuous', solver=SOLVER)
    elif metric == 'mean':
        safe_hs = MeanHalfspace()
    else:
        raise NotImplementedError('Invalid risk metric')
    # solve for the first time to eliminate CVXPY time overhead
    safe_hs.set_opt_pb_params(np.zeros([2, ]), np.zeros([2, num_samp]), [0])
    safe_hs.solve_opt_pb()

    # Reference trajectory generator
    ego_traj_gen = MPCReferenceTrajectory(horizon, ego_veh.A, ego_veh.B, Q=ego_ref_Q, QT=ego_ref_QT, R=ego_ref_R,
                                          control_Ax_leq_b=control_Ax_leq_b)

    # MPC Filter
    Q = np.eye(ego_veh.n) * mpc_Q
    QT = np.eye(ego_veh.n) * mpc_QT
    R = np.eye(ego_veh.m) * mpc_R
    if filter_slack:
        mpc_filter = MPCFilterWithSlack(num_obst, ego_veh, horizon, Q, QT, R, measurement_Ax_leq_b=measurement_Ax_leq_b,
                                        control_Ax_leq_b=control_Ax_leq_b)
    else:
        mpc_filter = MPCFilter(num_obst, ego_veh, horizon, Q, QT, R, measurement_Ax_leq_b=measurement_Ax_leq_b,
                               control_Ax_leq_b=control_Ax_leq_b)

    return (ENV_LIM,
            dt, horizon,
            num_samp, sim_steps,
            num_obst,
            ego_veh, EGO_COLOR, EGO_ALPHA,
            ego_traj_gen, ego_goal_state,
            obst_vehs, OBST_COLORS, OBST_ALPHA,
            obst_init_positions, obst_ref_vels,
            traj_gen_solver, filter_solver,
            noise_std_dev,
            safe_hs, mpc_filter,
            fig_name)
