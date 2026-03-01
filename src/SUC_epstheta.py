import numpy as np
import pandapower as pp
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc

import os
import gurobipy as gp
from gurobipy import GRB
from WT_error_gen import WT_sce_gen
from scipy.linalg import norm
import time
# import joblib
from joblib import Parallel, delayed

def solve_VaR(N_WDR, theta, epsilon_p, random_var_scenarios, b_p):
    '''
    solve the VaR by bilinear programming for Bonferonni approximation
    '''
    prob = gp.Model('VaR')
    alpha = prob.addMVar(N_WDR, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    beta = prob.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    m = prob.addMVar(N_WDR, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    eta = prob.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    prob.addConstr(theta * beta + 1/N_WDR * gp.quicksum(alpha) <= epsilon_p)
    prob.addConstr(alpha >= 1-m*(eta + random_var_scenarios @ b_p))
    prob.addConstr(beta >= m * norm(b_p, ord=2))
    prob.addConstr(alpha >= 0)
    prob.addConstr(m >= 0)

    prob.setObjective(eta, GRB.MINIMIZE)
    # no output
    prob.setParam('OutputFlag', 0)
    # set higher precision
    prob.setParam('IntFeasTol', 1e-9)
    prob.setParam('FeasibilityTol', 1e-9)
    prob.setParam('OptimalityTol', 1e-9)
    prob.optimize()
    return prob.objVal

def piecewise_linear(c_quadr_all, c_linear_all, pmax_all, pmin=0, n=6):
    # fit the piecewise linear function coefficients for a quadratic cost function with the form of c_quadr * x^2 + c_linear * x
    # generate samples from pmin to pmax
    x_all = np.linspace(pmin, pmax_all, n)
    y_all = c_quadr_all * x_all ** 2 + c_linear_all * x_all
    coeff_list = []
    intercept_list = []
    for i in range(n-1):
        coeff = (y_all[i+1] - y_all[i]) / (x_all[i+1] - x_all[i])
        intercept = y_all[i] - coeff * x_all[i]
        coeff_list.append(coeff)
        intercept_list.append(intercept)
    return coeff_list, intercept_list

def solve_SUC(T, num_gen, num_WT, num_branch, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, gen_UR_max, gen_DR_max, UR_extra_total, DR_extra_total,
              gen_ramp_up, gen_ramp_down, UT, DT, WT_pred, WT_error_scenarios_train, bigM,
              P_line_limit, gen_bus_list, WT_bus_list, N_WDR, epsilon, theta, MIPGap, rng, bearable_cost_prop,
              gen_cost, urc, drc, su, sd, gen_cost_quadra, gen_cost_fixed, gurobi_seed, method="proposed",
              njobs = 1, log_file_name = None, quadra_cost = False, thread = 16):
    '''
    solve the stochastic unit commitment problem
    '''

    # set small PTDF to zero to avoid numerical issues
    PTDF[np.abs(PTDF) < 1e-5] = 0

    t_start = time.time()
    prob = gp.Model('SUC')
    gen_power_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the scheduled power output of thermal generators
    gen_UR_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the upward reserve of thermal generators
    gen_DR_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the downward reserve of thermal generators
    gen_v_all = prob.addMVar((T, num_gen), vtype=GRB.BINARY) # the binary variable indicating the on/off status of thermal generators

    WT_schedule_all = prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the scheduled power output of wind turbines
    WT_curtail_all = prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the curtailment of wind turbines

    for t in range(T):
        # ---------------------------------------------------
        # power balance constraint at time step t
        prob.addConstr(gen_power_all[t, :].sum() + WT_schedule_all[t, :].sum() == load_bus_all[t, :].sum())

        # ---------------------------------------------------
        # constraints for thermal generators at time step t
        # pmax constraint
        prob.addConstr(gen_power_all[t, :] + gen_UR_all[t, :] <= gen_v_all[t, :] * gen_cap_individual)
        # pmin constraint
        prob.addConstr(gen_power_all[t, :] - gen_DR_all[t, :] >= gen_v_all[t, :] * gen_pmin_individual)
        if t > 0:
            # ramp constraint
            # upward
            prob.addConstr(gen_power_all[t, :] - gen_power_all[t-1, :] <= gen_ramp_up + (2-gen_v_all[t-1, :]-gen_v_all[t, :]) * gen_cap_individual)
            # downward
            prob.addConstr(gen_power_all[t-1, :] - gen_power_all[t, :] <= gen_ramp_down + (2-gen_v_all[t-1, :]-gen_v_all[t, :]) * gen_cap_individual)

        # reserve constraint
        prob.addConstr(gen_UR_all[t, :] <= gen_UR_max)
        prob.addConstr(0 <= gen_UR_all[t, :])
        prob.addConstr(gen_DR_all[t, :] <= gen_DR_max)
        prob.addConstr(0 <= gen_DR_all[t, :])

        # minimum up and down time constraint
        if t > 0:
            if t<= (T-UT):
                prob.addConstr(gen_v_all[t:t + UT, :].sum(axis=0) >= UT * (gen_v_all[t, :] - gen_v_all[t - 1, :]))
            else:
                prob.addConstr((gen_v_all[t:, :] - (gen_v_all[t, :] - gen_v_all[t - 1, :])).sum(axis=0) >= 0)

            if t<= (T-DT):
                prob.addConstr((1-gen_v_all[t:t + DT, :]).sum(axis=0) >= DT * (gen_v_all[t-1, :] - gen_v_all[t, :]))
            else:
                prob.addConstr((1-gen_v_all[t:, :] - (gen_v_all[t-1, :] - gen_v_all[t, :])).sum(axis=0) >= 0)

        # ---------------------------------------------------
        # constraints for wind turbines at time step t
        # relationship between WT_schedule and WT_curtail
        prob.addConstr(WT_schedule_all[t, :] + WT_curtail_all[t, :] == WT_pred[t, :])
        # curtailment smaller than the forecasted WT capacity
        prob.addConstr(WT_curtail_all[t, :] <= WT_pred[t, :])
        # curtailment non-negative
        prob.addConstr(0 <= WT_curtail_all[t, :])

    # ---------------------------------------------------
    # joint chance constraint
    # need to write the joint chance constraint as the form of Ax <= B\xi + d
    # where A \in R^{P*L}, x \in R^{L}, B \in R^{P*D}, \xi is the random vector \in R^{D}, d \in R^{P}
    # x is composed of [gen_UR_all_t=1, gen_DR_all_t=1, gen_power_all_t=1, WT_schedule_all_t=1,
    # gen_UR_all_t=2, gen_DR_all_t=2, gen_power_all_t=2, WT_schedule_all_t=2, ...]
    # \xi is ordered as [ran_vars_t=1, ran_vars_t=2, ..., ran_vars_t=T]
    # there are P = 2*T (system reserve) + 2*num_branch*T (branch flow constraints) individual constraints in the JCC
    P = 2*T + 2*num_branch*T
    L = gen_UR_all.size + gen_DR_all.size + gen_power_all.size + WT_curtail_all.size
    D = num_WT*T

    A = np.zeros((P, L))
    B = np.zeros((P, D))
    d = np.zeros(P)
    x = prob.addMVar(L, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    len_x_t = round(L / T) # the number of variables at each time step
    # link x to other variables
    for t in range(T):
        prob.addConstr(gen_UR_all[t, :] == x[len_x_t*t:len_x_t*t + num_gen])
        prob.addConstr(gen_DR_all[t, :] == x[len_x_t*t + num_gen:len_x_t*t + 2*num_gen])
        prob.addConstr(gen_power_all[t, :] == x[len_x_t*t + 2*num_gen:len_x_t*t + 3*num_gen])
        prob.addConstr(WT_schedule_all[t, :] == x[len_x_t*t + 3*num_gen:len_x_t*t + 3*num_gen + num_WT])

    len_ran_var_t = round(D / T) # the number of random variables at each time step
    num_constr_t = round(P / T) # the number of constraints at each time step
    for t in range(T):
        # system reserve constraints upward
        A[t*num_constr_t:t*num_constr_t + 1, len_x_t*t:len_x_t*t + num_gen] = -1
        B[t*num_constr_t:t*num_constr_t + 1, t*len_ran_var_t:(t+1)*len_ran_var_t] = 1
        d[t*num_constr_t:t*num_constr_t + 1] = -UR_extra_total

        # system reserve constraints downward
        A[t*num_constr_t + 1:t*num_constr_t + 2, len_x_t*t+num_gen:len_x_t*t+2*num_gen] = -1
        B[t*num_constr_t + 1:t*num_constr_t + 2, t*len_ran_var_t:(t+1)*len_ran_var_t] = -1
        d[t*num_constr_t + 1:t*num_constr_t + 2] = -DR_extra_total

        # transmission line constraints positive flow
        A[t*num_constr_t + 2:t*num_constr_t + 2 + num_branch, len_x_t*t+2*num_gen:len_x_t*t + 3*num_gen] = PTDF[:, gen_bus_list]
        A[t*num_constr_t + 2:t*num_constr_t + 2 + num_branch, len_x_t*t + 3*num_gen:len_x_t*t + 3*num_gen + num_WT] = PTDF[:, WT_bus_list]
        B[t*num_constr_t + 2:t*num_constr_t + 2 + num_branch, t*len_ran_var_t:(t+1)*len_ran_var_t] = -PTDF[:, WT_bus_list]
        d[t*num_constr_t + 2:t*num_constr_t + 2 + num_branch] = P_line_limit + PTDF @ load_bus_all[t]

        # transmission line constraints negative flow
        A[t*num_constr_t + 2 + num_branch:t*num_constr_t + 2 + 2*num_branch, len_x_t*t+2*num_gen:len_x_t*t + 3*num_gen] = -PTDF[:, gen_bus_list]
        A[t*num_constr_t + 2 + num_branch:t*num_constr_t + 2 + 2*num_branch, len_x_t*t + 3*num_gen:len_x_t*t + 3*num_gen + num_WT] = -PTDF[:, WT_bus_list]
        B[t*num_constr_t + 2 + num_branch:t*num_constr_t + 2 + 2*num_branch, t*len_ran_var_t:(t+1)*len_ran_var_t] = PTDF[:, WT_bus_list]
        d[t*num_constr_t + 2 + num_branch:t*num_constr_t + 2 + 2*num_branch] = P_line_limit - PTDF @ load_bus_all[t]

    #----------- process A, B, d -------
    # there are rows in B where all elements are zero because of the zero PTDF. We remove these rows
    # from the JCC and add them to the deterministic constraints
    zero_rows = np.where(np.all(B == 0, axis=1))[0]
    prob.addConstr(A[zero_rows] @ x <= d[zero_rows])
    # remove these rows from the JCC
    A = np.delete(A, zero_rows, axis=0)
    B = np.delete(B, zero_rows, axis=0)
    d = np.delete(d, zero_rows, axis=0)
    # P is updated
    P = A.shape[0]

    # prepare \xi
    random_var_scenario_index = rng.choice(WT_error_scenarios_train.shape[0], N_WDR, replace=False)
    random_var_scenarios = WT_error_scenarios_train[random_var_scenario_index, :, :].reshape(N_WDR, -1)

    # now we can apply the reformulation of the joint chance constraint, use SFLA for all methods because this is a starting point
    # define ancillary variables
    s = prob.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    r = prob.addMVar(N_WDR, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    prob.addConstr(s >= 0)
    prob.addConstr(r >= 0)
    prob.addConstr(epsilon * N_WDR * s - gp.quicksum(r) >= theta * N_WDR, name='epsNs')
    k = np.floor(N_WDR * epsilon).astype(int)

    b_xi_set_all = []
    q_all = []
    q_all_extend = [] # only used for the exact WDRJCC
    N_p_all = []
    N_p_all_for_r = []
    A_big = []
    B_norm_big = []
    d_big = []

    effective_p = 0
    for p in range(P):
        b_xi_set = random_var_scenarios @ B[p]  # the set of b^T @ \xi for each i in N_WDR
        q_p = np.sort(b_xi_set)[k]
        N_p = np.where(b_xi_set < q_p)[0]

        q_all.append(q_p) # q for each p. This is not skipped in the loop as the corresponding constraint is based on non-skipped parameters
        if len(N_p) == 0:
            continue
        # store these parameters so we can add them to the constraints at once
        b_xi_set_all.append(b_xi_set)
        N_p_all.append(N_p + effective_p*N_WDR)
        N_p_all_for_r.append(N_p)
        q_all_extend.append(np.tile(q_p, len(N_p)))

        # those matrices need to be "expanded"/duplicated so we can add all constraints at once
        A_big.append(np.tile(A[p:p+1], [len(N_p), 1]))
        B_norm_big.append(np.tile(norm(B[p], ord=2), len(N_p)))
        d_big.append(np.tile(d[p], len(N_p)))
        effective_p += 1

    q_all = np.array(q_all)
    if len(b_xi_set_all) > 0:
        # ==0 will only happen for the proposed method
        # for the proposed method, if the set N_p is empty, we do not need to add the constraint
        b_xi_set_all = np.hstack(b_xi_set_all)
        q_all_extend = np.hstack(q_all_extend)
        N_p_all = np.hstack(N_p_all)
        N_p_all_for_r = np.hstack(N_p_all_for_r)
        kappa = np.ones(N_WDR)

        kappa_big = np.tile(kappa, P)
        A_big = np.vstack(A_big)
        B_norm_big = np.hstack(B_norm_big)
        d_big = np.hstack(d_big)

        prob.addConstr(kappa_big[N_p_all] * (b_xi_set_all[N_p_all] + d_big - A_big @ x) / B_norm_big >= s - r[N_p_all_for_r], name="N_p")
        # prob.addConstr(kappa_big[N_p_all] * (b_xi_set_all[N_p_all] + d_big - A_big @ x) / B_norm_big >= 0)

        prob.addConstr((q_all + d - A @ x) / norm(B, ord=2, axis=1) >= s, name="q_all")

    # ---------------------------------------------------
    # Define the cost (objective) function
    # the unit commitment cost
    UC = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    # start up cost
    SU = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    # shut down cost
    SD = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    prob.addConstr(UC == SU + SD)

    if T > 1:
        prob.addConstr(SU[1:] >= su * (gen_v_all[1:] - gen_v_all[:-1]))
        prob.addConstr(SD[1:] >= sd * (gen_v_all[:-1] - gen_v_all[1:]))
    prob.addConstr(SU >= 0)
    prob.addConstr(SD >= 0)

    # fuel cost
    FC = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    if quadra_cost:
        prob.addConstr(FC >= gen_cost * gen_power_all + gen_cost_quadra * gen_power_all ** 2 + gen_cost_fixed*gen_v_all)
    else:
        # do piecewise linear approximation of the fuel cost
        coeff_list, intercept_list = piecewise_linear(gen_cost_quadra, gen_cost, gen_cap_individual)
        for i in range(len(coeff_list)):
            prob.addConstr(FC >= coeff_list[i] * gen_power_all + intercept_list[i] * gen_v_all + gen_cost_fixed*gen_v_all)

    RC = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    prob.addConstr(RC == urc * gen_UR_all + drc * gen_DR_all)

    # wind power curtailment cost
    k_cur = 65 # the cost (USD) of wind power curtailment, 50 GBP https://como.ceb.cam.ac.uk/media/preprints/c4e-preprint-304.pdf
    CP = prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    prob.addConstr(CP >= k_cur * WT_curtail_all)


    prob.setObjective(UC.sum() + FC.sum() + RC.sum() + CP.sum(), GRB.MINIMIZE)
    print(f'spent {time.time() - t_start} seconds to build the model.')
    # Solve the problem
    # set MIP gap
    prob.setParam('MIPGap', MIPGap)
    prob.setParam('IntFeasTol', 1e-9)
    prob.setParam('FeasibilityTol', 1e-9)
    prob.setParam('OptimalityTol', 1e-9)
    # # fix seed
    prob.setParam('Seed', gurobi_seed)
    prob.setParam('Threads', thread)
    if log_file_name is not None:
        prob.setParam('LogFile', log_file_name)

    # set time limit to 1 hr
    prob.setParam('TimeLimit', 3600)

    prob.optimize()


    ### now to optimising epsilon or theta
    if (method not in ['proposed_eps', 'ori_eps', 'proposed_theta']) or (prob.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]) or (prob.SolCount == 0):
        return prob, x, gen_UR_all, gen_DR_all, gen_v_all,  gen_power_all, WT_schedule_all, WT_curtail_all, A, B, d, np.nan, np.nan, np.nan, np.nan
    
    # get the base objecive
    min_cost_base = prob.objVal
    x_base = x.X.copy()

    eps_min = 0
    eps_max = epsilon # because we are minimising epsilon
    
    prob = gp.Model('SUC')
    gen_power_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the scheduled power output of thermal generators
    gen_UR_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the upward reserve of thermal generators
    gen_DR_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the downward reserve of thermal generators
    gen_v_all = prob.addMVar((T, num_gen), vtype=GRB.BINARY) # the binary variable indicating the on/off status of thermal generators

    WT_schedule_all = prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the scheduled power output of wind turbines
    WT_curtail_all = prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY) # the curtailment of wind turbines

    for t in range(T):
        # ---------------------------------------------------
        # power balance constraint at time step t
        prob.addConstr(gen_power_all[t, :].sum() + WT_schedule_all[t, :].sum() == load_bus_all[t, :].sum())

        # ---------------------------------------------------
        # constraints for thermal generators at time step t
        # pmax constraint
        prob.addConstr(gen_power_all[t, :] + gen_UR_all[t, :] <= gen_v_all[t, :] * gen_cap_individual)
        # pmin constraint
        prob.addConstr(gen_power_all[t, :] - gen_DR_all[t, :] >= gen_v_all[t, :] * gen_pmin_individual)
        if t > 0:
            # ramp constraint
            # upward
            prob.addConstr(gen_power_all[t, :] - gen_power_all[t-1, :] <= gen_ramp_up + (2-gen_v_all[t-1, :]-gen_v_all[t, :]) * gen_cap_individual)
            # downward
            prob.addConstr(gen_power_all[t-1, :] - gen_power_all[t, :] <= gen_ramp_down + (2-gen_v_all[t-1, :]-gen_v_all[t, :]) * gen_cap_individual)

        # reserve constraint
        prob.addConstr(gen_UR_all[t, :] <= gen_UR_max)
        prob.addConstr(0 <= gen_UR_all[t, :])
        prob.addConstr(gen_DR_all[t, :] <= gen_DR_max)
        prob.addConstr(0 <= gen_DR_all[t, :])

        # minimum up and down time constraint
        if t > 0:
            if t<= (T-UT):
                prob.addConstr(gen_v_all[t:t + UT, :].sum(axis=0) >= UT * (gen_v_all[t, :] - gen_v_all[t - 1, :]))
            else:
                prob.addConstr((gen_v_all[t:, :] - (gen_v_all[t, :] - gen_v_all[t - 1, :])).sum(axis=0) >= 0)

            if t<= (T-DT):
                prob.addConstr((1-gen_v_all[t:t + DT, :]).sum(axis=0) >= DT * (gen_v_all[t-1, :] - gen_v_all[t, :]))
            else:
                prob.addConstr((1-gen_v_all[t:, :] - (gen_v_all[t-1, :] - gen_v_all[t, :])).sum(axis=0) >= 0)

        # ---------------------------------------------------
        # constraints for wind turbines at time step t
        # relationship between WT_schedule and WT_curtail
        prob.addConstr(WT_schedule_all[t, :] + WT_curtail_all[t, :] == WT_pred[t, :])
        # curtailment smaller than the forecasted WT capacity
        prob.addConstr(WT_curtail_all[t, :] <= WT_pred[t, :])
        # curtailment non-negative
        prob.addConstr(0 <= WT_curtail_all[t, :])

    # ---------------------------------------------------
    # joint chance constraint
    # need to write the joint chance constraint as the form of Ax <= B\xi + d
    # where A \in R^{P*L}, x \in R^{L}, B \in R^{P*D}, \xi is the random vector \in R^{D}, d \in R^{P}
    # x is composed of [gen_UR_all_t=1, gen_DR_all_t=1, gen_power_all_t=1, WT_schedule_all_t=1,
    # gen_UR_all_t=2, gen_DR_all_t=2, gen_power_all_t=2, WT_schedule_all_t=2, ...]
    # \xi is ordered as [ran_vars_t=1, ran_vars_t=2, ..., ran_vars_t=T]
    # there are P = 2*T (system reserve) + 2*num_branch*T (branch flow constraints) individual constraints in the JCC
    P = 2*T + 2*num_branch*T
    L = gen_UR_all.size + gen_DR_all.size + gen_power_all.size + WT_curtail_all.size
    D = num_WT*T

    A = np.zeros((P, L))
    B = np.zeros((P, D))
    d = np.zeros(P)
    x = prob.addMVar(L, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    len_x_t = round(L / T) # the number of variables at each time step
    # link x to other variables
    for t in range(T):
        prob.addConstr(gen_UR_all[t, :] == x[len_x_t*t:len_x_t*t + num_gen])
        prob.addConstr(gen_DR_all[t, :] == x[len_x_t*t + num_gen:len_x_t*t + 2*num_gen])
        prob.addConstr(gen_power_all[t, :] == x[len_x_t*t + 2*num_gen:len_x_t*t + 3*num_gen])
        prob.addConstr(WT_schedule_all[t, :] == x[len_x_t*t + 3*num_gen:len_x_t*t + 3*num_gen + num_WT])

    len_ran_var_t = round(D / T) # the number of random variables at each time step
    num_constr_t = round(P / T) # the number of constraints at each time step
    for t in range(T):
        # system reserve constraints upward
        A[t*num_constr_t:t*num_constr_t + 1, len_x_t*t:len_x_t*t + num_gen] = -1
        B[t*num_constr_t:t*num_constr_t + 1, t*len_ran_var_t:(t+1)*len_ran_var_t] = 1
        d[t*num_constr_t:t*num_constr_t + 1] = -UR_extra_total

        # system reserve constraints downward
        A[t*num_constr_t + 1:t*num_constr_t + 2, len_x_t*t+num_gen:len_x_t*t+2*num_gen] = -1
        B[t*num_constr_t + 1:t*num_constr_t + 2, t*len_ran_var_t:(t+1)*len_ran_var_t] = -1
        d[t*num_constr_t + 1:t*num_constr_t + 2] = -DR_extra_total

        # transmission line constraints positive flow
        A[t*num_constr_t + 2:t*num_constr_t + 2 + num_branch, len_x_t*t+2*num_gen:len_x_t*t + 3*num_gen] = PTDF[:, gen_bus_list]
        A[t*num_constr_t + 2:t*num_constr_t + 2 + num_branch, len_x_t*t + 3*num_gen:len_x_t*t + 3*num_gen + num_WT] = PTDF[:, WT_bus_list]
        B[t*num_constr_t + 2:t*num_constr_t + 2 + num_branch, t*len_ran_var_t:(t+1)*len_ran_var_t] = -PTDF[:, WT_bus_list]
        d[t*num_constr_t + 2:t*num_constr_t + 2 + num_branch] = P_line_limit + PTDF @ load_bus_all[t]

        # transmission line constraints negative flow
        A[t*num_constr_t + 2 + num_branch:t*num_constr_t + 2 + 2*num_branch, len_x_t*t+2*num_gen:len_x_t*t + 3*num_gen] = -PTDF[:, gen_bus_list]
        A[t*num_constr_t + 2 + num_branch:t*num_constr_t + 2 + 2*num_branch, len_x_t*t + 3*num_gen:len_x_t*t + 3*num_gen + num_WT] = -PTDF[:, WT_bus_list]
        B[t*num_constr_t + 2 + num_branch:t*num_constr_t + 2 + 2*num_branch, t*len_ran_var_t:(t+1)*len_ran_var_t] = PTDF[:, WT_bus_list]
        d[t*num_constr_t + 2 + num_branch:t*num_constr_t + 2 + 2*num_branch] = P_line_limit - PTDF @ load_bus_all[t]

    #----------- process A, B, d -------
    # there are rows in B where all elements are zero because of the zero PTDF. We remove these rows
    # from the JCC and add them to the deterministic constraints
    zero_rows = np.where(np.all(B == 0, axis=1))[0]
    prob.addConstr(A[zero_rows] @ x <= d[zero_rows])
    # remove these rows from the JCC
    A = np.delete(A, zero_rows, axis=0)
    B = np.delete(B, zero_rows, axis=0)
    d = np.delete(d, zero_rows, axis=0)
    # P is updated
    P = A.shape[0]

    # prepare \xi
    random_var_scenario_index = rng.choice(WT_error_scenarios_train.shape[0], N_WDR, replace=False)
    random_var_scenarios = WT_error_scenarios_train[random_var_scenario_index, :, :].reshape(N_WDR, -1)

    # now we can apply the reformulation of the joint chance constraint

    if ('proposed' in method) or ('ori' in method) or ('exact' in method):
        # define ancillary variables
        s = prob.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        r = prob.addMVar(N_WDR, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        prob.addConstr(s >= 0)
        prob.addConstr(r >= 0)

        if 'eps' in method:
            epsilon = prob.addMVar(1, lb=eps_min, ub=eps_max)
            prob.addConstr(epsilon * N_WDR * s - gp.quicksum(r) >= theta * N_WDR, name='epsNs')
            k = np.floor(N_WDR * eps_max).astype(int)
        elif 'theta' in method:
            theta_base = theta
            theta = prob.addMVar(1, lb=theta_base, ub=GRB.INFINITY)
            # initialise theta to be the base theta
            prob.addConstr(epsilon * N_WDR * s - gp.quicksum(r) >= theta * N_WDR, name='epsNs') # in this case, epsilon is fixed at the base epsilon (used for solving the initial UC problem)
            k = np.floor(N_WDR * epsilon).astype(int)

        b_xi_set_all = []
        q_all = []
        q_all_extend = [] # only used for the exact WDRJCC
        N_p_all = []
        N_p_all_for_r = []
        A_big = []
        B_norm_big = []
        d_big = []

        effective_p = 0
        for p in range(P):
            b_xi_set = random_var_scenarios @ B[p]  # the set of b^T @ \xi for each i in N_WDR
            q_p = np.sort(b_xi_set)[k]
            if ('proposed' in method) or ('exact' in method):
                N_p = np.where(b_xi_set < q_p)[0]
            elif 'ori' in method:
                N_p = np.arange(N_WDR)
            else:
                raise ValueError('method can only be either "proposed" or "ori".')

            q_all.append(q_p) # q for each p. This is not skipped in the loop as the corresponding constraint is based on non-skipped parameters
            if len(N_p) == 0:
                continue
            # store these parameters so we can add them to the constraints at once
            b_xi_set_all.append(b_xi_set)
            N_p_all.append(N_p + effective_p*N_WDR)
            N_p_all_for_r.append(N_p)
            q_all_extend.append(np.tile(q_p, len(N_p)))

            # those matrices need to be "expanded"/duplicated so we can add all constraints at once
            A_big.append(np.tile(A[p:p+1], [len(N_p), 1]))
            B_norm_big.append(np.tile(norm(B[p], ord=2), len(N_p)))
            d_big.append(np.tile(d[p], len(N_p)))
            effective_p += 1

        q_all = np.array(q_all)
        if len(b_xi_set_all) > 0:
            # ==0 will only happen for the proposed method
            # for the proposed method, if the set N_p is empty, we do not need to add the constraint
            b_xi_set_all = np.hstack(b_xi_set_all)
            q_all_extend = np.hstack(q_all_extend)
            N_p_all = np.hstack(N_p_all)
            N_p_all_for_r = np.hstack(N_p_all_for_r)
            kappa = np.ones(N_WDR)

            kappa_big = np.tile(kappa, P)
            A_big = np.vstack(A_big)
            B_norm_big = np.hstack(B_norm_big)
            d_big = np.hstack(d_big)

            if ('proposed' in method) or ('ori' in method):
                prob.addConstr(kappa_big[N_p_all] * (b_xi_set_all[N_p_all] + d_big - A_big @ x) / B_norm_big >= s - r[N_p_all_for_r], name="N_p")
                # prob.addConstr(kappa_big[N_p_all] * (b_xi_set_all[N_p_all] + d_big - A_big @ x) / B_norm_big >= 0)

        if 'proposed' in method:
            prob.addConstr((q_all + d - A @ x) / norm(B, ord=2, axis=1) >= s, name="q_all")

    else:
        raise ValueError('method can only be either "proposed" or "ori" or "exact" or "wcvar" or "bonferroni".')
    # ---------------------------------------------------
    # Define the cost (objective) function
    # the unit commitment cost
    UC = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    # start up cost
    SU = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    # shut down cost
    SD = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    prob.addConstr(UC == SU + SD)

    if T > 1:
        prob.addConstr(SU[1:] >= su * (gen_v_all[1:] - gen_v_all[:-1]))
        prob.addConstr(SD[1:] >= sd * (gen_v_all[:-1] - gen_v_all[1:]))
    prob.addConstr(SU >= 0)
    prob.addConstr(SD >= 0)

    # fuel cost
    FC = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    if quadra_cost:
        prob.addConstr(FC >= gen_cost * gen_power_all + gen_cost_quadra * gen_power_all ** 2 + gen_cost_fixed*gen_v_all)
    else:
        # do piecewise linear approximation of the fuel cost
        coeff_list, intercept_list = piecewise_linear(gen_cost_quadra, gen_cost, gen_cap_individual)
        for i in range(len(coeff_list)):
            prob.addConstr(FC >= coeff_list[i] * gen_power_all + intercept_list[i] * gen_v_all + gen_cost_fixed*gen_v_all)

    RC = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    prob.addConstr(RC == urc * gen_UR_all + drc * gen_DR_all)

    # wind power curtailment cost
    k_cur = 65 # the cost (USD) of wind power curtailment, 50 GBP https://como.ceb.cam.ac.uk/media/preprints/c4e-preprint-304.pdf
    CP = prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY)
    prob.addConstr(CP >= k_cur * WT_curtail_all)


    prob.addConstr(UC.sum() + FC.sum() + RC.sum() + CP.sum() <= min_cost_base * bearable_cost_prop) # allow certain optimality loss when optimising epsilon or theta
    print(f'spent {time.time() - t_start} seconds to build the model.')

    if 'eps' in method:
        prob.setObjective(epsilon, GRB.MINIMIZE)
    elif 'theta' in method:
        prob.setObjective(theta, GRB.MAXIMIZE)

    # Solve the problem
    # set MIP gap
    prob.setParam('MIPGap', MIPGap)
    prob.setParam('IntFeasTol', 1e-9)
    prob.setParam('FeasibilityTol', 1e-9)
    prob.setParam('OptimalityTol', 1e-9)
    # # fix seed
    prob.setParam('Seed', gurobi_seed)
    prob.setParam('Threads', thread)
    if log_file_name is not None:
        prob.setParam('LogFile', log_file_name)

    # set time limit to 1 hr
    prob.setParam('TimeLimit', 3600)

    prob.optimize()

        # get solve time
    if (prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT]) or (prob.SolCount == 0):
        min_cost = np.nan
        t_solve = np.nan
        print('No feasible solution found.')
    else:
        min_cost = (UC.sum() + FC.sum() + RC.sum() + CP.sum()).getValue()
        t_solve = prob.Runtime

    return prob, x, gen_UR_all, gen_DR_all, gen_v_all,  gen_power_all, WT_schedule_all, WT_curtail_all, A, B, d, t_solve, min_cost, min_cost_base, x_base

def main():
    num_gen = 100  # the number of thermal generators
    num_WT = 10  # the number of wind turbines
    T = 24 # the number of time steps
    N_samples_train = 1000 # the number of wind power scenarios used for training
    N_samples_test = 5000 # the number of wind power scenarios used for testing
    thread = 4

    MIPGap = 0.001
    gurobi_seed = 120000

    quadra_cost = True

    method = 'proposed_eps' # proposed_eps, ori_eps, proposed_theta
    N_WDR = 100 # the number of scenarios for the WDRJCC
    epsilon = 0.1 # the risk level
    theta = 1e-1 # the Wasserstein radius. Bonferroni approximation requires small theta for feasibility
    bearable_cost_prop = 1.05

    load_scaling_factor = 1 # the scaling factor for the load
    network_name = 'case24_ieee_rts' # others can be case118, case300, case24_ieee_rts, case14, case5, case4gs, case_ieee30

    bigM =1e5 # this is only for "exact"
    log_file_name = None # the log file name
    #------------------

    network_dict = {'case118': pp.networks.case118(),
                    'case300': pp.networks.case300(),
                    'case24_ieee_rts': pp.networks.case24_ieee_rts(),
                    'case14': pp.networks.case14(),
                    'case5': pp.networks.case5(),
                    'case4gs': pp.networks.case4gs(),
                    'case_ieee30': pp.networks.case_ieee30()}

    seed = gurobi_seed
    rng = np.random.RandomState(seed)
    rng_fixed = np.random.RandomState(0)  # this is to avoid too much randomness that requires too many runs to have stable results

    # Tstart should also be randomly generated
    Tstart = rng.randint(0, 48 - T)  # rng.randint(0, 48-T)

    # load network model
    network = network_dict[network_name]

    # load network load data
    load_location = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'UK_norm_load_curve_highest.npy')
    network_load = np.load(load_location)
    # the network load is at half-hourly resolution, we need to average the consceutive time steps to get hourly resolution
    network_load = np.mean(np.vstack([network_load[::2],
                                      network_load[1::2]]), axis=0)
    # duplicate the network load to make it two days
    network_load = np.tile(network_load, 2)
    network_load = network_load[Tstart:Tstart+T]

    # -------------------------------------
    pp.rundcpp(network)
    _, ppci = _pd2ppc(network)
    bus_info = ppci['bus']
    branch_info = ppci['branch']
    PTDF = makePTDF(ppci["baseMVA"], bus_info, branch_info,
                    using_sparse_solver=False)

    num_branch = len(branch_info)

    # get load info
    load_bus_size = bus_info[:, 2] * load_scaling_factor

    load_total = np.sum(load_bus_size)
    # we then get the load curves at all buses, using the network_load curve
    load_bus_all = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

    ###### set generator capacity
    gen_cap_total = network.gen['p_mw'].sum() * load_scaling_factor  # the total generation capacity
    gen_cap_individual = gen_cap_total / num_gen  # the individual generation capacity
    # add some randomness when assigning the generation capacity to each generator
    gen_cap_individual = rng_fixed.uniform(0.6, 1.4, num_gen) * gen_cap_individual
    gen_pmin_individual = 0.25 * gen_cap_individual  # the individual minimum generation capacity. This is based on "DISTRIBUTIONALLY ROBUST CHANCE-CONSTRAINED GENERATION EXPANSION PLANNING"

    gen_ramp_up = rng_fixed.uniform(0.8 / 100, 30 / 100,
                              num_gen) * 60 * gen_cap_individual  # the ramping-up limit (%/min) of gas generators
    gen_ramp_down = gen_ramp_up  # the ramping-down limit of thermal generators
    # the following is based on the Alberta Electric System Operator, which states that the regulation range should be at most 10 times of the ramp rate in (MW/min)
    gen_UR_max = gen_ramp_up / 60 * 10  # the maximum upward reserve of thermal generators
    gen_DR_max = gen_UR_max  # the maximum downward reserve of thermal generators
    # generator cost parameters
    gen_cost = rng.uniform(23.13, 57.03, num_gen)  # the cost of gas generators (USD/MWh)
    gen_cost_quadra = rng.uniform(0.002, 0.008, num_gen)  # the quadratic cost of gas generators (USD/MWh^2)
    gen_cost_fixed = rng.uniform(0, 600,
                                 num_gen) * gen_cap_individual / 400  # the fixed cost of gas generators (USD/hr)
    urc = drc = gen_cost / 2  # the cost of reserve (USD/MWh)
    su = rng.uniform(20, 150, num_gen) * gen_cap_individual  # the start-up cost per start-up per MW
    sd = rng.uniform(2, 15, num_gen) * gen_cap_individual  # the shut-down cost per shut-down per MW
    # the minimum up and down time of gas generators
    UT = 2
    DT = 1

    ###### system reserve requirement
    UR_extra_total = 0. * load_total  # the required extra upward reserve. It is set 0 considering the major mismatch comes from wind
    DR_extra_total = 0. * load_total  # the required extra downward reserve

    # get generator locations
    bus_list = np.arange(bus_info.shape[0])
    gen_bus_list = rng_fixed.choice(bus_list, num_gen, replace=True)
    WT_bus_list = rng_fixed.choice(bus_list, num_WT, replace=True)

    # get line info
    P_line_limit = np.abs(ppci['branch'][:, 5])  # the line flow limit
    # clip on 2 times of the total load to avoid numerical issues
    P_line_limit = np.clip(P_line_limit, 0, 2 * load_total)

    WT_total = 0.6 * load_total
    WT_individual = WT_total / num_WT
    # load the wind power scenarios, which is decomposed into prediction and error scenarios
    WT_pred, WT_error_scenarios, WT_full_scenarios = WT_sce_gen(num_WT, N_samples_train + N_samples_test)
    WT_pred = WT_pred[Tstart:Tstart+T] * WT_individual  # scale
    WT_error_scenarios = WT_error_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
    WT_full_scenarios = WT_full_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
    # generate training and testing scenarios
    WT_error_scenarios_train = WT_error_scenarios[:N_samples_train]
    WT_error_scenarios_test = WT_error_scenarios[N_samples_train:]

    # perform SUC
    input_param_dict = {'T': T, 'num_gen': num_gen, 'num_WT': num_WT, 'num_branch': num_branch,
                        'load_bus_all': load_bus_all, 'PTDF': PTDF, 'gen_cap_individual': gen_cap_individual,
                        'gen_pmin_individual': gen_pmin_individual, 'gen_UR_max': gen_UR_max, 'gen_DR_max': gen_DR_max,
                        'UR_extra_total': UR_extra_total, 'DR_extra_total': DR_extra_total, 'gen_ramp_up': gen_ramp_up,
                        'gen_ramp_down': gen_ramp_down, 'UT': UT, 'DT': DT, 'WT_pred': WT_pred,
                        'WT_error_scenarios_train': WT_error_scenarios_train, 'P_line_limit': P_line_limit,
                        'gen_bus_list': gen_bus_list, 'WT_bus_list': WT_bus_list, 'N_WDR': N_WDR, 'epsilon': epsilon,
                        'thread': thread,
                        'theta': theta, 'method': method, 'MIPGap': MIPGap, 'gen_cost': gen_cost,
                        'gen_cost_quadra': gen_cost_quadra, 'gen_cost_fixed': gen_cost_fixed,
                        'urc': urc, 'drc': drc, 'su': su, 'sd': sd, 'bigM': bigM, 'gurobi_seed': gurobi_seed,
                        'log_file_name': log_file_name, 'rng': rng, 'quadra_cost': quadra_cost, 'bearable_cost_prop': bearable_cost_prop}
    prob, x, gen_UR_all, gen_DR_all, gen_v_all, \
    gen_power_all, WT_schedule_all, WT_curtail_all, A, B, d, t_solve, min_cost, min_cost_base, x_base = solve_SUC(**input_param_dict)

    # Check the status of the solution
    if np.isnan(min_cost):
        raise ValueError('The problem does not have an optimal solution.')


    # print the total power from generators, total reserve from generators, total load
    print('------------------------------------')
    print('Total power from generators:', gen_power_all.X.sum(axis = 1))
    print('Total upward reserve from generators:', gen_UR_all.X.sum(axis = 1))
    print('Total downward reserve from generators:', gen_DR_all.X.sum(axis = 1))
    print('Total load:', load_bus_all.sum(axis = 1))

    # calculate the out-of-sample JCC satisfaction rate
    random_var_scenarios_test = WT_error_scenarios_test.reshape(WT_error_scenarios_test.shape[0], -1)
    reliability_test = np.mean(np.all(A @ x.X <= random_var_scenarios_test@B.T + d, axis=1))
    reliability_test_base = np.mean(np.all(A @ x_base <= random_var_scenarios_test@B.T + d, axis=1))
    print('------------------------------------')
    print('Optimal value:', prob.objVal)
    print(f'The out-of-sample JCC satisfaction rate is {reliability_test*100}%, the original satisfaction rate is {reliability_test_base*100}%')
    print(f'The original objective value is {min_cost_base:.2f}, the more robust cost is {min_cost:.2f}, which is {((min_cost - min_cost_base)/min_cost_base*100):.2f}% higher than the original cost.')

    # plot the generation power output of a thermal generator
    import matplotlib.pyplot as plt
    T_series = np.arange(T)
    fig, axs = plt.subplots(3, 1, figsize=[6, 8])
    gen_idx = 0
    axs[0].step(T_series, gen_power_all.X[:, gen_idx], color='b', label='gen')
    # plot gen + upward reserve
    axs[0].step(T_series, gen_power_all.X[:, gen_idx] + gen_UR_all.X[:, 0], color='r', linestyle='--', label='gen + UR')
    # plot gen - downward reserve
    axs[0].step(T_series, gen_power_all.X[:, gen_idx] - gen_DR_all.X[:, 0], color='g', linestyle='--', label='gen - DR')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Power')
    axs[0].set_xticks(T_series, T_series)
    axs[0].set_ylim([0, 1.1 * gen_cap_individual[gen_idx]])
    axs[0].axhline(gen_cap_individual[gen_idx], color='black', linestyle='--', label='gen cap')
    axs[0].grid()
    axs[0].set_title('thermal generator')
    axs[0].legend(ncol=4)

    # now for wind
    axs[1].step(T_series, WT_schedule_all.X[:, 0], label='WT P_schedule')
    axs[1].step(T_series, WT_full_scenarios[0, :, 0], color='r', linestyle='--', label='WT P real max')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Power')
    axs[1].set_xticks(T_series, T_series)
    axs[1].set_ylim([0, 1.1 * WT_individual])
    axs[1].legend(ncol=3)
    axs[1].grid()
    axs[1].set_title('wind turbine')

    # plot the load curve
    axs[2].step(T_series, load_bus_all.sum(axis=1))
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Power')
    axs[2].set_xticks(T_series, T_series)
    axs[2].set_ylim([0, 1.1 * load_bus_all.sum(axis=1).max()])
    axs[2].grid()
    axs[2].set_title('total load curve')

    plt.tight_layout()
    # plt.show()

    # plot a 2*1 heatmap to show the generation power and the commitment status
    fig, axs = plt.subplots(1, 2, figsize=[8, 6])
    axs[0].imshow(gen_power_all.X.T, aspect='auto', cmap='viridis')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Generator')
    axs[0].set_xticks(T_series, T_series)
    # show only 10 ylabels with equal interval
    axs[0].set_yticks(np.linspace(0, num_gen - 1, 10).astype(int), np.linspace(0, num_gen - 1, 10).astype(int))
    axs[0].set_title('Generation power output')
    # display a colorbar
    plt.colorbar(axs[0].imshow(gen_power_all.X.T, aspect='auto', cmap='viridis'), ax=axs[0])

    axs[1].imshow(gen_v_all.X.T, aspect='auto', cmap='viridis')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Generator')
    axs[1].set_xticks(T_series, T_series)
    # show only 10 ylabels with equal interval
    axs[1].set_yticks(np.linspace(0, num_gen-1, 10).astype(int), np.linspace(0, num_gen-1, 10).astype(int))
    axs[1].set_title('Generator commitment status')
    # display a colorbar
    plt.colorbar(axs[1].imshow(gen_v_all.X.T, aspect='auto', cmap='viridis'), ax=axs[1])

    # check if gen_v_all is binary with a tolerance of 1e-8
    print(f'gen_v_all is binary: {np.all(np.abs(gen_v_all.X - np.round(gen_v_all.X)) < 1e-8)}')

    plt.tight_layout()
    # plt.show()

    print(f'spent {t_solve} seconds for solving the SUC')
    print(f'The method used is {method}')
    print('------------------------------------')
    print(f'theta = {theta}, epsilon = {epsilon}, N_WDR = {N_WDR}, load_scaling_factor = {load_scaling_factor}, network_name = {network_name}')

if __name__ == '__main__':
    main()