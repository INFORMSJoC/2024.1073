import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import os
import numpy as np
import pandas as pd
from SUC import solve_SUC
import pandapower as pp
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
from WT_error_gen import WT_sce_gen
import itertools
from gurobipy import GRB
from joblib import Parallel, delayed

bigM = 1e5  # this is only for "exact"
thread = 4 # the number of threads for Gurobi solver
n_jobs = 16 # higher than 20 would cause memory issues
# ---------------------------------------------------------------------------------
T_list = [4, 8, 12, 16, 20, 24, 28, 32]  # the number of time steps
theta_list = [5e-1, 1e-1]  # [5e-1] the Wasserstein radius. Bonferroni approximation requires small theta for feasibility
epsilon_list = [0.05, 0.025]  # [0.05, 0.025] the risk level
# generate ten seeds with a step size of 1000
gurobi_seed_list = [i for i in range(10000*0, 10000*300, 10000)] # range(0, 10000*10, 10000), range(100000, 10000*20, 10000), range(0, 10000*20, 10000)
num_gen_list = [100]  # [200, 1000] the number of thermal generators
N_WDR_list = [100]  # the number of scenarios for the WDRJCC
load_scaling_factor_list = [1]  # [1] the scaling factor for the load
method_list = ['ori', 'wcvar', 'bonferroni']  # proposed, ori, exact, wcvar, bonferroni. the method to reformulate the WDRJCC
network_name_list = ['case24_ieee_rts']  # case118, case24_ieee_rts
quadra_cost = True

# find the combination of all these parameters
param_comb = list(itertools.product(network_name_list, load_scaling_factor_list, epsilon_list, theta_list, T_list, num_gen_list, N_WDR_list, gurobi_seed_list, method_list))

save_path_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', f'SUC_results_bigM{int(bigM)}_thread{int(thread)}')
if not os.path.exists(save_path_root):
    os.makedirs(save_path_root)

# solve the SUC for all the combinations
for param in param_comb:
    network_name, load_scaling_factor, epsilon, theta, T, num_gen, N_WDR, gurobi_seed, method = param

    log_file_name = (f'{network_name}_theta{theta}_epsilon{epsilon}_gurobi_seed{gurobi_seed}'
                      f'_num_gen{num_gen}_N_WDR{N_WDR}_load_scaling_factor{load_scaling_factor}_{method}_T{T}{"quadra_cost" if quadra_cost else ""}.txt')
    log_file_name = os.path.join(save_path_root, log_file_name)

    result_dict_path = (f'result_{network_name}_theta{theta}_epsilon{epsilon}_gurobi_seed{gurobi_seed}'
                         f'_num_gen{num_gen}_N_WDR{N_WDR}_load_scaling_factor{load_scaling_factor}_{method}_T{T}{"quadra_cost" if quadra_cost else ""}.npy')
    result_dict_path = os.path.join(save_path_root, result_dict_path)

    # remove files
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    if os.path.exists(result_dict_path):
        os.remove(result_dict_path)