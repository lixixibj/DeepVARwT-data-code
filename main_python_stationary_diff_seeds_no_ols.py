import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve, inv, eigvalsh
import math
import os
import timeit
from joblib import Parallel, delayed

def kron(a, b):
    """
    Kronecker product of matrices a and b
    """
    return np.kron(a, b)

def make_var_cov_matrix_for_innovation_of_varp(lower_triang_params, m, order):
    """
    Get variance-covariance matrix of innovations of VAR(p).
    """
    lower_t_matrix = np.eye(m)
    count_temp = 0
    for i in range(m):
        for j in range(i+1):
            lower_t_matrix[i, j] = lower_triang_params[count_temp]
            count_temp = count_temp + 1
    
    var_cov_matrix = lower_t_matrix @ lower_t_matrix.T
    return var_cov_matrix

def A_coeffs_for_causal_VAR(A_coeffs_from_lstm, p, d, var_cov_innovations_varp):
    """
    Generate causal VAR coefficients.
    """
    Id = np.eye(d)
    all_p = np.random.randn(d, d, p)
    initial_A_coeffs = A_coeffs_from_lstm.reshape(d, d, p)
    
    for i1 in range(p):
        A = initial_A_coeffs[:, :, i1]
        v = Id + A @ A.T
        eigvals = eigvalsh(Id + A @ A.T)
        B = cholesky(Id + A @ A.T, lower=True)
        all_p[:, :, i1] = solve(B, A)
    
    all_phi = np.random.randn(d, d, p, p)  # [ , , i, j] for phi_{i, j}
    all_phi_star = np.random.randn(d, d, p, p)  # [ , , i, j] for phi_{i, j}*
    
    # Set initial values
    Sigma = Id.copy()
    Sigma_star = Id.copy()
    L = Id.copy()
    L_star = L.copy()
    
    # Recursion algorithm (Ansley and Kohn 1986, lemma 2.3)
    for s in range(p):
        all_phi[:, :, s, s] = L @ all_p[:, :, s].copy() @ inv(L_star)
        all_phi_star[:, :, s, s] = L_star @ all_p[:, :, s].copy().T @ inv(L)
        
        if s >= 1:
            for k in list(range(1, s + 1)):
                all_phi[:, :, s, k - 1] = all_phi[:, :, s - 1, k - 1].copy() - \
                                         all_phi[:, :, s, s].copy() @ all_phi_star[:, :, s - 1, s - k].copy()
                all_phi_star[:, :, s, k - 1] = all_phi_star[:, :, s - 1, k - 1].copy() - \
                                              all_phi_star[:, :, s, s].copy() @ all_phi[:, :, s - 1, s - k].copy()
        
        Sigma_next = Sigma - all_phi[:, :, s, s].copy() @ Sigma_star @ all_phi[:, :, s, s].copy().T
        Sigma_star = Sigma_star - all_phi_star[:, :, s, s].copy() @ Sigma @ all_phi_star[:, :, s, s].copy().T
        L_star = cholesky(Sigma_star, lower=True)
        Sigma = Sigma_next
        L = cholesky(Sigma, lower=True)
    
    # cal T matrix
    lower_t_for_innovations_varp = cholesky(var_cov_innovations_varp, lower=True)
    T = lower_t_for_innovations_varp @ inv(L)
    
    all_A = all_phi[:, :, p - 1, 0:p].copy()
    for i in range(p):
        all_A[:, :, i] = T @ all_A[:, :, i].copy() @ inv(T)
    
    return all_A

def get_A_coeff_m_for_VAR_1(inital_A_m, m, order):
    """
    Put VAR(p) into VAR(1) and get coefficient matrix of state equation.
    """
    F_list = []
    for c in range(order):
        F_list.append(inital_A_m[:, :, c])
    
    # concatenate
    F_temp1 = np.concatenate(F_list, axis=1)
    
    mp = m * order
    added_tensor = np.eye(mp - m, mp)
    coffs_m = np.concatenate((F_temp1, added_tensor), axis=0)
    return coffs_m

def put_A_coeff_together(inital_A_m, order):
    """
    Put VAR coefficient matrices together.
    """
    A_list = []
    for c in range(order):
        A_list.append(inital_A_m[:, :, c])
    
    # concatenate
    A = np.concatenate(A_list, axis=1)
    return A

def make_var_cov_of_innovations_var1(lower_triang_params_form_lstm, m, order):
    """
    Put VAR(p) into VAR(1) and get var-cov matrix of innovations of state equation.
    """
    mp = m * order
    lower_t_matrix = np.eye(m)
    count_temp = 0
    for i in range(m):
        for j in range(i+1):
            lower_t_matrix[i, j] = lower_triang_params_form_lstm[count_temp]
            count_temp = count_temp + 1
    
    var_cov_matrix = lower_t_matrix @ lower_t_matrix.T
    
    zeros_cols = np.zeros((m, mp - m))
    zeros_rows = np.zeros((mp - m, mp))
    c = np.concatenate((var_cov_matrix, zeros_cols), axis=1)
    var_cov_of_innovations_for_var1 = np.concatenate((c, zeros_rows), axis=0)
    return var_cov_of_innovations_for_var1

def calculate_p0(m, order, A_coeffs_var1, var_cov_innovations_var1):
    """
    Calculate variance-covariance matrix of (y_b,...,y_1)
    """
    i_matrix = np.eye(m * m * order * order)
    i_F_F = i_matrix - kron(A_coeffs_var1, A_coeffs_var1)
    Q_transposed = var_cov_innovations_var1.T
    r, c = var_cov_innovations_var1.shape
    vec_Q = Q_transposed.reshape(r * c, 1)
    
    vec_p0 = inv(i_F_F) @ vec_Q
    p0_temp = vec_p0.reshape(m * order, m * order)
    p0 = p0_temp.T
    return p0

def transfrom_var_cov_matrix(var_cov_matrix, order, m):
    """Transform variance-covariance matrix of (y_b,...,y_1) to of (y_1,...,y_p)"""
    var_cov_matrix_new = np.zeros_like(var_cov_matrix)
    for r in range(order):
        for c in range(order):
            block = var_cov_matrix[
                (order-1-r)*m : (order-r)*m,
                (order-1-c)*m : (order-c)*m
            ]
            var_cov_matrix_new[r*m:(r+1)*m, c*m:(c+1)*m] = block
    return var_cov_matrix_new

def get_lagged_observations(target_y, current_t, order, m):
    """
    Get lagged observations of detrend series.
    """
    lagged_list = []
    for i in range(order):
        lagged_list.append(target_y[current_t-i-1, :].reshape(m, 1))
    lagged_obs = np.concatenate(lagged_list, axis=0)
    return lagged_obs

def compute_log_likelihood(target, var_coeffs, residual_parameters, m, order):
    """
    Compute -log-likelihood.
    """
    len_of_seq = target.shape[0]
    log_likelihood_temp = 0
    
    # calculate var-cov of innovations of var(p)
    var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(residual_parameters, m, order)
    all_casual_coeffs = A_coeffs_for_causal_VAR(var_coeffs, order, m, var_cov_innovations_varp)
    A_coeffs_var1 = get_A_coeff_m_for_VAR_1(all_casual_coeffs, m, order)
    A = put_A_coeff_together(all_casual_coeffs, order)
    var_cov_innovations_var1 = make_var_cov_of_innovations_var1(residual_parameters, m, order)
    
    # calculate variance-covariance matrix of first p obs (y_1,...,y_p)
    var_cov_matrix_for_initial_p_obs = calculate_p0(m, order, A_coeffs_var1, var_cov_innovations_var1)
    var_cov_matrix_for_initial_p_obs_update = transfrom_var_cov_matrix(var_cov_matrix_for_initial_p_obs, order, m)
    
    # get first p obs
    first_p_obs = np.zeros((order * m, 1))
    for i in range(order):
        b = i * m
        e = i * m + m
        first_p_obs[b:e, :] = target[i, :].reshape(m, 1)
    
    # Add small regularization to avoid singular matrix
    # reg = 1e-8 * np.eye(var_cov_matrix_for_initial_p_obs_update.shape[0])
    # var_cov_matrix_for_initial_p_obs_update += reg
    
    log_likelihood_temp = log_likelihood_temp + np.log(np.linalg.det(var_cov_matrix_for_initial_p_obs_update)) + \
                         first_p_obs.T @ inv(var_cov_matrix_for_initial_p_obs_update) @ first_p_obs
    
    for t in range(order, len_of_seq):
        lagged_obs = get_lagged_observations(target, t, order, m)
        error = target[t, :].reshape(m, 1) - A @ lagged_obs
        
        # Add small regularization to avoid singular matrix
        #var_cov_innovations_varp_reg = var_cov_innovations_varp + 1e-8 * np.eye(m)
        
        log_likelihood_temp = log_likelihood_temp + np.log(np.linalg.det(var_cov_innovations_varp)) + \
                             error.T @ inv(var_cov_innovations_varp) @ error
    
    return 0.5 * (log_likelihood_temp + m * len_of_seq * np.log(2 * math.pi))

def objective_function(params, data, m, order):
    """
    Objective function for optimization (negative log-likelihood)
    """
    try:
        # Split parameters
        ar_size = m * m * order
        ar_params = params[:ar_size]
        residual_params = params[ar_size:]
        
        # Compute negative log-likelihood
        neg_log_likelihood = compute_log_likelihood(data, ar_params, residual_params, m, order)
        
        # Return scalar value
        if isinstance(neg_log_likelihood, np.ndarray):
            return neg_log_likelihood.item()
        return neg_log_likelihood
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e10  # Return large value if computation fails

def optimize_params(data, iters, m, order, res_saving_path):
    """
    Network training - Python version using scipy optimization
    """
    seq_len = data.shape[0]
    y = data.values
    
    # Initialize parameters randomly
    total_ar_params = m * m * order
    total_residual_params = int(m * (m + 1) / 2)
    total_params = total_ar_params + total_residual_params
    
    # Random initialization
    initial_params = np.random.randn(total_params) * 0.1
    
    print(f"Total params: {total_params}")
    
    # Create folder for saving results
    pretrained_model_file_path = res_saving_path + 'pretrained_model/'
    if not os.path.exists(pretrained_model_file_path):
        os.makedirs(pretrained_model_file_path)
    
    log_likelihood = []
    
    # Optimize
    result = minimize(
        objective_function,
        initial_params,
        args=(y, m, order),
        method='L-BFGS-B',
        options={'maxiter': iters, 'disp': True}
    )
    
    # Extract final parameters
    final_params = result.x
    ar_size = m * m * order
    final_ar_params = final_params[:ar_size]
    final_residual_params = final_params[ar_size:]
    
    # Save results (using final loss value)
    final_loss = result.fun
    loss_df = pd.DataFrame({'likelihood': [final_loss]})
    loss_df.to_csv(res_saving_path + 'log_likelihood_loss_phase2.csv')
    
    # Get final coefficients in original format
    var_coeffs = final_ar_params.reshape(m, m, order)
    var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(final_residual_params, m, order)
    all_coeffs = A_coeffs_for_causal_VAR(final_ar_params, order, m, var_cov_innovations_varp)
    
    return all_coeffs, var_cov_innovations_varp

def process_section(num, data_saving_path, iters, m, order, res_saving_path):
    """Process a single experiment"""
    path_of_dataset = data_saving_path + f'exp{num}_len800_VAR2_m3_stationary_train.csv'
    data = pd.read_csv(path_of_dataset).iloc[:, 1:]
    res_saving_path1 = res_saving_path + str(num) + '/'
    
    if not os.path.exists(res_saving_path1):
        os.makedirs(res_saving_path1)
    
    try:
        np.random.seed(num)
        all_coeffs, var_cov_m = optimize_params(data, iters, m, order, res_saving_path1)
    except Exception as e:
        print(f'Error in experiment {num}: {e}')
        np.random.seed(200)
        all_coeffs, var_cov_m = optimize_params(data, iters, m, order, res_saving_path1)
    
    return all_coeffs, var_cov_m

# Main execution
if __name__ == "__main__":
    start = timeit.default_timer()
    
    # Parameters
    m = 3
    order = 2
    iters = 6000
    
    data_saving_path = './simulation-study/sim_exp100/simulated_data_len800_VAR2_m3_stationary/'
    res_saving_path = './simulation-study/100-res-python-version/'
    
    # 100 experiments
    exps = 100
    
    # Run the process in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_section)(num, data_saving_path, iters, m, order, res_saving_path) 
        for num in range(exps)
    )
    
    # Extract results
    coeffs_all = np.zeros((exps, m * m * order))
    var_cov_all = np.zeros((exps, m * m))
    
    for num, (all_coeffs, var_cov_m) in enumerate(results):
        coeffs = []
        for i in range(order):
            coeffs.extend(all_coeffs[:, :, i].flatten().tolist())
        coeffs_all[num, :] = coeffs
        var_cov_all[num, :] = var_cov_m.flatten().tolist()
    
    # Save results
    pd.DataFrame(coeffs_all).to_csv('./simulation-study/100-res-python-version/all_coeffs_100exps_python_rep.csv')
    pd.DataFrame(var_cov_all).to_csv('./simulation-study/100-res-python-version/all_var_cov_100exps_python_rep.csv')
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)