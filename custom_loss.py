1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-




import torch
import pandas as pd
import numpy as np




def compute_error_for_trend_estimation(
        target,
        # trend shape(seq,batch,input_size)
        trend):
    r"""
        OLS for trend estimation in Phase1.
        Parameters
        ----------
        target
           description: the observations
           type: tensor
           shape: (T,m)

        trend
           description: estimated trends
           type: tensor
           shape: (T,batch=1,m)

        Returns
        -------
        error
           description: errors
           type: tensor
    """

    len_of_seq = target.shape[0]
    trend_value=trend[:,0,:]
    error=torch.sum(torch.square(trend_value-target))

    return error



def compute_log_likelihood(
        target,
        var_coeffs,
        residual_parameters,
        m,
        order,
        trend):



    r"""
        Compute  -log-likelihood.
        Parameters
        ----------
        target
           description: the observations
           type: tensor
           shape: (T,m)

        var_coeffs
           description: var coefficient parameters
           type: tensor
           shape: (m*m*p,)

        residual_parameters
           description:  parameters for lower triangular matrix
           type: tensor
           shape: ((m+1)*m/2,)

        trend
           description: estimated trend
           type: tensor
           shape: (T,batch=1,m)


        Returns
        -------
        -log-likelihood
           description: -log-likelihood
           type: tensor 
    """

    len_of_seq = target.shape[0]
    log_likelihood_temp= 0
    penalty_temp = 0

    # calculate var-cov of innovations of var(p)
    var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(residual_parameters, m,order)
    all_casual_coeffs = A_coeffs_for_causal_VAR(var_coeffs, order, m,var_cov_innovations_varp)
    A_coeffs_var1 = get_A_coeff_m_for_VAR_1(all_casual_coeffs, m, order)
    A=put_A_coeff_together(all_casual_coeffs, order)
    var_cov_innovations_var1=make_var_cov_of_innovations_var1(residual_parameters, m, order)
    #calculate variance-covariance matrix of first p obs (y_1,...,y_p)
    #var_cov_matrix_for_initial_p_obs: shape(mp*mp), but this variance-covariance matrix is for first p obs (y_b,...,y_1) and it needs to be tranformed for (y_1,...,y_p)
    var_cov_matrix_for_initial_p_obs = calculate_p0(m, order, A_coeffs_var1, var_cov_innovations_var1)
    #transfrom it to variance-covariance matrix of (y_1,...,y_p)
    var_cov_matrix_for_initial_p_obs_update=transfrom_var_cov_matrix(var_cov_matrix_for_initial_p_obs, order, m)

    #get first p obs
    first_p_obs=torch.zeros(order*m, 1)
    for i in range(order):
        trend_value = trend[i, 0, :]
        b=i*m
        e=i*m+m
        first_p_obs[b:e,:]=target[i,:].reshape(m,1)-trend_value.reshape(m, 1)

    log_likelihood_temp = log_likelihood_temp + torch.log(torch.det(var_cov_matrix_for_initial_p_obs_update)) + torch.mm(
        torch.mm(first_p_obs.t(), torch.inverse(var_cov_matrix_for_initial_p_obs_update)), first_p_obs)


    for t in range(order,len_of_seq):
        trend_value = trend[t, 0, :]
        lagged_obs=get_lagged_observations_detrended(target,trend, t, order, m)
        error=target[t,:].reshape(m, 1)-trend_value.reshape(m, 1)-torch.mm(A,lagged_obs)
        log_likelihood_temp=log_likelihood_temp+torch.log(torch.det(var_cov_innovations_varp))+torch.mm(
            torch.mm(error.t(), torch.inverse(var_cov_innovations_varp)), error)

    return 0.5*(log_likelihood_temp+m*len_of_seq*torch.log(torch.tensor(2*math.pi)))






def transfrom_var_cov_matrix(var_cov_matrix,order,m):

    r"""
        Transform variance-covariance matrix of (y_b,...,y_1) to of (y_1,...,y_p)
        Parameters
        ----------
        var_cov_matrix
           description: variance-covariance matrix of (y_b,...,y_1)
           type: tensor
           shape: (m*p,m*p)

        order
           description: order of VAR
           type: int
  
        m
           description: number of series
           type: int  

        Returns
        -------
        var_cov_matrix
           description: variance-covariance matrix of (y_1,...,y_p)
           type: tensor
           shape: (m*p,m*p)
    """

    for r in range(1,order):
        b_r = r * m
        e_r=r*m+m
        for c in range(r):
            b_c=c* m
            e_c=c* m+m
            #copy upper block matrix
            elem_temp=var_cov_matrix[b_c:e_c,b_r:e_r].clone()
            var_cov_matrix[b_c:e_c,b_r:e_r]=var_cov_matrix[b_r:e_r,b_c:e_c].clone()
            var_cov_matrix[b_r:e_r,b_c:e_c]=elem_temp

    return (var_cov_matrix)







def calculate_p0(m,order,A_coeffs_var1,var_cov_innovations_var1):
    r"""
        Calcualte variance-covariance matrix of (y_b,...,y_1)
        Parameters
        ----------
        A_coeffs_var1
           description: coefficient matrix of VAR(1) (put VAR(p) into VAR(1))
           type: tensor
           shape: (m*p,m*p)

        var_cov_innovations_var1
           description: variance-covariance matrix of VAR(1)
           type: tensor
           shape: (m*p,m*p)

        order
           description: order of VAR
           type: int
  
        m
           description: number of series
           type: int  

        Returns
        -------
        p0
           description: variance-covariance matrix of (y_b,...,y_1)
           type: tensor
           shape: (m*p,m*p)
    """

    i_matrix=torch.eye(m*m*order*order)
    i_F_F=i_matrix-kron(A_coeffs_var1,A_coeffs_var1)

    Q_transposed = torch.transpose(var_cov_innovations_var1, 0, 1)
    r=var_cov_innovations_var1.shape[0]
    c=var_cov_innovations_var1.shape[1]
    vec_Q = Q_transposed.reshape(r * c, 1)
    #
    vec_p0=torch.mm(torch.inverse(i_F_F),vec_Q)
    #
    p0_temp=vec_p0.reshape(m*order,m*order)
    #
    p0=torch.transpose(p0_temp,0,1)
    # print('p0')
    # print(p0)
    return (p0)




"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""
import torch
def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. 
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)








def A_coeffs_for_causal_VAR(A_coeffs_from_lstm,p,d,var_cov_innovations_varp):
    r"""
        Generate causal VAR coefficients.
        Parameters
        ----------
        A_coeffs_from_lstm
           description: intial A coeffs from LSTM network
           type: tensor
           shape: (d*d*p,)
        p
           description: order of VAR model
           type: int

        d
           description: the number of series
           type: int

        var_cov_innovations_varp
           description: the var-cov matrix of innovations of VAR(p)
           type: tensor
           shape:(m,m)

        Returns
        -------
        all_A
           description: causal VAR coeffs
           type: tensor
           shape: (d,d,p)
    """
    Id = torch.eye(d)
    all_p = torch.randn(d, d, p)
    initial_A_coeffs = A_coeffs_from_lstm.reshape(d, d, p)

    for i1 in range(p):
        A = initial_A_coeffs[:, :, i1]
        v=Id + torch.mm(A, A.t())
        B = torch.linalg.cholesky(Id + torch.mm(A, A.t())).float()

        all_p[:, :, i1] = torch.linalg.solve(B, A)


    all_phi = torch.randn(d, d, p, p)  # [ , , i, j] for phi_{i, j}
    all_phi_star = torch.randn(d, d, p, p)  # [ , , i, j] for phi_{i, j}*
    # Set initial values
    Sigma = Id
    Sigma_star = Id
    L= Id
    L_star = L
    #Gamma = Id
    # Recursion algorithm (Ansley and Kohn 1986, lemma 2.3)
    for s in range(p):
        all_phi[:, :, s, s] = torch.mm(torch.mm(L, all_p[:, :, s].clone()), torch.inverse(L_star))
        all_phi_star[:, :, s, s] = torch.mm(torch.mm(L_star, all_p[:, :, s].clone().t()), torch.inverse(L))
        if s >= 1:
            for k in list(range(1, s + 1)):
                all_phi[:, :, s, k - 1] = all_phi[:, :, s - 1, k - 1].clone() - torch.mm(all_phi[:, :, s, s].clone(),
                                                                                 all_phi_star[:, :, s - 1, s - k].clone())
                all_phi_star[:, :, s, k - 1] = all_phi_star[:, :, s - 1, k - 1].clone() - torch.mm(all_phi_star[:, :, s, s].clone(),
                                                                                           all_phi[:, :, s - 1, s - k].clone())
        Sigma_next = Sigma - torch.mm(all_phi[:, :, s, s].clone(), torch.mm(Sigma_star, all_phi[:, :, s, s].clone().t()))
        Sigma_star = Sigma_star - torch.mm(all_phi_star[:, :, s, s].clone(),
                                                   torch.mm(Sigma, all_phi_star[:, :, s, s].clone().t()))
        L_star = torch.cholesky(Sigma_star)
        Sigma = Sigma_next
        L = torch.cholesky(Sigma)

#   cal T matrix
    lower_t_for_innovations_varp=torch.linalg.cholesky(var_cov_innovations_varp)
    T=torch.mm(lower_t_for_innovations_varp,torch.inverse(L))


    all_A = all_phi[:, :, p - 1, 0:p].clone()
    for i in range(p):
        all_A[:,:,i]=torch.mm(torch.mm(T,all_A[:,:,i].clone()),torch.inverse(T))
    return all_A


def make_var_cov_matrix_for_innovation_of_varp(lower_triang_params,m,order):
    r"""
        Get variance-covariance matrix of innovations of VAR(p).
        Parameters
        ----------
        lower_triang_params
           description: elements for lower triangular matrix for generating var-cov matrix
           type: tensor 
           shape((m*(m+1)/2)，)

        m
           description: the number of series
           type: int

        order
           description: the order of VAR model
           type: int

        Returns
        -------
        var_cov_matrix
           description: variance-covariance matrix of innovations of VAR(p)
           type: tensor
           shape: (m,m)
    """
    mp=m*order
    number_of_parms=m*(m+1)/2
    lower_t_matrix=torch.eye(m,m)
    count_temp=0
    for i in range(m):
        for j in range(i+1):
            lower_t_matrix[i,j]=lower_triang_params[count_temp]
            count_temp=count_temp+1

    var_cov_matrix = torch.mm(lower_t_matrix, lower_t_matrix.t())

    return (var_cov_matrix)


def get_A_coeff_m_for_VAR_1(inital_A_m,m,order):
    r"""
        Put VAR(p) into VAR(1) and get coefficient matrix of state equation.
        Parameters
        ----------
        inital_A_m
           description: initial A coeffs
           type: tensor 
           shape(m*m*p，)

        m
           description: the number of series
           type: int

        order
           description: order of VAR model
           type: int

        Returns
        -------
        coffs_m
           description: coeffs matrix of state equation.
           type: tensor
           shape: (m*p,m*p)
    """
    F_list=[]
    for c in range(order):
        F_list.append(inital_A_m[:,:,c])
    #concentrate
    F_temp1=torch.cat(F_list, dim=1)

    mp=m*order
    added_tensor= torch.eye((mp-m), mp)
    coffs_m=torch.cat((F_temp1, added_tensor), 0)
    return coffs_m


def make_var_cov_of_innovations_var1(lower_triang_params_form_lstm,m,order):
    r"""
        Put VAR(p) into VAR(1) and get var-cov matrix of innovations of state equation.
        Parameters
        ----------
        lower_triang_params_form_lstm
           description: elements for lower triangular matrix for generating var-cov matrix
           type: tensor 
           shape: ((m*(m+1)/2)，)

        m
           description: number of series
           type: int

        order
           description: order of VAR model
           type: int

        Returns
        -------
        var_cov_of_innovations_for_var1
           description: var-cov matrix of innovations of state euqation
           type: tensor
           shape: (m*p,m*p)
    """
    mp=m*order
    number_of_parms=m*(m+1)/2
    lower_t_matrix=torch.eye(m,m)
    count_temp=0
    for i in range(m):
        for j in range(i+1):
            lower_t_matrix[i,j]=lower_triang_params_form_lstm[count_temp]
            count_temp=count_temp+1

    var_cov_matrix = torch.mm(lower_t_matrix, lower_t_matrix.t())

    zeros_cols = torch.zeros([m, (mp - m)])
    #2.generate (mp-m)*mp matrix
    zeros_rows = torch.zeros([(mp - m),mp ])
    c = torch.cat((var_cov_matrix, zeros_cols), 1)
    var_cov_of_innovations_for_var1 = torch.cat((c, zeros_rows), 0)
    return (var_cov_of_innovations_for_var1)



def put_A_coeff_together(inital_A_m,order):
    r"""
        Put VAR coefficient matrices together.
        Parameters
        ----------
        inital_A_m
           description: initial A coeffs
           type: tensor 
           shape: (m，m，p)
        m
           description: number of series
           type: int

        order
           description: order of VAR model
           type: int

        Returns
        -------
        A
           description: coefficient matrix
           type: tensor
           shape: (m,m*p)

    """
    A_list=[]
    for c in range(order):
        A_list.append(inital_A_m[:,:,c])
    #concentrate
    A=torch.cat(A_list, dim=1)
    return A


def get_lagged_observations_detrended(target_y,trend_obs,current_t,order,m):

    r"""
        Get lagged observations of detrend series.
        Parameters
        ----------
        target_y
           description: observations of series
           type: tensor 
           shape: (T,m)

        trend_obs
           description: estimated trends
           type: tensor 
           shape: (T,batch=1,m)

        current_t
           description: time t
           type: int


        m
           description: number of series
           type: int

        order
           description: order of VAR model
           type: int

        Returns
        -------
        lagged_obs
           description: lagged observations
           type: tensor
           shape: (m*p,m)

    """   
    lagged_list=[]
    for i in range(order):
        lagged_list.append(target_y[(current_t-i-1),:].reshape(m, 1)-trend_obs[(current_t-i-1),0,:].reshape(m,1))
    lagged_obs = torch.cat(lagged_list, dim=0)
    return lagged_obs



import math

def make_var_covar_matrix(residual_parameters,m,order):
    r"""
        Make variance-covariance matrix for innovations of VAR(1).
        Parameters
        ----------
        residual_parameters
           description: residual parameters
           type: tensor 
           shape: ((m*(m+1)/2)，)
        m
           description: the number of series
           type: int

        order
           description: the order of VAR model
           type: int

        Returns
        -------
        var_cov_m
           description: variance-covariance matrix for innovations of VAR(1)
           type: tensor
           shape: (m*p,m*p)
    """  

    #generate void matrix
    mp=m*order
    number_of_parms=(m*(m+1))/2
    covariance_matrix=torch.eye(m,m)
    count_temp=0
    #make upper triangel matrix
    for i in range(m):
        for j in range(m-i):
            covariance_matrix[i,i+j]=residual_parameters[count_temp]
            count_temp=count_temp+1
    covariance_matrix_semi_positive = torch.mm(covariance_matrix.t(), covariance_matrix)
    zeros_cols = torch.zeros([m, (mp - m)])
    #2.generate (mp-m)*mp matrix
    zeros_rows = torch.zeros([(mp - m),mp ])
    c = torch.cat((covariance_matrix_semi_positive, zeros_cols), 1)
    var_cov_m = torch.cat((c, zeros_rows), 0)
    return (var_cov_m)
