#test
import numpy as np
from numpy.linalg import inv, cholesky

# var(p) to var(1), obtain big A matrix using Eq. (3.18) in my Phd thesis
def build_companion_matrix(A_list):
    p = len(A_list)
    k = A_list[0].shape[0]
    Ac = np.zeros((k * p, k * p))
    for i in range(p):
        Ac[0:k, i * k:(i + 1) * k] = A_list[i]
    if p > 1:
        Ac[k:, 0:(k * (p - 1))] = np.eye(k * (p - 1))
    return Ac

#compute Gamma(0),...,Gamma(p) using Eq. (3.19), (3.17) in my Phd thesis
def compute_acf(A_all, sigma_innov, k, p):
    kp = k * p
    A_list = [A_all[i] for i in range(p)]
    A = build_companion_matrix(A_list)
    sigma_star = np.zeros((kp, kp))
    sigma_star[:k, :k] = sigma_innov

    I_kpkp = np.eye(kp * kp)
    kron_A = np.kron(A, A)
    vec_sigma_u = sigma_star.reshape(-1, 1)
    vec_TY0 = np.linalg.solve(I_kpkp - kron_A, vec_sigma_u)

    Gamma_matrix = vec_TY0.reshape(kp, kp)

    initial_gamma = []
    for i in range(p):
        start = i * k
        end = (i + 1) * k
        initial_gamma.append(Gamma_matrix[0:k, start:end].T)
    #gamma(p)

    gamma_p_t=np.zeros((k, k))
    for i in range(p):
        #using (3.17) in my phd thesis
        gamma_p_t=gamma_p_t+A_list[i] @ (initial_gamma[p-1-i].T)

    initial_gamma.append(gamma_p_t.T)

    return initial_gamma

#compute PACF, P1,...,Pp using Eq. (3.25) to (3.28) in my Phd thesis 
def compute_pacf(Gamma, p):
    m = Gamma[0].shape[0]  # Gamma(0) is an (m x m) matrix

    A = np.zeros((m, m, p, p))        # Stores A_{s+1, i}
    A_star = np.zeros((m, m, p, p))   # Stores A^*_{s+1, i}
    Sigma = [None] * (p + 1)          # Covariance matrices Σ_s
    Sigma_star = [None] * (p + 1)     # Covariance matrices Σ_s^*
    L = [None] * (p + 1)              # Cholesky factors L_s of Σ_s
    L_star = [None] * (p + 1)         # Cholesky factors L_s^* of Σ_s^*
    P = [None] * p                    # P_{s+1} matrices

    # Initialization for s = 0
    Sigma[0] = Gamma[0]
    Sigma_star[0] = Gamma[0]
    L[0] = np.linalg.cholesky(Sigma[0])
    L_star[0] = np.linalg.cholesky(Sigma_star[0])

    for s in range(p):
        # Compute A_{s+1, s+1} using formula (3.25)
        temp = Gamma[s + 1].T.copy()
        for j in range(s):
            temp -= A[:, :, s - 1, j] @ Gamma[s - j].T
        A_diag = temp @ np.linalg.inv(Sigma_star[s])
        A[:, :, s, s] = A_diag  # Stores A_{s+1, s+1}

        # Compute A^*_{s+1, s+1} using formula (3.26)
        temp_star = Gamma[s + 1].copy()
        for j in range(s):
            temp_star -= A_star[:, :, s - 1, j] @ Gamma[s - j] 
        A_diag_star = temp_star @ np.linalg.inv(Sigma[s])
        A_star[:, :, s, s] = A_diag_star  # Stores A^*_{s+1, s+1}

        # Compute A_{s+1,i} and A^*_{s+1,i} for i = 1, ..., s using formula (3.26)
        for i in range(s):
            # Formula  A_{s+1,i} = A_{s,i} - A_{s+1,s+1} A^*_{s,s-i+1}
            A[:, :, s, i] = A[:, :, s - 1, i] - A_diag @ A_star[:, :, s - 1, s - i - 1] #subtracted 1 from s-i (JY)

            # Formula A^*_{s+1,i} = A^*_{s,i} - A^*_{s+1,s+1} A_{s,s-i+1}
            A_star[:, :, s, i] = A_star[:, :, s - 1, i] - A_diag_star @ A[:, :, s - 1, s - i - 1]  #subtracted 1 from s-i (JY)

        # Compute P_{s+1} using formula (3.27): P_{s+1} = L_s^{-1} A_{s+1,s+1} L_s^*
        Linv = np.linalg.inv(L[s])
        Lst = L_star[s]
        P[s] = Linv @ A_diag @ Lst        #removed .T from Lst.T (JY)

        # Compute Σ_{s+1} using formula (3.28)
        #Sigma_s1 = Gamma[0].copy()
        # for j in range(s + 1):
        #     Sigma_s1 -= A[:, :, s, j] @ Gamma[j + 1]
        Sigma[s + 1] = Sigma[s]- A[:, :, s, s]@ Sigma_star[s]@ A[:, :, s, s].T  #using (3.46)

        # Compute Σ^*_{s+1} using formula (3.28)
        # Sigma_s1_star = Gamma[0].copy()
        # for j in range(s + 1):
        #     Sigma_s1_star -= A_star[:, :, s, j] @ Gamma[j + 1].T
        Sigma_star[s + 1] =  Sigma_star[s]- A_star[:, :, s, s]@ Sigma[s]@ A_star[:, :, s, s].T  #using (3.46)

        # Cholesky decomposition of Σ_{s+1} and Σ^*_{s+1}
        L[s + 1] = np.linalg.cholesky(Sigma[s + 1])
        L_star[s + 1] = np.linalg.cholesky(Sigma_star[s + 1])

    return P

#do the inverse AK transformation
def inverse_AK_transform(P):
    p = len(P)
    d = P[0].shape[0]
    A_tilde = [None] * p  
    for i in range(p):
        B_i = np.linalg.inv(np.linalg.cholesky(np.eye(d) - P[i] @ P[i].T))
        A_tilde[i] = B_i @ P[i]
    return A_tilde

import numpy as np

# # 设置维度
# k = 3  # 变量数量
# p = 2  # 滞后阶数

# # 构造 VAR(2) 系数矩阵 A_all, shape = (p, k, k)
# A1 = np.array([
#     [0.4, 0.1, 0.0],
#     [0.0, 0.3, 0.2],
#     [0.1, 0.0, 0.5]
# ])

# A2 = np.array([
#     [0.2, 0.0, 0.1],
#     [0.0, 0.2, 0.1],
#     [0.1, 0.1, 0.2]
# ])

# A_all = np.stack([A1, A2])  # shape: (2, 3, 3)

# # 构造创新协方差矩阵, shape = (k, k)
# sigma_innov = np.array([
#     [1.0, 0.2, 0.1],
#     [0.2, 0.8, 0.3],
#     [0.1, 0.3, 0.6]
# ])
# # compute Gamma(0), ..., Gamma(p) 共 p+1 个 (m x m) 矩阵
# Gamma=compute_acf(A_all, sigma_innov, k, p)
# print("\ncovariance matrices")
# print(Gamma[0])
# print(" ")
# print(Gamma[1])
# print(" ")
# print(Gamma[2])

# #compute P1,...,Pp
# P = compute_pacf(Gamma, p)
# print("\npartial correlation matrices")
# print(P[0])
# print(" ")
# print(P[1])

# A_tilde=inverse_AK_transform(P)
# print("\nA_tilde")
# print(A_tilde[0])
# print(" ")
# print(A_tilde[1])

