import numpy as np

def check_var_causality(A_tensor):
    """
    Check if a VAR(p) process is stationary (causal) by evaluating the companion matrix.
    
    Parameters:
    A_tensor (np.ndarray): shape (p, k, k), coefficient matrices A1 to Ap
    
    Returns:
    bool: True if all eigenvalues of the companion matrix have modulus < 1 (i.e., causal/stationary)
    """
    p, k, _ = A_tensor.shape

    # Top block of the companion matrix
    top_row = np.concatenate(A_tensor, axis=1)  # shape: (k, k*p)

    if p == 1:
        Phi = top_row
    else:
        # Construct lower identity and zero block
        I = np.eye(k * (p - 1))
        Z = np.zeros((k * (p - 1), k))
        bottom = np.hstack([I, Z])  # shape: ((p-1)*k, p*k)
        Phi = np.vstack([top_row, bottom])  # shape: (p*k, p*k)

    print(Phi)

    eigvals = np.linalg.eigvals(Phi)
    print(np.abs(eigvals))
    return np.all(np.abs(eigvals) < 1)

# 定义 A1 和 A2
A1 = np.array([
    [-1.0842, -0.1245,  0.3137],
    [-0.7008, -0.3754, -0.2064],
    [ 0.3166,  0.3251,  0.2135]
])

A2 = np.array([
    [-0.5449, -0.3052, -0.1952],
    [-0.4057,  0.5129,  0.3655],
    [ 0.0054, -0.2911,  0.2066]
])

# 构造 VAR(2) 系数张量
A_tensor = np.stack([A1, A2], axis=0)

# 检查因果性
is_causal = check_var_causality(A_tensor)
print("Is the VAR(2) model causal?", is_causal)
