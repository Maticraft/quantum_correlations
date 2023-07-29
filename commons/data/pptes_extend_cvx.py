import cvxpy as cp
import numpy as np
import scipy.linalg as la
from math import log2, floor
from itertools import combinations
from qiskit.quantum_info import DensityMatrix
from cvxpy.expressions.expression import Expression    

from commons.metrics import partial_transpose_dense


def expr_as_np_array(cvx_expr):
    if cvx_expr.is_scalar():
        return np.array(cvx_expr)
    elif len(cvx_expr.shape) == 1:
        return np.array([v for v in cvx_expr])
    else:
        # then cvx_expr is a 2d array
        rows = []
        for i in range(cvx_expr.shape[0]):
            row = [cvx_expr[i,j] for j in range(cvx_expr.shape[1])]
            rows.append(row)
        arr = np.array(rows)
        return arr


def np_array_as_expr(np_arr):
    aslist = np_arr.tolist()
    expr = cp.bmat(aslist)
    return expr


def cp_ppt(rho, qargs):
    if not isinstance(rho, Expression):
        rho = cp.Constant(shape=rho.shape, value=rho)
    part_trans = cp_partial_transpose(rho, qargs)
    norm = cp.norm(part_trans, p = 'nuc')

    return (norm - 1) / 2


## Verified implementation from qutip
def cp_partial_transpose(rho, qargs):
    rho = expr_as_np_array(rho)
    num_qubits = int(log2(rho.shape[0]))
    dims = [2 for i in range(num_qubits)]

    # Inverted qubits indexes correction
    qargs = [num_qubits - q - 1 for q in qargs]

    """
    Based on Jonas' implementation using numpy.
    Very fast for dense problems.
    """
    mask = np.zeros(num_qubits, dtype=int)
    mask[qargs] = 1
    mask = mask.tolist()

    nsys = len(mask)
    pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
    pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                            [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])

    data = rho.reshape(np.array([dims, dims]).flatten()).transpose(pt_idx).reshape(rho.shape)

    return np_array_as_expr(data)


# Finding extension of the matrix with a SDP (semidefinite program)
def find_extension(rho, ext_qubits, threshold):
    N = int(log2(rho.shape[0]))
    ext_dim = 2**(ext_qubits - N)
    qubits = np.arange(N)

    combs = []
    for k in range(1, floor(N/2) + 1):
        qubits_for_trace = list(combinations(qubits, k))

        if k == N/2:
            qubits_for_trace = qubits_for_trace[:int(len(qubits_for_trace)/2)]

        for comb in qubits_for_trace:
            combs.append(list(comb))
        

    X = cp.Variable((ext_dim, ext_dim), complex=True)

    constraints = [cp.kron(rho, X) >> 0]
    constraints += [
        cp.trace(X) == 1
    ]
    constraints += [
        cp_ppt(cp.kron(rho, X), comb) <= 0.001 for comb in combs
    ]

    prob = cp.Problem(cp.Minimize(cp.abs(cp.trace(cp.kron(rho, X)) - 1)),
                  constraints)
    prob.solve()

    if prob.value < threshold:
            return np.kron(rho, X.value)
    else:
        return None


def find_extension_fast(rho, ext_qubits, threshold, biparts = None):
    N = int(log2(rho.shape[0]))
    std_dim = 2**N
    ext_dim = 2**ext_qubits
    extra_dim = 2**(ext_qubits - N)
    qubits = np.arange(ext_qubits)

    if type(biparts) == list:
        combs = biparts
    else:
        combs = []
        for k in range(1, floor(N/2) + 1):
            qubits_for_trace = list(combinations(qubits, k))

            if k == N/2:
                qubits_for_trace = qubits_for_trace[:int(len(qubits_for_trace)/2)]

            for comb in qubits_for_trace:
                combs.append(list(comb))
        

    X = cp.Variable((ext_dim, ext_dim), complex=True)

    constraints = [X >> 0]
    constraints += [
        cp_partial_transpose(X, comb) >> 0 for comb in combs
    ]
    constraints += [cp.trace(X) == 1]

    for x in range(std_dim):
        for y in range(std_dim):
            exp = 0
            for i in range(extra_dim):
                exp += X[x*extra_dim + i, y*extra_dim + i]
            constraints += [exp == rho[x, y]]

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    if prob.value < threshold:
            return X.value
    else:
        return None


def lambda_min(rho):
    N = int(log2(rho.shape[0]))
    qubits = np.arange(N)
    l_min = 1

    for k in range(1, floor(N/2) + 1):
        qubits_for_trace = list(combinations(qubits, k))

        if k == N/2:
            qubits_for_trace = qubits_for_trace[:int(len(qubits_for_trace)/2)]

        for comb in qubits_for_trace:
            rho_t = partial_transpose_dense(DensityMatrix(rho), list(comb))
            eigvals = np.real(la.eigvals(rho_t.data))
            eigval_min = np.min(eigvals[np.logical_or(eigvals > 0.0001, eigvals < 0.0001)])
            if eigval_min < l_min:
                l_min = eigval_min

    return l_min