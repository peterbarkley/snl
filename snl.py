import numpy as np
import cvxpy as cp
from time import time
from datetime import datetime
from scipy.linalg import cho_factor, cho_solve
from numba import jit, prange
from oars.matrices.core import getZfromGraph
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix
import json

def loadLogs(n, title= '_dist_log.json'):
    '''
    Load the log files of the two block distributed MPI version

    Args:
        n (int): number of compute nodes

    Returns:
        list: list of logs of 2n prox functions

    '''
    logs = []
    for node in range(2*n):
        logname = str(node)+title
        with open(logname) as f:
            data = json.load(f)
            logs.append(data)
    return logs

def _log(func):
    '''
    Logging wrapper for prox functions adding start and stop times to log along with current location estimate
    '''
    def wrapper(self, *args, **kwargs):
        if not self.logging:
            return func(self, *args, **kwargs)
        start = time()
        Y = func(self, *args, **kwargs)
        end = time()
        self.log.append((start, end) + tuple(self.estimate))
        return Y

    return wrapper

def getBlockIncidence(Ni):
    '''
    Returns a symmetrized incidence matrix based on the neighborhoods in Ni

    Args:
        Ni (list): list of list of integer node indexes which each node is neighbors with

    Returns:
        (ndarray): e x 2n numpy array with a row for each edge in the symmetrized communication graph
    '''
    n = len(Ni)
    M = []
    already_created = set([])
    for i in range(n):
        row = np.zeros(2*n)
        row[i] = 1
        row[n+i] = -1
        M.append(row)
        for j in Ni[i]:
            if j < i and (j, i) in already_created:
                continue

            row = np.zeros(2*n)
            row[i] = 1
            row[j+n] = -1
            M.append(row)
            row = np.zeros(2*n)
            row[j] = 1
            row[i+n] = -1
            M.append(row)
            already_created.add((min(i,j), max(i,j)))
    return np.array(M)

def getZfromNeighbors(Ni):
    '''
    Get the 2-Block graph Z matrix from the neighbors Ni and the 
    cutoff for the number of neighbors to return

    Args:
        Ni (list): list of lists with the neighbors within radius rd of sensor i, truncated to length cutoff

    Returns:
        Z (ndarray): 2n x 2n matrix with the neighbors of each sensor in the 2-block graph
    '''
    n = len(Ni)
    cutoff = max([len(Ni[i]) for i in range(n)])
    v = 2/(cutoff+1) # starting weight
    A = np.zeros((2*n, 2*n))
    for i in range(n):
        A[i, n+i] = v
        A[n+i, i] = v
        for j in Ni[i]:
            A[n+i, j] = v
            A[j, n+i] = v
            A[n+j, i] = v
            A[i, n+j] = v

    Z = getZfromGraph(A)
    return Z

def generateRandomData(n, m, d=2, rd=1, nf=0, cutoff=7, seed=0):
    """
    Generate randomize problem data for n points and m anchors inside [0, 1]^d

    Args:
        n (int): number of sensors
        m (int): number of anchors
        d (int): dimension
        rd (float): radius
        nf (float): noise factor
        cutoff (int): target number of neighbors to return
        seed (int): random seed

    Returns:
        a (ndarray): m x d array of anchor point locations
        x (ndarray): n x d array of sensor point locations
        da (dict): dictionary with the noisy squared distances between sensors and anchors 
        where da[i,k] gives the squared distance b/t sensor i and anchor k.
        dx (dict): dictionary with the noisy squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and sensor j.
        aa (dict): dictionary with the constant aa[i,k] = da[i,k] - ||a[k]||^2
        Ni (list): list of lists with the neighbors within radius rd of sensor i in Ni[i], truncated to cutoff
        Na (list): list of lists with the anchors within radius rd of sensor i in Na[i]
    """
    np.random.seed(seed)
    a = np.random.rand(m, d)
    x = np.random.rand(n, d)

    da = {}
    
    # Na is a list with n empty lists
    Na = [[] for i in range(n)]
    for i in range(n):
        Na[i] = []
        for k in range(m):
            dist = np.linalg.norm(x[i] - a[k])
            if dist < rd:
                dval = (dist*(1+nf*np.random.randn()))**2
                da[i, k] = dval
                Na[i].append(k)

    dx, Ni = getNeighbors(x, rd, nf, cutoff)
    
    aa = {(i, k): da[i, k] - np.linalg.norm(a[k])**2 for i in range(n) for k in Na[i]}
    return a, x, da, dx, aa, Ni, Na


def getNeighbors(x, rd=1, nf=0, cutoff=7):
    """
    Get the neighbors of each sensor within a radius rd

    Args:
        x (ndarray): n x d array of sensor point locations
        rd (float): radius
        nf (float): noise factor
        cutoff (int): number of neighbors to return

    Returns:
        dx (dict): dictionary where dx[i,j] gives the square of the noisy distance between i and j
        Ni (list): list of lists with the neighbors within radius rd of sensor i, truncated to length cutoff
    """
    n = len(x)
    dx = {}
    candidates = [set([]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(x[i] - x[j])
            if dist < rd:
                perturbed_dist = (dist*(1+nf*np.random.randn()))**2
                dx[i, j] = perturbed_dist
                dx[j, i] = perturbed_dist
                candidates[i].add(j)
                candidates[j].add(i)
    
    Ni = [sorted(np.random.choice(list(candidates[i]), min(len(candidates[i]), cutoff), replace=False)) for i in range(n)]
    return dx, Ni

def double_dual_update(Z, X, A, AX_old, n, neighborhoods, nsize): 
    for i in prange(n):
        # Update A_i
        A[i] = (np.sum([X[j]+X[n+j] for j in neighborhoods[i]], axis=0) + X[n+i])/nsize[i]

        # Update Z
        Z[i] = Z[i] + A[i] - 0.5*AX_old[i]
        AX_old[i] = A[i] + X[i]
        
        # Update A_{i+n}
        A[i+n] = (np.sum([X[j]+X[n+j] for j in neighborhoods[i]], axis=0) + X[i])/nsize[i]

        # Update Z
        Z[i+n] = Z[i+n] + A[i+n] - 0.5*AX_old[i+n] 
        AX_old[i+n] = A[i+n] + X[i+n]

def single_dual_update(Z, X, A, AX_old, n, neighborhoods, nsize): 
    # Update A
    for i in prange(n):
        A[i] = (np.sum([X[j+n] for j in neighborhoods[i]], axis=0) + X[n+i])/nsize[i]
        A[i+n] = (np.sum([X[j] for j in neighborhoods[i]], axis=0) + X[i])/nsize[i]

    # Update Z
    for i in prange(2*n):
        Z[i] = Z[i] + A[i] - 0.5*AX_old[i]
        AX_old[i] = A[i] + X[i]

def getSNLProxData(n, a, d, dx, aa, Ni, Na):
    nodeprox = [snl_node for _ in range(n)]
    nodedata = [{'a': a, 'd': d, 'dx': dx, 'aa': aa, 'i': i, 'Ni': Ni[i], 'Na': Na[i], 'shape': (n+d, n+d), 'rho': 1, 'logging':True, 'factor':1.01, 'multiplier':1.0} for i in range(n)]

    psdProx = [snl_node_psd for _ in range(n)]
    psdData = [{'shape': (n+d, n+d), 'd': d, 'i': i, 'Ni': Ni[i], 'logging':True} for i in range(n)]

    proxlist = psdProx + nodeprox
    data = psdData + nodedata

    return data, proxlist

def solve_admm_double(a, n, dx, aa, Ni, Na, double=True, warmstartprimal=None, alpha=1.0, d=2, itrs=100, verbose=False):
    """
    Solve the SNL problem using ADMM over the graph defined by Ni
    each node finds  :math:`X_i^{k+1} = \\argmin_{X_i}\\frac{\\alpha}{|N_i|} (\\sum_{j \\in N_i} |d_{ij} - x_{ii} - x_{jj} + 2x_{ij}| + \\sum_{k \\in N_a} |d_{ik} - x_{ii} + 2a_k x_{i}|) + 0.5||X_i - Z^k_i||_F^2`
    subject to :math:`X_{:d, :d} = I` and :math:`X` is PSD, and X is a subset of the full matrix over i and j in N_i
    where :math:`d_{ij} = ||x_i - x_j||^2`, :math:`d_{ik} = ||x_i - a_k||^2 - ||a_k||^2` and :math:`N_i` is the set of neighbors of node i

    Then each node sends X_i to its neighbors and receives X_j from its neighbors
    Then each node finds a_i^{k+1} = \\frac{1}{|N_i|}\\sum_{j \\in N_i} X_j^{k+1} (averaging over X_j)
    Then each node finds Z_i^{k+1} = Z_i^k + a_i^{k+1} - 0.5a_i^k - 0.5X_i^k

    Args:
        a (ndarray): |a| x d array of anchor point locations where a[0] gives the location of anchor 0.
        n (int): number of sensors
        dx (dict): dictionary with the squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and anchor j.
        aa (dict): dictionary with the constant aa[i,k] = da[i,k] - ||a[k]||^2
        Ni (dict): dictionary with the neighbors of each sensor where Ni[i] gives the neighbors of sensor i.
        Na (dict): dictionary with the anchors of each sensor where Na[i] gives the anchors of sensor i.
        double (bool): whether to use the communciation matrix [[A, A+I; A+I, A]] if true or else [[0, A+I; A+I; 0]]
        warmstartprimal (ndarray) (optional): n+d x n+d ndarray for warm starting
        alpha (float): scaling parameter
        d (int): dimension
        itrs (int): number of iterations
        verbose (bool): whether to be verbose

    Returns:
        X (ndarray): n+d x n+d array with result (averaged over all nodes)
        log (list): length 2n list of length itrs lists giving the start and stop time and node position estimate at each iteration for each prox function
        all_X (ndarray): (2*n, n+d, n+d) array of results at each node

    """
    if warmstartprimal is not None:
        Z = np.array([warmstartprimal.copy() for i in range(2*n)])
    else:
        Z = np.zeros((2*n, n+d, n+d)) # Initialize Z, the consensus variable
        for i in range(2*n):
            Z[i][:d, :d] = np.eye(d)
    X = Z.copy() # Initialize X, the variable for storing the resolvents
    A = Z.copy() # Initialize A, the variable for storing the average of the neighbors
    AX_old = X + A # Initialize A_old, the variable for storing the previous average of the neighbors
    neighborhoods = [set(Ni[i]) for i in range(n)]
    for i in range(n):
        for j in Ni[i]:
            neighborhoods[j].add(i)
    if double:
        nsize = [1+2*len(neighborhoods[i]) for i in range(n)]
    else:
        nsize = [1+len(neighborhoods[i]) for i in range(n)]

    # Initialize the cvxpy proximal operators
    snl_nodes = [snl_node(shape=(n+d, n+d), a=a, i=i, Ni=Ni[i], Na=Na[i], d=d, dx=dx, aa=aa) for i in range(n)]
    psd_nodes = [snl_node_psd(shape=(n+d, n+d), i=i, Ni=Ni[i], d=d) for i in range(n)]
    cvx_nodes = psd_nodes + snl_nodes
    log = [[] for i in range(2*n)]
    itr_period = max(1, itrs//10)

    # Main Loop
    for itr in range(itrs):
        start = time()
        if verbose and itr%itr_period==0: 
            xbar = np.mean(X, axis=0)
            xdiff = np.linalg.norm(X - xbar)
            print(f'{datetime.now()} {itr} xdiff {xdiff}')

        # This can run in parallel
        # Find the proximal value for each node        
        for i in range(n):
            X[i] = cvx_nodes[i].prox(Z[i], tau=alpha/nsize[i])
            X[i+n] = cvx_nodes[i+n].prox(Z[i+n], tau=alpha/nsize[i])


        # This can run in parallel
        # Find the average of the neighbors for each node
        # and update Z
        if double:
            double_dual_update(Z, X, A, AX_old, n, neighborhoods, nsize)
        else:
            single_dual_update(Z, X, A, AX_old, n, neighborhoods, nsize)
        
        # Logging
        stop = time()
        for i in range(2*n):
            log_entry = (start, stop) + tuple(X[i][(i%n)+d, :d])
            log[i].append(log_entry)
    return np.mean(X, axis=0), log, X

def node_objective_value(Z, dx, i, Ni, Na, d, a, aa):
    '''
    Compute the objective function value for the SNL node problem
    for a specific node i using the full X matrix

    Args:
        Z (ndarray): (d+n) x (d+n) array of sensor locations
        dx (dict): dictionary with the squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and anchor j.
        i (int): sensor index
        Ni (list): list of neighbors
        Na (list): list of anchors
        d (int): dimension
        a (ndarray): |a| x d array of anchor point locations where a[0] gives the location of anchor 0.
        aa (dict): dictionary with the constant aa[i,k] = da[i,k] - ||a[k]||^2

    Returns:
        float: objective function value
    '''
    idx = i+d
    val = 0
    for j in Ni:
        val += np.abs(dx[i,j] - Z[idx,idx] - Z[j + d, j + d] + 2*Z[j + d, idx])

    for k in Na:
        val += np.abs(aa[i,k] - Z[idx,idx] + 2*a[k]@Z[idx, :d])
    return val

def recoverZ(Xvars, n, d, Ni):
    '''
    Recover the (d+n) x (d+n) matrix Z from the Fusion variables Xvars
    '''
    Z = np.zeros((d+n, d+n))
    Z[:d, :d] = np.eye(d)
    for i in range(n):
        for k in range(d):
            Z[i+d, k] = Xvars[i].index(d,k).level()[0]
            Z[k, i+d] = Xvars[i].index(d,k).level()[0]
        Z[i+d, i+d] = Xvars[i].index(d,d).level()[0]
        for jdx, j in enumerate(Ni[i]):
            Z[i+d, j+d] = Xvars[i].index(d,d+1+jdx).level()[0]
            Z[j+d, i+d] = Xvars[i].index(d,d+1+jdx).level()[0]

    return Z

def arrayIndextoListIndex(i, j, n):
    '''
    Convert an index from an array to a list index
    '''

    r = min(i, j)
    c = max(i, j)
    return r*n + c - r*(r+1)//2

def solve_snl_fusion(a, n, aa, dx, Ni, Na, d=2):
    '''
    Solve the SNL problem using MOSEK Fusion API.

    Args:
        a (ndarray): |a| x d array of anchor point locations where a[k] gives the location of anchor k.
        n (int): number of sensors
        aa (dict): dictionary with the squared distances between sensors and anchors less the squared norm of the anchor
        where aa[i,k] gives the value for sensor i and anchor k.
        dx (dict): dictionary with the squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and anchor j.
        d (int): dimension of the space (default 2)

    Returns:
        Z (ndarray): (d+n) x (d+n) symmetric array [I, X^T; X, Y] where X approximates sensor locations and Y approximates X @ X^T
        value (float): optimal value of the problem
    '''
    M = Model("sdo2")

    # Setting up the variables
    Xvar = M.variable("X", [d, n], Domain.unbounded())
    Yvar = M.variable("Y", n*(n+1)//2, Domain.unbounded())
    sizes = [d+1+len(Ni[i]) for i in range(n)]
    Xvars = []
    numSensorDevs = sum(len(Ni[i]) for i in range(n))
    numAnchorDevs = sum(len(Na[i]) for i in range(n))
    for i in range(n):

        Xvars.append(M.variable("Z" + str(i), Domain.inPSDCone(sizes[i])))
        uvar = M.variable("u" + str(i), numSensorDevs, Domain.greaterThan(0.0))
        vvar = M.variable("v" + str(i), numSensorDevs, Domain.greaterThan(0.0))
        wvar = M.variable("w" + str(i), numAnchorDevs, Domain.greaterThan(0.0))
        yvar = M.variable("y" + str(i), numAnchorDevs, Domain.greaterThan(0.0))

    # Setting up constant coefficient matrices
    I = Matrix.eye(d)
    A = {}
    L = {}
    for i in range(n):
        for jdx, j in enumerate(Ni[i]):
            aij = np.zeros((sizes[i], sizes[i]))
            aij[d,d] = -1
            aij[d,d+1+jdx] = 1
            aij[d+1+jdx,d] = 1
            aij[d+1+jdx,d+1+jdx] = -1
            A[i,j] = Matrix.dense(aij)
        for kdx, k in enumerate(Na[i]):
            aik = np.zeros((sizes[i], sizes[i]))
            aik[d,d] = -1
            aik[d,:d] = a[k]
            aik[:d,d] = a[k]
            L[i,k] = Matrix.dense(aik)
        

    # Objective
    M.objective(ObjectiveSense.Minimize, Expr.add(Expr.add(Expr.sum(uvar), Expr.sum(vvar)), Expr.add(Expr.sum(wvar), Expr.sum(yvar))))

    # Constraints
    s_idx = 0
    a_idx = 0
    for i in range(n):
        M.constraint("c" + str(i), Xvars[i].slice([0,0],[d, d]), Domain.equalsTo(I))
        M.constraint("cd" + str(i) + '_' + str(i), Expr.sub(Xvars[i].index(d,d), Yvar.index(arrayIndextoListIndex(i,i,n))), Domain.equalsTo(0.0))
        for k in range(d):
            M.constraint("ct" + str(i) + '_' + str(i) + '_' + str(k), Expr.sub(Xvars[i].index(k, d), Xvar.index(k, i)), Domain.equalsTo(0.0))
        for jdx, j in enumerate(Ni[i]):
            # Set the absolute value variables
            M.constraint("cs" + str(i) + '_' + str(j), Expr.add(Expr.sub(uvar.index(s_idx), vvar.index(s_idx)), Expr.dot(A[i,j], Xvars[i])), Domain.equalsTo(- dx[i,j]))
            s_idx += 1

            # Ensure that the entries for j in Xvars[i] match the entries for j in Xvars[j]
            # The entries for j in Xvars[i] are in the d+1+jdx row and column
            # The entries for j in Xvars[j] are in the d row and column
            # diagonal entry
            M.constraint("cd" + str(i) + '_' + str(j), Expr.sub(Xvars[i].index(d+1+jdx, d+1+jdx), Yvar.index(arrayIndextoListIndex(j,j,n))), Domain.equalsTo(0.0))

            # top d entries
            for k in range(d):
                M.constraint("ct" + str(i) + '_' + str(j) + '_' + str(k), Expr.sub(Xvars[i].index(k, d+1+jdx), Xvar.index(k, j)), Domain.equalsTo(0.0))

            # ij entry
            M.constraint("ci" + str(i) + '_' + str(j), Expr.sub(Xvars[i].index(d, d+1+jdx), Yvar.index(arrayIndextoListIndex(i,j,n))), Domain.equalsTo(0.0))

            # off diagonal entries
            for kdx in range(jdx):
                k = Ni[i][kdx]
                M.constraint("co" + str(i) + '_' + str(j) + '_' + str(k), Expr.sub(Xvars[i].index(d+1+jdx, d+1+kdx), Yvar.index(arrayIndextoListIndex(j,k,n))), Domain.equalsTo(0.0))

        for kdx, k in enumerate(Na[i]):
            M.constraint("ca" + str(i) + '_' + str(k), Expr.add(Expr.sub(wvar.index(a_idx), yvar.index(a_idx)), Expr.dot(L[i,k], Xvars[i])), Domain.equalsTo(- aa[i,k]))
            a_idx += 1
    M.solve()

    return recoverZ(Xvars, n, d, Ni), M.primalObjValue()

def getVectorizationMatrix(a, n, Ni, Na, d=2):
    '''
    Get the matrix that vectorizes the matrix elements for the SNL problem
    such that for a (|Ni| + |Na|) by (n+d)^2 matrix A and x = vec(X) in R^{(n+d)^2}, we have
    Ax = [x_ii + x_jj - 2x_ij for i in n, for j in Ni[i]] + [x_ii - 2 a_j x_:d for i in n, for j in Na[i]].

    Args:
        a (ndarray): |a| x d array of anchor point locations where a[0] gives the location of anchor 0.
        n (int): number of sensors
        Ni (dict): dictionary with the neighbors of each sensor where Ni[i] gives the neighbors of sensor i.
        Na (dict): dictionary with the anchors of each sensor where Na[i] gives the anchors of sensor i.
        d (int): dimension

    Returns:
        A (ndarray): (|Ni| + |Na|) x (n+d)^2 matrix that vectorizes the matrix elements
    '''
    # Get the number of neighbors and anchors
    numNeighbors = sum([len(Ni[i]) for i in range(n)])
    numAnchors = sum([len(Na[i]) for i in range(n)])

    # Initialize the matrix
    A = np.zeros((numNeighbors + numAnchors, (n+d)**2))

    # Fill in the matrix
    row = 0
    for i in range(n):
        iidx = (d+i)*(n+d) + d + i
        for j in Ni[i]:
            A[row, iidx] = 1
            A[row, (d+j)*(n+d) + d + j] = 1
            A[row, iidx + j - i] = -2
            row += 1

    for i in range(n):
        
        iidx = (d+i)*(n+d) + d + i
        istart = (d+i)*(n+d)
        for j in Na[i]:
            A[row, iidx] = 1
            A[row, istart:istart+d] = -2*a[j]
            row += 1

    return A

def solve_snl_vec(a, n, aa, dx, Ni, Na, splitPSD=True, d=2, solverargs={}, verbose=False):
    '''
    Solve the SNL problem using cvxpy

    Args:
        a (ndarray): |a| x d array of anchor point locations where a[0] gives the location of anchor 0.
        n (int): number of sensors
        da (dict): dictionary with the squared distances between sensors and anchors 
        where da[i,k] gives the squared distance b/t sensor i and anchor k.
        dx (dict): dictionary with the squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and anchor j.
        d (int): dimension

    Returns:
        x (ndarray): n x d array of sensor locations
    '''

    # Variables
    if splitPSD:
        Z = cp.Variable((n+d, n+d), symmetric=True)
    else:
        Z = cp.Variable((n+d, n+d), PSD=True)

    A = getVectorizationMatrix(a, n, Ni, Na, d)
    k = A.shape[0]
    dvec = [dx[i,j] for i in range(n) for j in Ni[i]] + [aa[i,j] for i in range(n) for j in Na[i]]
    dvec = np.array(dvec)
    assert len(dvec) == k
    objective = cp.Minimize(cp.norm(dvec - A @ cp.vec(Z), 1))

    # Constraints
    constraints = [Z[:d, :d] == np.eye(d)]
    if splitPSD:
        for i in range(n):
            idxs = list(range(d)) + [i+d] + [d+j for j in Ni[i]]
            constraints.append(Z[np.ix_(idxs, idxs)] >> 0)

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(**solverargs)

    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)

    # Extract the solution
    return Z.value, prob.value

def getSTS(d, ni):
    """
    Returns the matrix S^T S
    where S is the scaling matrix for the vectorized submatrix
    0.5||X||_F^2 => 0.5||Sx||^2
    for x = (x_ii, x_ij, x_jj, x_ij', x_j'j' ... x_i1, ..., x_id)
    the scaling for the diagonal elements is 1 and for the off-diagonal elements is 2

    Args:
        d (int): dimension
        ni (list): list of neighbors

    Returns:
        ndarray: S^T S
    """
    m = 1 + d + 2*len(ni)
    S = np.eye(m)*2
    for i in range(len(ni) + 1):
        S[i*2, i*2] = 1
    return S
   
def admm_firststep(y, b, w, cho_factor, rhoBT):    
    """
    L2 prox

    min 0.5||Sx||^2 + rho/2||y + Bx - b - w||^2

    x = rho*(S^T S + rho B^T B)^-1 B^T(w + b - y)
    
    Args:
        y (array): primal variable
        b (array): constant
        w (array): dual variable
        cho_factor (tuple): cholesky factorization of the matrix (rho*B^T B + S^T S)
        rhoBT (array): precomputed rho B^T

    Returns:
        array: primal variable x
    """
    d = cho_solve(cho_factor, rhoBT @ (w + b - y))
    return d

@jit
def admm_secondstep(b, B, x, w, tau, rho):
    """
    L1 prox
    min_y tau||y||_1 + rho/2||y + Bx^k - b - w^k||^2

    Args:
        b (array): constant
        B (array): matrix
        x (array): primal variable from first step
        w (array): dual variable from previous iteration
        tau (float): splitting parameter
        rho (float): ADMM parameter
    """
    d = b - B@x + w
    return np.sign(d)*np.maximum(0, np.abs(d) - tau/rho)

class snl_node():
    """
    Prox class for the SNL node objective and top Z block identity constraint
    """
    def __init__(self, shape, i, dx, a, aa, Ni, Na, d=2, rho=1.0, tol=1e-3, maxiter=100, multiplier=0.5, factor=1.01, logging=False, **kwargs):
        self.approx = True
        self.d = d 
        self.shape = shape 
        self.i = i 
        self.dx = dx 
        self.aa = aa 
        self.Ni = Ni 
        self.Na = Na 
        self.tol = tol 
        self.maxiter = maxiter 
        self.iteration = 0 # iteration counter
        self.a = a 
        self.multiplier = multiplier 
        self.B = getB(self.a, self.Ni, self.Na, self.d, self.multiplier) # matrix B s.t. ||b-Bx||_1 = \\sum |b_i - X_ii - X_jj + 2X_ij| + \\sum|b_i - X_ii + 2a_k X_i|
        self.rho = rho # ADMM prox parameter
        self.rhoBT = self.rho*self.B.T # precomputed rho* B^T
        self.logging = logging
        if self.logging: self.log = []

        # cholesky factorization of the matrix rho*(B^T B) + S^T S
        self.cho_factor = cho_factor(getSTS(self.d, self.Ni) + self.rho*self.B.T@self.B)
        self.n = self.B.shape[0] # number of rows in B (|Ni| + |Na|)
        self.w = np.zeros(self.n) # dual variable for ADMM
        self.x = np.zeros(self.B.shape[1]) # primal variable for ADMM, (x_ii, x_ij, x_jj, x_ij', x_j'j' ... x_i1, ..., x_id)
        self.y = np.zeros(self.n) # primal variable for ADMM
        self.b = np.zeros(self.n) # constant for ADMM s.t. absolute values can be written as ||b-Bx||_1
        self.xk = np.zeros(self.B.shape[1]) # constant for ADMM with the same shape as x for storing input to the prox from other nodes
        self.ii = (self.i + self.d, self.i + self.d) # diagonal index for the sensor
        self.Y = np.zeros(self.shape) # matrix for storing the output of the prox
        self.factor = factor #data.get('factor', 1.01) # factor for the tolerance decay

        self.idxs = list(range(self.d)) + [self.i+self.d] + [j + self.d for j in self.Ni]
        # precompute the base b vector
        self.bb = np.concatenate([self.multiplier*np.array([self.dx[self.i,j] for j in self.Ni]), np.array([self.aa[self.i,k] for k in self.Na])])

    def objective(self, X):
        '''
        Compute the objective function value for the SNL node problem (used for logging)
        f(X) = \\sum_{j \\in N_i} |d_{ij} - x_{ii} - x_{jj} + 2x_{ij}| + \\sum_{k \\in N_a} |d_{ik} - x_{ii} + 2a_k x_{i}|

        Args:
            X (ndarray): n x n matrix of sensor locations

        Returns:
            float: objective function value
        '''
        val = 0
        for j in self.Ni:
            val += np.abs(self.dx[self.i,j] - X[self.ii] - X[j + self.d, j + self.d] + 2*X[j + self.d, self.i + self.d])

        for k in self.Na:
            val += np.abs(self.aa[self.i,k] - X[self.ii] + 2*self.a[k]@X[self.i + self.d, :self.d])

        # val = np.linalg.norm(self.bb - self.B@self.xk, 1)
        return val

    def getTol(self):
        return max(1/(self.iteration**self.factor), 1e-7)

    def admm(self, xk, tau, tol):

        self.b = self.bb - self.B@xk
        
        # Main loop
        for itr in range(self.maxiter):
            self.x = admm_firststep(self.y, self.b, self.w, self.cho_factor, self.rhoBT)
            self.y = admm_secondstep(self.b, self.B, self.x, self.w, tau, self.rho)
            delta = self.b - self.y - self.B@self.x
            self.w = self.w + delta
            dd = np.linalg.norm(delta)
            if dd < tol:
                break

        return self.x

    def vectorize(self, X):
        self.xk[0] = X[self.ii]
        for jdx, j in enumerate(self.Ni):
            self.xk[2*jdx + 1] = X[self.i + self.d, j + self.d]
            self.xk[2*(jdx + 1)] = X[j + self.d, j + self.d]
        self.xk[2*len(self.Ni) + 1:] = X[self.i + self.d, :self.d]

    def unvectorize(self, X):
        self.Y = X.copy()
        self.Y[self.ii] += self.xk[0]
        for jdx, j in enumerate(self.Ni):
            self.Y[self.i + self.d, j + self.d] += self.xk[2*jdx + 1]
            self.Y[j + self.d, j + self.d] += self.xk[2*(jdx + 1)]
            self.Y[j + self.d, self.i + self.d] += self.xk[2*jdx + 1]
            
        self.Y[self.i + self.d, :self.d] += self.xk[2*len(self.Ni) + 1:]
        self.Y[:self.d, self.i + self.d] += self.xk[2*len(self.Ni) + 1:]

    @_log
    def prox(self, X, tau=1, tol=None):
        """
        Proximal operator for the SNL node objective function
        still need to implement tau
        """
        self.iteration += 1
        self.vectorize(X) # vectorize the input matrix X into xk
        if tol is None: tol = self.getTol()
        self.xk = self.admm(self.xk, tau, tol) # solve the ADMM problem
        
        # Set the values
        self.unvectorize(X) # unvectorize the output xk into the output matrix Y

        # Update the diagonal elements in the reference block back to 1
        self.Y[:self.d, :self.d] = np.eye(self.d)

        # Log estimate
        self.estimate = self.Y[self.i + self.d, :self.d]
        return self.Y

class snl_node_psd():
    """
    Prox class for the SNL node PSD constraint
    """
    def __init__(self, shape, i, Ni, d=2, logging=False, **kwargs):
        self.shape = shape 
        self.i = i 
        self.d = d 
        self.Ni = Ni 
        self.logging = logging 
        if self.logging: self.log = []
        self.Z = np.zeros((self.d + len(self.Ni) + 1, self.d + len(self.Ni) + 1))
        self.ii = (self.i + self.d, self.i + self.d)
        self.idxs = list(range(self.d)) + [self.i+self.d] + [j + self.d for j in self.Ni]
        self.Y = np.zeros(self.shape)

    def objective(self, X):
        return 0 #self.obj

    @_log
    def prox(self, X, tau=1.0, tol=None):
        self.Z = X[np.ix_(self.idxs, self.idxs)]
        
        # PSD projection
        eig, eigv = np.linalg.eigh(self.Z)
        poseigs = eig > 0
        U = eigv[:, poseigs]
        self.Z = U @ np.diag(eig[poseigs]) @ U.T
        self.Y = X.copy()
        self.Y[np.ix_(self.idxs, self.idxs)] = self.Z

        # Log estimate
        self.estimate = self.Y[self.i + self.d, :self.d]
        return self.Y

def getB(a, Ni, Na=None, d=2, multiplier=0.5):
    '''
    Return the matrix B for the SNL node objective function
    B is a matrix of size (m, 1 + d + 2*len(Ni)) where m = len(Ni) + len(Na)
    for the SNL node objective function \\sum_{j \\in Ni} |d_{ij} - x_{ii} - x_{jj} + 2x_{ij}| + \\sum_{k \\in Na} |aa_{ik} - x_{ii} + 2a_k x_{i}|,
    this will be equivalent to ||b - Bx||_1 where b is the vector of size m, b = (d_ij for j in Ni, aa_ik for k in Na)
    the first column of B corresponds to x_ii, the next 2*len(Ni) columns correspond to x_{ij} and x_{jj} for j in Ni
    the last d columns correspond to x_{i} for k in Na

    Args:
        a (array): matrix of size (n, d) where n is the number of nodes and d is the dimension of the reference block
        Ni (list): list of neighbors of node i
        Na (list): list of anchors for node i 
        d (int): dimension of the reference block
        multiplier (float): multiplier for the Ni block

    Returns:
        array: matrix B
    '''
    if Na is None:
        Na = list(range(len(a)))
    m = len(Na) + len(Ni)

    B = np.zeros((m, 1 + 2*len(Ni) + d))
    B[:len(Ni),0] = multiplier*np.ones(len(Ni))
    B[len(Ni):,0] = np.ones(len(Na))
    for i in range(len(Ni)):
        B[i,2*i+1:2*i+3] = multiplier*np.array([-2, 1])
    for i in range(len(Ni), len(Ni) + len(Na)):
        k = Na[i - len(Ni)]
        # print(k, a[k, :], A[i,2*len(Ni)+1:2*len(Ni)+1+d])
        B[i,2*len(Ni)+1:2*len(Ni)+1+d] = -2*a[k,:]

    return B
 