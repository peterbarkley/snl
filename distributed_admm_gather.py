import sys
from sys import path
from pathlib import Path
# Sets parent directory as a string path
parent_dir = str(Path(__file__).resolve().parent.parent)
path.append(parent_dir)
from mpi4py import MPI
import numpy as np
from time import time
from datetime import datetime
from snl import snl_node, snl_node_psd

def solve_admm_dist(a, n, dx, aa, Ni, Na, warmstartprimal=None, alpha=1., itrs=1000, d=2, logging=True, verbose=False, check_termination=False):
    '''
    Solve the SNL problem using a distributed version of ADMM

    Args:    
        a (ndarray): |a| x d array of anchor point locations where a[0] gives the location of anchor 0.
        n (int): number of sensors (which is one half the number of resolvents)
        dx (dict): dictionary with the squared distances between sensors 
        where dx[i,j] gives the squared distance b/t sensor i and anchor j.
        aa (dict): dictionary with the constant aa[i,k] = da[i,k] - ||a[k]||^2
        Ni (dict): dictionary with the neighbors of each sensor where Ni[i] gives the neighbors of sensor i.
        Na (dict): dictionary with the anchors of each sensor where Na[i] gives the anchors of sensor i.
        warmstartprimal (ndarray) (optional): n+d x n+d ndarray for warm starting
        alpha (float): scaling parameter
        itrs (int): number of iterations
        d (int): dimension
        logging (bool): whether to log the results
        verbose (bool): whether to be verbose
        check_termination (bool): whether to check for termination

    Returns:
        xbar (ndarray): n+d x n+d array with result (averaged over all nodes)
        runningtime (float): time it took to run the algorithm
        update_norm (float): norm of the update
        itr (int): number of iterations
    '''
    assert(MPI.COMM_WORLD.Get_size() == 1)
    icomm = MPI.COMM_WORLD.Spawn(command=sys.executable,
                                 args=[__file__, 'child'],
                                 maxprocs=n)
    
    ws = warmstartprimal is not None
    
    icomm.bcast((n, d, a, aa, dx, alpha, itrs, ws, verbose, logging, check_termination), root=MPI.ROOT)
    if ws:
        icomm.bcast(warmstartprimal, root=MPI.ROOT)

    # Create neighborhoods - includes neighbors if they use 
    neighborhoods = [set(Ni[i]) for i in range(n)]
    for i in range(n):
        for j in Ni[i]:
            neighborhoods[j].add(i)
    for i in range(n):
        icomm.send((Ni[i], Na[i], neighborhoods[i]), dest=i)

    runningtime = time()
     
    xbar = icomm.recv(source=0) # this is the sum of the x values over all resolvents
    xbar /= n
    if check_termination:
        update_norm = icomm.recv(source=0)
        itr = icomm.recv(source=0)
    else:
        update_norm = None
        itr = itrs
    runningtime = time() - runningtime
    
    icomm.Disconnect()
    return xbar, runningtime, update_norm, itr

def main_child():
    icomm = MPI.Comm.Get_parent()
    assert icomm != MPI.COMM_NULL
    worker(icomm)
    icomm.Disconnect()

def worker(icomm):
    myrank = icomm.Get_rank()

    # Build intracommunicator
    intracomm = MPI.COMM_WORLD

    # Receive data from parent
    n, d, a, aa, dx, alpha, itrs, ws, verbose, logging, check_termination = icomm.bcast((), root=0)
    shape = (n+d, n+d)
    # Z stores consensus value
    if ws:
        Z = icomm.bcast((), root=0)
    else:
        Z = np.zeros(shape)
        Z[:d, :d] = np.eye(d)
    Zs = [Z, Z.copy()]
    Xs = [Z.copy(), Z.copy()] # store resolvent value
    As = [Z.copy(), Z.copy()] # store average value
    AX_old = [2*Z.copy(), 2*Z.copy()]
    Xsum = np.zeros(shape)

    # Get node specific data
    close_nodes, close_anchors, neighborhood = icomm.recv(source=0)
    neighbor_list = list(neighborhood)
    dist_graph_comm = intracomm.Create_dist_graph_adjacent(neighbor_list, neighbor_list)
    recv_buff = np.empty((len(neighbor_list), *shape), dtype=np.float64)
    neighbor_vals = {j:np.zeros(shape) for j in neighborhood}
    nsize = 1+2*len(neighborhood) # count includes other resolvent on sensor and both resolvents for each neighbor
    tau = alpha/nsize

    # Initialize proxes
    snl_function = snl_node(shape=shape, a=a, i=myrank, Ni=close_nodes, Na=close_anchors, d=d, dx=dx, aa=aa, logging=logging) 
    psd_function = snl_node_psd(shape=shape, i=myrank, Ni=close_nodes, d=d, logging=logging)

    # Main loop
    for itr in range(itrs):
        Xs[0] = snl_function.prox(Zs[0], tau=tau)
        Xs[1] = psd_function.prox(Zs[1], tau=tau)

        Xsum = Xs[0] + Xs[1]

        dist_graph_comm.Neighbor_allgather(sendbuf=Xsum, recvbuf=recv_buff)

        neighbor_sum = np.sum(recv_buff, axis=0)
        As[0] = (neighbor_sum + Xs[1])/nsize
        As[1] = (neighbor_sum + Xs[0])/nsize

        Zs[0] += As[0]
        Zs[0] -= 0.5*AX_old[0]
        AX_old[0] = As[0] + Xs[0]

        Zs[1] += As[1]
        Zs[1] -= 0.5*AX_old[1]
        AX_old[1] = As[1] + Xs[1]
        if check_termination and itr > 0 and itr % 20 == 0:
            update_norm = (np.linalg.norm(neighbor_sum/len(neighborhood) - Xs[0]) + np.linalg.norm(neighbor_sum/len(neighborhood) - Xs[1]))/(4*(3+len(neighborhood))**2)
            local_stop = update_norm < 1e-3
            global_stop = intracomm.allreduce(local_stop, op=MPI.LAND)
            if global_stop:
                break
    if logging:
        import json
        
        log = snl_function.log
        idx = myrank 
        with open('logs/' + str(idx) + '_dist_log.json', 'w') as f:
            json.dump(log, f)
        log = psd_function.log
        idx = myrank + n
        with open('logs/' + str(idx) + '_dist_log.json', 'w') as f:
            json.dump(log, f)
            
    xbar = np.zeros(shape)
    Xsum *= .5
    intracomm.Reduce([Xsum, MPI.DOUBLE], [xbar, MPI.DOUBLE], root=0)

    if myrank == 0: 
        icomm.send(xbar, dest=0)
        if check_termination:
            icomm.send(update_norm, dest=0)
            icomm.send(itr, dest=0)
            
if __name__ == '__main__':
    if 'child' in sys.argv:
        main_child()