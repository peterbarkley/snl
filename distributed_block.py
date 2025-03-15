
import sys
from mpi4py import MPI
import numpy as np
from time import time
from datetime import datetime
from oars.matrices import getTwoBlockSimilar

def distributed_block_sparse_solve(n, data, resolvents, warmstartprimal, Z=None, itrs=1001, gamma=0.9, alpha=1.0, logging=False, verbose=False):
    '''
    Solve the 2-Block resolvent splitting algorithm in parallel using MPI

    Args:
        n (int): Number of resolvents
        data (list): List of data for each resolvent
        resolvents (list): List of resolvents
        warmstartprimal (ndarray): Warm start for the primal variables
        Z (ndarray): Weight matrix
        itrs (int): Number of iterations
        gamma (float): Step size
        alpha (float): Proximal parameter
        verbose (bool): Print verbose output

    Returns:
        x_bar (ndarray): The average of the primal variables
        results (list): Not implemented (will be list of dictionaries with the results for each resolvent)

    '''
    assert(MPI.COMM_WORLD.Get_size() == 1)
    assert(n % 2 == 0)
    icomm = MPI.COMM_WORLD.Spawn(command=sys.executable,
                                 args=[__file__, 'child'],
                                 maxprocs=n//2)

    # Send data to workers
    if Z is None:
        Z, _ = getTwoBlockSimilar(n)

    icomm.bcast((n, Z, gamma, alpha, itrs, logging, warmstartprimal, verbose), root=MPI.ROOT)
    
    for i in range(n//2):
        j = i+n//2
        icomm.send((data[i], data[j], resolvents[i], resolvents[j]), dest=i)

    if verbose:print(datetime.now(), 'Data sent to workers', flush=True)
        
    x_bar = icomm.recv(source=0)
    results = icomm.recv(source=0)
    icomm.Disconnect()
    return x_bar, results

def main_child():
    icomm = MPI.Comm.Get_parent()
    assert icomm != MPI.COMM_NULL
    worker(icomm)
    icomm.Disconnect()

def buildComms(icomm, myrank, Z, zerotol=1e-7):
    '''
    Builds communicators for myrank as required by Z
    '''
    n = Z.shape[0]//2
    Ni = [[] for _ in range(n)]
    Nj = [[] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if r != c and not np.isclose(Z[n+r][c],0.0, atol=zerotol):
                Ni[r].append(c)
                Nj[c].append(r)
                
    igroups = [icomm.group.Incl([i] + Ni[i]) for i in range(n)]
    jgroups = [icomm.group.Incl([i] + Nj[i]) for i in range(n)]

    leftcomms = [icomm.Create_group(group) for group in igroups]
    rightcomms = [icomm.Create_group(group) for group in jgroups]

    myleftdeps = [(leftcomms[j], -Z[n+j, myrank], j) for j in Nj[myrank]]
    myrightdeps = [(rightcomms[j], -Z[j, myrank+n], j) for j in Ni[myrank]]
    return leftcomms[myrank], rightcomms[myrank], myleftdeps, myrightdeps


def worker(icomm):
    myrank = icomm.Get_rank()

    # Build intracommunicator
    intracomm = MPI.COMM_WORLD

    # Receive data from parent
    n, Z, gamma, alpha, itrs, logging, warmstartprimal, verbose = icomm.bcast((), root=0)
    first_data, second_data, first_resolvent, second_resolvent = icomm.recv(source=0)

    my_left_comm, my_right_comm, my_left_deps, my_right_deps = buildComms(intracomm, myrank, Z)

    v = [warmstartprimal, -warmstartprimal.copy()]

    res = [first_resolvent(**first_data), second_resolvent(**second_data)]
    if logging:
        res[0].logging = True
        res[1].logging = True
    shape = warmstartprimal.shape
    sum_x = np.zeros(shape)
    sum_y = np.zeros(shape)
    itr_period = max(1, itrs // 10)
    t = time()
    myrankshift = n//2 + myrank
    reqs = {}
    if verbose and (myrank == 0 or myrank == n//2 - 1):
        print(datetime.now(), 'Worker', myrank, 'started', flush=True)
    if verbose and myrank == 0:
        print('date\t time\t\trank iter ||delta v||')
    for itr in range(itrs):

        # First block
        my_x = res[0].prox(v[0], alpha)
        for comm, wt, j in my_left_deps:
            if j in reqs:
                reqs[j].Wait()
            reqs[j] = comm.Ireduce([my_x*wt, MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
        req = my_left_comm.Ireduce([-my_x*Z[myrankshift, myrank], MPI.DOUBLE], [sum_x, MPI.DOUBLE], op=MPI.SUM)
        
        # Wait for req
        req.Wait()

        # Second block
        my_y = res[1].prox(v[1]+sum_x, alpha)
        for comm, wt, j in my_right_deps:
            if j in reqs:
                reqs[j].Wait()
            reqs[j] = comm.Ireduce([my_y*wt, MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)
        req = my_right_comm.Ireduce([-my_y*Z[myrank, myrankshift], MPI.DOUBLE], [sum_y, MPI.DOUBLE], op=MPI.SUM)
        update_1 = gamma*(2*my_y - sum_x)
        v[1] = v[1] - update_1
        
        # Wait for sum_y
        req.Wait()
        update_0 = gamma*(2*my_x - sum_y)

        v[0] = v[0] - update_0

        if verbose and myrank == 0 and itr % itr_period == 0:
            v_sq = np.linalg.norm(update_0)**2 + np.linalg.norm(update_1)**2
            print(datetime.now(), myrank, itr, np.round(v_sq**0.5,3), flush=True)

    if verbose and myrank == 0:
        print(datetime.now(), 'Worker', myrank, 'finished, time', time()-t, flush=True)

    result = [{'first_x': my_x, 'second_x': my_y, 'first_v0': v[0], 'second_v0': v[1]}]

    if logging:
        import json
        for i in [0, 1]:
            log = res[i].log
            idx = myrank + i*(n//2)
            with open(str(idx) + '_dist_log.json', 'w') as f:
                json.dump(log, f)


    results = intracomm.gather(result)
    # Clean up requests
    for req in reqs.values():
        req.Wait()
    xbar = np.zeros(my_x.shape)
    z = my_x + my_y
    intracomm.Reduce([z, MPI.DOUBLE], [xbar, MPI.DOUBLE], op=MPI.SUM)
    
    if myrank == 0:
        xbar = xbar/n
        icomm.send(xbar, dest=0)
        icomm.send(results, dest=0)
        
    for comm, _, _ in my_left_deps:
        comm.Disconnect()
    for comm, _, _ in my_right_deps:
        comm.Disconnect()
    my_left_comm.Disconnect()
    my_right_comm.Disconnect()

if __name__ == '__main__':
    if 'child' in sys.argv:
        main_child()