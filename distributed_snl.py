
import sys
from sys import path
from pathlib import Path
# Sets parent directory as a string path
parent_dir = str(Path(__file__).resolve().parent.parent)
path.append(parent_dir)
import json
from mpi4py import MPI
import numpy as np
from time import time
from datetime import datetime

def distributed_block_sparse_solve(n, data, resolvents, Z, warmstartprimal=None, itrs=1001, gamma=0.9, alpha=1.0, logging=False, verbose=False, check_termination=False):
    '''
    Solve the 2-Block resolvent splitting algorithm in parallel using MPI

    Args:
        n (int): Number of resolvents
        data (list): List of data for each resolvent
        resolvents (list): List of resolvents
        Z (ndarray): Weight matrix
        warmstartprimal (ndarray): Warm start for the primal variables
        itrs (int): Number of iterations
        gamma (float): Step size
        alpha (float): Proximal parameter
        logging (bool): Whether to log the results
        verbose (bool): Print verbose output
        check_termination (bool): Whether to check for termination

    Returns:
        x_bar (ndarray): The average of the primal variables.
        variance (float): The variance of the primal variables.
        sum_gradient (ndarray): The sum of the gradients.
        results (list): The final x and v results for each resolvent.
        runningtime (float): The time it took to run the algorithm.
        itr (int): The number of iterations.
        update_norm (float): The norm of the update.
    '''
    assert(MPI.COMM_WORLD.Get_size() == 1)
    assert(n % 2 == 0)
    icomm = MPI.COMM_WORLD.Spawn(command=sys.executable,
                                 args=[__file__, 'child'],
                                 maxprocs=n//2)

    # Send data to workers    
    ws = warmstartprimal is not None
    
    icomm.bcast((n, Z, gamma, alpha, itrs, logging, verbose, ws, check_termination), root=MPI.ROOT)
    if ws:
        icomm.bcast(warmstartprimal, root=MPI.ROOT)

    # Each node gets two resolvents and their data
    for i in range(n//2):
        j = i+n//2
        icomm.send((data[i], data[j], resolvents[i], resolvents[j]), dest=i)
    runningtime = time()
    if verbose:print(datetime.now(), 'Data sent to workers', flush=True)
        
    # Receive results from workers after they finish
    x_bar = icomm.recv(source=0)
    runningtime = time() - runningtime
    variance = icomm.recv(source=0)
    sum_gradient = icomm.recv(source=0)
    results = icomm.recv(source=0)
    itr = icomm.recv(source=0)
    if check_termination:
        update_norm = icomm.recv(source=0)
    else:
        update_norm = None
    icomm.Disconnect()
    return x_bar, variance, sum_gradient, results, runningtime, itr, update_norm

# Child process
def main_child():
    icomm = MPI.Comm.Get_parent()
    assert icomm != MPI.COMM_NULL
    worker(icomm)
    icomm.Disconnect()


# Helper function to get neighbors
def getNeighbors(Z, n, myrank, zerotol=1e-7):
    firstneighbors = set()
    secondneighbors = set()
    for i in range(n):
        # First prox sends to:
        if i != myrank and not np.isclose(Z[n+i, myrank], 0.0, atol=zerotol):
            firstneighbors.add(i)
        # Second prox receives from:
        if i != myrank and not np.isclose(Z[myrank+n, i], 0.0, atol=zerotol):
            secondneighbors.add(i)
    assert(firstneighbors == secondneighbors)
    return list(firstneighbors)

# Worker process - core parallel algorithm execution
def worker(icomm):
    myrank = icomm.Get_rank()

    # Build intracommunicator
    intracomm = MPI.COMM_WORLD

    # Receive data from parent
    n, Z, gamma, alpha, itrs, logging, verbose, ws, check_termination = icomm.bcast((), root=0)
    if ws:
        warmstartprimal = icomm.bcast((), root=0)
        v = [warmstartprimal, -warmstartprimal.copy()]
        shape = warmstartprimal.shape
    first_data, second_data, first_resolvent, second_resolvent = icomm.recv(source=0)
    
    if not ws:
        shape = first_data['shape']
        v = [np.zeros(shape), np.zeros(shape)]

    res = [first_resolvent(**first_data), second_resolvent(**second_data)]
    if logging:
        res[0].logging = True
        res[1].logging = True
        
    sum_x = np.zeros(shape)
    sum_y = np.zeros(shape)
    update_0 = np.zeros(shape)
    update_1 = np.zeros(shape)
    itr_period = max(1, itrs // 10)
    t = time()
    myrankshift = n//2 + myrank
    neighbor_list = getNeighbors(Z, n//2, myrank)

    # Create distributed graph communicator
    dist_graph_comm = intracomm.Create_dist_graph_adjacent(neighbor_list, neighbor_list)
    

    recv_buff = np.empty((len(neighbor_list), *shape), dtype=np.float64)
    reqs = {}
    
    if verbose and myrank == 0:
        print('date\t time\t\trank iter ||delta v||', flush=True)
    for itr in range(itrs):

        # First block
        my_x = res[0].prox(v[0], alpha)
        dist_graph_comm.Neighbor_allgather(sendbuf=my_x, recvbuf=recv_buff)
        
        np.copyto(sum_x, -my_x*Z[myrankshift, myrank])
        for i, neighbor in enumerate(neighbor_list):
            sum_x -= recv_buff[i]*Z[myrankshift, neighbor]

        # Second block
        my_y = res[1].prox(v[1]+sum_x, alpha)
        dist_graph_comm.Neighbor_allgather(sendbuf=my_y, recvbuf=recv_buff)
        # 
        np.copyto(sum_y, -my_y*Z[myrank, myrankshift])
        for i, neighbor in enumerate(neighbor_list):
            sum_y -= recv_buff[i]*Z[myrank, neighbor+n//2]
            
        np.multiply(my_y, 2, out=update_1)   # update_1 = 2 * my_y
        update_1 -= sum_x                    # update_1 = (2 * my_y) - sum_x
        update_1 *= gamma                    # apply gamma
        v[1] -= update_1                     # update v[1] in place

        np.multiply(my_x, 2, out=update_0)   # update_1 = 2 * my_y
        update_0 -= sum_y                    # update_1 = (2 * my_y) - sum_x
        update_0 *= gamma                    # apply gamma
        v[0] -= update_0                     # update v[1] in place

        if check_termination and itr > 0 and itr % 20 == 0:
            update_norm = (np.linalg.norm(update_0) + np.linalg.norm(update_1))/(4*(3+len(neighbor_list))**2)
            local_stop = update_norm < 1e-3
            global_stop = intracomm.allreduce(local_stop, op=MPI.LAND)
            if global_stop:
                break

        if verbose and myrank == 0 and itr % itr_period == 0:
            v_sq = np.linalg.norm(update_0)**2 + np.linalg.norm(update_1)**2
            print(datetime.now(), myrank, itr, np.round(v_sq**0.5,3), flush=True)

    if verbose and myrank == 0:
        print(datetime.now(), 'Worker', myrank, 'finished, time', time()-t, flush=True)

    result = [{'first_x': my_x, 'second_x': my_y, 'first_v0': v[0], 'second_v0': v[1]}]

    if logging:
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
    intracomm.Allreduce([z, MPI.DOUBLE], [xbar, MPI.DOUBLE], op=MPI.SUM)
    my_variance = np.sum((my_x - xbar)**2 + (my_y - xbar)**2)
    variance = np.zeros(1)
    intracomm.Reduce([my_variance, MPI.DOUBLE], [variance, MPI.DOUBLE], op=MPI.SUM)
    my_sumgradient = (v[0] - res[0].prox(v[0], alpha)) + (v[1]+sum_x - res[1].prox(v[1]+sum_x, alpha))
    sumgradient = np.zeros(my_x.shape)
    intracomm.Reduce([my_sumgradient, MPI.DOUBLE], [sumgradient, MPI.DOUBLE], op=MPI.SUM)

    if myrank == 0:
        xbar = xbar/n
        icomm.send(xbar, dest=0)
        variance /= n
        icomm.send(variance[0], dest=0)
        icomm.send(np.linalg.norm(sumgradient), dest=0)
        icomm.send(results, dest=0)
        icomm.send(itr, dest=0)
        if check_termination:
            icomm.send(update_norm, dest=0)
    
    dist_graph_comm.Free()

if __name__ == '__main__':
    if 'child' in sys.argv:
        main_child()