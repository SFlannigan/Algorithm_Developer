
import numpy as np
from scipy.linalg import expm

def Rot_min(x,total_gate):
        
    # For problem 1
    #
    # Any number of rotations on a single spin can be done with only three
    # rotations: A rotation around each of the axes in turn.
    #
    # Similarly, I don't think the order of these rotations really matters. 
    
    # define pauli operators
    sz = 1/2 * np.array([[1.0,0],[0,-1.0]])
    sx = 1/2 * np.array([[0,1.0],[1.0,0]])
    sy = 1/2 * np.array([[0,-1j*1.0],[1j*1.0,0]])

    opt_gate = np.matmul(np.matmul(expm(-1j*x[0]*sx),expm(-1j*x[1]*sy)),expm(-1j*x[2]*sz))
    
    cost = np.linalg.norm(opt_gate - total_gate)

    return cost

