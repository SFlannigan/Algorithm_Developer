
import numpy as np
from scipy.linalg import expm

def Rot_min_2(x,total_gate):
    
    # For problem 2
    #
    # Any number of rotations on a single spin can be done with only three
    # rotations: A rotation around each of the axes in turn.
    #
    # But this time, we cannot apply a roataion around Y. 
    
    # define pauli operators
    sz = 1/2 * np.array([[1.0,0],[0,-1.0]])
    sx = 1/2 * np.array([[0,1.0],[1.0,0]])
    sy = 1/2 * np.array([[0,-1j*1.0],[1j*1.0,0]])

    opt_gate = np.matmul(np.matmul(expm(-1j*x[0]*sx),np.matmul(expm(-1j*np.pi/2*sz),expm(-1j*x[1]*sx))),expm(-1j*(x[2]-np.pi/2)*sz))
    
    cost = np.linalg.norm(opt_gate - total_gate)

    return cost

