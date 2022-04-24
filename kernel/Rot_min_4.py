
import numpy as np
from scipy.linalg import expm

def Rot_min_4(x,total_gate,S1,S2):


    opt_gate = np.matmul(expm(-1j*x[0]*S1),expm(-1j*(x[1])*S2))
    
    cost = np.linalg.norm(opt_gate - total_gate)
    
    
    
    

    return cost

