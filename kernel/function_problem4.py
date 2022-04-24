#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:32:25 2022

@author: stuart
"""

import numpy as np
from kernel.herm_conj import herm_conj
from kernel.Rot_min_4 import Rot_min_4
from kernel.Commute_Xgates import commute_Xgates
from scipy.linalg import expm
from scipy.optimize import minimize 

from random import seed
from random import random


def function_problem4(input_str,lenQ1,lenQ2):
        
    
    # define pauli operators
    sz = 1/2 * np.array([[1.0,0],[0,-1.0]])
    sx = 1/2 * np.array([[0,1.0],[1.0,0]])
    sy = 1/2 * np.array([[0,-1j*1.0],[1j*1.0,0]])
    iden = np.eye(2)
    
    # As two sequential CX(0,1) give the identity, we know that the rotation
    # must be theta = pi. So we can define the operators:
    theta_cx = np.pi
    CX01 =np.kron(expm(-1j*theta_cx*sz),expm(-1j*theta_cx*sx))
    CX10 =np.kron(expm(-1j*theta_cx*sx),expm(-1j*theta_cx*sz))
    
    gate_str = input_str.split(", ")
    num_gates = len(gate_str)
    
    total_gate = np.eye(4)
    
    gate_seq = ["" for x in range(num_gates)]
    gate_qu = ["" for x in range(num_gates)]
    gate_a = np.zeros(num_gates)
    for n in range(0, num_gates):
        gate = gate_str[n][gate_str[n].find("{")+1:gate_str[n].find("}")]
        
        if gate == "CX":
            gate_val = int(gate_str[n][gate_str[n].find("(")+1:gate_str[n].find(",")])
            if gate_val == 0:
                gate_OP = CX01
                gate_seq[n] = "CX01"
                gate_qu[n] = "0"
            else:
                gate_OP = CX10
                gate_seq[n] = "CX10"
                gate_qu[n] = "1"
        else:
            gate_seq[n] = gate
            gate_qu[n] = gate_str[n][gate_str[n].find("(")+1:gate_str[n].find(",")]
            gate_ang = float(gate_str[n][gate_str[n].find(",")+1:gate_str[n].find(")")])
            gate_a[n] = gate_ang
            gate_ang = gate_ang/180*np.pi  
            
                    
            if gate == "X":
                Rop = expm(-1j*gate_ang*sx)
            elif gate == "Z":
                Rop = expm(-1j*gate_ang*sz)
            else:
                Rop = expm(-1j*gate_ang*sy)     
                
            gate_OP = [iden,iden]
            gate_OP[int(gate_qu[n])] = Rop
            gate_OP = np.kron(gate_OP[0],gate_OP[1])
        
        total_gate = np.matmul(gate_OP,total_gate)
    
    # Check gate is unitary
    #  np.linalg.norm(np.matmul(herm_conj(total_gate),total_gate) - np.kron(np.array([[1.0,0],[0,1.0]]),np.array([[1.0,0],[0,1.0]])))
    
    
    ## Check to see if there are two X gates in the sequence, and attempts to 
    # bring them together by commuting them through the operators in between
    ind = [n for n, x in enumerate(gate_seq) if x == "X"]
    
    
    ind_0 = []
    ind_1 = []
    for n in range(0,len(ind)):
        if gate_qu[ind[n]] == "0":
            ind_0.append(ind[n])
        else:
            ind_1.append(ind[n])
      
    if len(ind_0)>1:
        gate_seq,gate_qu,gate_a = commute_Xgates(ind_0,gate_seq,gate_qu,gate_a)
    if len(ind_1)>1:
        gate_seq,gate_qu,gate_a = commute_Xgates(ind_1,gate_seq,gate_qu,gate_a)
            
           
    
    ## Next check to see if there are two Z gates in the sequence, and attempt to 
    # bring them together by commuting them through the operators in between
    ind = [n for n, x in enumerate(gate_seq) if x == "Z"]
    ind_0 = []
    ind_1 = []
    for n in range(len(ind)):
        if gate_qu[ind[n]] == "0":
            ind_0.append(ind[n])
        else:
            ind_1.append(ind[n])
    
    
    if len(ind_0)>1:
        gate_seq,gate_qu,gate_a = commute_Zgates(ind_0,gate_seq,gate_qu,gate_a)
    if len(ind_1)>1:
        gate_seq,gate_qu,gate_a = commute_Zgates(ind_1,gate_seq,gate_qu,gate_a)
        
        
    
    
    
    ## Format output string

    sig = 3 # output will be displayed with 3 sig. figs. for clarity
    out_str = ""
    time_val = 0.0
    for n in range(0, len(gate_seq)):
        
        if gate_seq[n] == "CX01":
            out_str = out_str+"{CX}(0,1,"+str(time_val)+")"
            time_val = time_val + lenQ2
        elif gate_seq[n] == "CX10":
            out_str = out_str+"{CX}(1,0,"+str(time_val)+")"
            time_val = time_val + lenQ2
        else:
            out_str = out_str+"{"+gate_seq[n]+"}("+gate_qu[n] + ", " + str(round(gate_a[n],sig))+", "+str(time_val)+")"
            if n != len(gate_seq)-1:
                if gate_seq[n+1] == "CX01" or gate_seq[n+1] == "CX10" or gate_qu[n+1] == gate_qu[n]:
                    time_val = time_val + lenQ1
       
        if n != (len(gate_seq)-1):
            out_str = out_str + ", "
        

    return [out_str]



