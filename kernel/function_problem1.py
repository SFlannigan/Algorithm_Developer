#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:32:25 2022

@author: stuart
"""

import numpy as np
from kernel.herm_conj import herm_conj
from kernel.Rot_min import Rot_min
from scipy.linalg import expm
from scipy.optimize import minimize 
from numpy.linalg import eig


def function_problem1(input_str):

    gate_str = input_str.split(", ")
    num_gates = len(gate_str)
    
    # define pauli operators
    sz = 1/2 * np.array([[1.0,0],[0,-1.0]])
    sx = 1/2 * np.array([[0,1.0],[1.0,0]])
    sy = 1/2 * np.array([[0,-1j*1.0],[1j*1.0,0]])
    
    total_gate = np.array([[1.0,0],[0,1.0]])
    for n in range(0, num_gates):
        gate = gate_str[n][gate_str[n].find("{")+1:gate_str[n].find("}")]
        gate_ang = float(gate_str[n][gate_str[n].find("(")+1:gate_str[n].find(")")])
        
        gate_ang = gate_ang/180*np.pi   
                
        if gate == "X":
            Rop = expm(-1j*gate_ang*sx)
        elif gate == "Z":
            Rop = expm(-1j*gate_ang*sz)
        else:
            Rop = expm(-1j*gate_ang*sy)        
        
        total_gate = np.matmul(Rop,total_gate)
        
    # For debugging: test if operation is unitary
    # np.matmul(np.conjugate(np.transpose(total_gate)),total_gate)
    # np.matmul(herm_conj(total_gate),total_gate)
    
    x0 = [1,1,1]
    fout = minimize(Rot_min, x0, args=(total_gate))
    xout = fout.x
    err = fout.fun
    
    gate_val = np.array([0,1,2])
    gate_str = ["X","Y","Z"]
    
    # check if any operations are zero rotations and remove these from the list.
    gate_val = gate_val[np.abs(xout)>1e-3]
    xout = xout[np.abs(xout)>1e-3]
    
    gate_str[gate_val[0]]
    
    sig = 3 # output will be displayed with 3 sig. figs. for clarity
    out_str = ""
    for n in range(0, len(xout)):
        out_str = out_str+"{"+gate_str[gate_val[n]]+"}("+str(round(xout[n]/np.pi*180,sig))+")"
        if n != (len(xout)-1):
            out_str = out_str + ", "
        

    return [out_str,err]



