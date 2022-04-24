#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:32:25 2022

@author: stuart
"""

import numpy as np
from kernel.herm_conj import herm_conj
from kernel.Rot_min_3 import Rot_min_3
from scipy.linalg import expm
from scipy.optimize import minimize 
from numpy.linalg import eig

from random import seed
from random import random


def function_problem3(input_str,lenZ,lenX):

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


    np.matmul(expm(-1j*np.pi*sx),expm(1j*np.pi*sz))
    
    np.matmul(np.matmul(expm(-1j*np.pi/2*sx),expm(-1j*np.pi*sy)),expm(-1j*np.pi/2*sx))
    
    # The approach here is to initialise the minimisation at different points
    # in parameter space. The minimsation will find a different combination of 
    # rotataions (as these are not unique). We then check which rotation
    # sequence has the smallest time cost and then we use that one.
    min_cost = 1000
    for n in range(0,10):
        # Inputs are randomised. So we don't get stuck in local minima
        x0 = [np.pi*random() - np.pi*random(),np.pi*random() - np.pi*random(),np.pi*random() - np.pi*random()]
        
        eta = 0
        fout = minimize(Rot_min_3, x0, args=(total_gate))
        xout = fout.x
        err = fout.fun
        
        time_cost = 0
        if np.abs(xout[0]) > 1e-3:
            time_cost = time_cost + lenX
        
        if np.abs(xout[1]) > 1e-3 and np.abs(xout[2]) > 1e-3:
            time_cost = time_cost + 2*lenZ + lenX
        elif np.abs(xout[1]) > 1e-3:
            time_cost = time_cost + 2*lenZ + lenX
        elif np.abs(xout[2]) > 1e-3:
            time_cost = time_cost + lenZ
            
            
        if time_cost < min_cost:
            min_cost = time_cost
            xsv = xout
            errsv = err
            
    xout = xsv
    err = errsv
    time_cost = min_cost
    
    
    gate_val = np.array([0,1,2])
    gate_str = ["X","Y","Z"]
    

    sig = 3 # output will be displayed with 3 sig. figs. for clarity
    out_str = ""
    yval=0
    for n in range(0, len(xout)):
        
        if gate_val[n] == 0:
            if np.abs(xout[n])>1e-3: # check if the rotation is non-zero
                out_str = out_str+"{"+gate_str[gate_val[n]]+"}("+str(round(xout[n]/np.pi*180,sig))+")"
        elif gate_val[n]==1:
            if np.abs(xout[n])>1e-3: # check if the rotation is non-zero
                out_str = out_str+"{Z}(90), {X}("+str(round(xout[n]/np.pi*180,sig))+")"
                yval = 1
        else:
            if yval == 1: # must apply extra 90 degree z rotation
                xout[n] = xout[n] - np.pi/2
            
            if np.abs(xout[n])>1e-3: # check if the rotation is non-zero
                out_str = out_str+"{"+gate_str[gate_val[n]]+"}("+str(round(xout[n]/np.pi*180,sig))+")"

        if n != (len(xout)-1):
            if np.abs(xout[n])>1e-3:
                out_str = out_str + ", "
        

    return [out_str,err,time_cost]



