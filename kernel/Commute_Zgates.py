#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:35:55 2022

@author: stuart
"""

import numpy as np
from kernel.Rot_min_4 import Rot_min_4
from scipy.linalg import expm
from scipy.optimize import minimize 
from random import seed
from random import random

def commute_Zgates(ind,gate_seq_new,gate_qu_new,gate_a_new):
    
    # define pauli operators
    sz = 1/2 * np.array([[1.0,0],[0,-1.0]])
    sx = 1/2 * np.array([[0,1.0],[1.0,0]])
    sy = 1/2 * np.array([[0,-1j*1.0],[1j*1.0,0]])
    iden = np.eye(2)
    
    
    if len(ind)>1:
        
        for m in range(0,len(ind)-1):
            n = len(ind) - m - 1
            if ind[n] != 0 and n != 0:
                dd = ind[n] - ind[n-1]
                success = 1
                for diff in range(dd):
                    if gate_seq_new[ind[n]-diff-1] == "CX01" and gate_qu_new[ind[n]-diff-1] == gate_qu_new[ind[n]-diff]:
                        # Rotations act on different qubits so they can commute
                        gate_seq_new[ind[n]-diff-1] = gate_seq_new[ind[n]-diff]
                        gate_seq_new[ind[n]-diff] = "CX01"
                        gate_qu_new[ind[n]-diff-1] = gate_qu_new[ind[n]-diff]
                        gate_qu_new[ind[n]-diff] = ""
                        gate_a_new[ind[n]-diff-1] = gate_a_new[ind[n]-diff]
                        gate_a_new[ind[n]-diff] = 0.0
                    elif gate_seq_new[ind[n]-diff-1] == "CX10" and gate_qu_new[ind[n]-diff-1] == gate_qu_new[ind[n]-diff]:
                        # Rotations act on different qubits so they can commute
                        gate_seq_new[ind[n]-diff-1] = gate_seq_new[ind[n]-diff]
                        gate_seq_new[ind[n]-diff] = "CX10"
                        gate_qu_new[ind[n]-diff-1] = gate_qu_new[ind[n]-diff]
                        gate_qu_new[ind[n]-diff] = ""
                        gate_a_new[ind[n]-diff-1] = gate_a_new[ind[n]-diff]
                        gate_a_new[ind[n]-diff] = 0.0
                    elif gate_seq_new[ind[n]-diff-1] == "X" and gate_qu_new[ind[n]-diff-1] != gate_qu_new[ind[n]-diff]:
                        # Rotations act on different qubits so they can commute
                        gate_seq_new[ind[n]-diff-1] = gate_seq_new[ind[n]-diff]
                        gate_seq_new[ind[n]-diff] = "X"
                        gate_temp = gate_qu_new[ind[n]-diff-1]
                        gate_qu_new[ind[n]-diff-1] = gate_qu_new[ind[n]-diff]
                        gate_qu_new[ind[n]-diff] = gate_temp
                        ang_temp = gate_a_new[ind[n]-diff-1]
                        gate_a_new[ind[n]-diff-1] = gate_a_new[ind[n]-diff]
                        gate_a_new[ind[n]-diff] = ang_temp
                    elif gate_seq_new[ind[n]-diff-1] == "X" and gate_qu_new[ind[n]-diff-1] == gate_qu_new[ind[n]-diff]:
                        # Rotations act on the same qubit so they DO NOT commute
                        # We then variationally optimise and see if we can obtain a
                        # suitable commutation relation
                        
                        gate_ang = gate_a_new[ind[n]-diff-1]/180*np.pi 
                        R1 = expm(-1j*gate_ang*sx)
                        
                        gate_ang = gate_a_new[ind[n]-diff]/180*np.pi 
                        R2 = expm(-1j*gate_ang*sz)
                        
                        total_gate = np.matmul(R2,R1)
                        
                        x0 = [np.pi*random() - np.pi*random(),np.pi*random() - np.pi*random()]
                        fout = minimize(Rot_min_4, x0, args=(total_gate,sx,sz))
                        
                        if fout.fun > 1e-3: # cannot find a suitable relation
                            success = 0
                            break
                        else:
                            gate_seq_new[ind[n]-diff-1] = "Z"
                            gate_seq_new[ind[n]-diff] = "X"
                            gate_a_new[ind[n]-diff-1] = fout.x[1] * 180/np.pi
                            gate_a_new[ind[n]-diff] = fout.x[0] * 180/np.pi
                            
                    else:
                        success = 0
                        break
                
                if success == 1:
                    gate_seq_new.pop(ind[n] - dd+1)
                    gate_qu_new.pop(ind[n] - dd+1)
                    gate_a_new[ind[n]-dd] = gate_a_new[ind[n]-dd] + gate_a_new[ind[n]-dd+1]
                    gate_a_new = np.delete(gate_a_new, ind[n]-dd+1)
                
        gate_seq = gate_seq_new
        gate_qu = gate_qu_new
        gate_a = gate_a_new
        
        # Remove rotations with angle zero
        for m in range(len(gate_seq)):
            n = len(gate_seq) - m - 1
            if (np.abs(gate_a[n]) < 1e-3 or np.abs(np.abs(gate_a[n])-360) < 1e-3) and gate_seq[n] == "X":
                gate_seq.pop(n)
                gate_qu.pop(n)
                gate_a = np.delete(gate_a,n)
            elif (np.abs(gate_a[n]) < 1e-3 or np.abs(np.abs(gate_a[n])-360) < 1e-3) and gate_seq[n] == "Z":
                gate_seq.pop(n)
                gate_qu.pop(n)
                gate_a = np.delete(gate_a,n)
    
    
    
    return [gate_seq,gate_qu,gate_a]