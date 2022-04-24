#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:32:25 2022

@author: stuart
"""

# The approach for problem 4 is to read the sequence of gates, find any rotations
# around the same axis on the same qubit and attempt to bring these together
# in the sequence (so they can be combined and applied as a single rotation).
# This is done by attempting to commute one rotation through all the rest in the sequence.
# 
# We know that applications on different qubits can be interchanged (commuted)
# and CX(0,1) commutes with R_X (on qubit 1) etc. 
# 
# For an X and Z gate, we attempt to variationally minimise the error for 
# applying rotations around these axes in reverse order but with different angles.

import numpy as np
from kernel.function_problem4 import function_problem4

input_str = "{X}(1,90), {Z}(1,180), {CX}(0,1), {X}(1,90)"

#input_str = "{X}(1,90), {Z}(1,180), {CX}(1,0), {X}(1,90)"

#input_str = "{X}(1,90), {Z}(0,180), {CX}(0,1), {X}(1,90)"

lenQ2 = 100
lenQ1 = 10

out_str = function_problem4(input_str,lenQ1,lenQ2)

print(out_str)