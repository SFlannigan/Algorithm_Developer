#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:32:25 2022

@author: stuart
""" 

# The approach for problem 1 is to variationally minimise the rotation angles 
# needed to reproduce the full sequence of gates, but with a smaller set of 
# operations. 


import numpy as np
from kernel.function_problem1 import function_problem1

input_str = "{X}(90), {X}(90)"

#input_str = "{X}(90), {Y}(180), {X}(90)"

#input_str = "{X}(75), {Y}(180), {X}(90), {Z}(35), {Y}(40)"

out_str,err = function_problem1(input_str)

print(out_str)