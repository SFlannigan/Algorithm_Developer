#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:32:25 2022

@author: stuart
"""

# The approach for problem 3 is to initialise the minimisation (as in problems 1 and 2)
# at different points in parameter space. The minimsation will find a different
# combination of rotataions (as these are not unique). We then check which rotation
# sequence has the smallest time cost and then we use that one.

import numpy as np
from kernel.function_problem3 import function_problem3

input_str = "{X}(90), {X}(90)"

#input_str = "{X}(90), {Y}(180), {X}(90)"

#input_str = "{X}(75), {Y}(180), {X}(90), {Z}(35), {Y}(40)"

lenZ = 100
lenX = 10

out_str,err,time_cost = function_problem3(input_str,lenZ,lenX)

print(out_str)
print(time_cost)


