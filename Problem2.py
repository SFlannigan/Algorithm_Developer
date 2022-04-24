#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:32:25 2022

@author: stuart
"""

# The approach for problem 2 is the same as for problem 1, but now without Y
# rotations

import numpy as np
from kernel.function_problem2 import function_problem2

input_str = "{X}(90), {X}(90)"

#input_str = "{X}(90), {Y}(180), {X}(90)"

#input_str = "{X}(75), {Y}(180), {X}(90), {Z}(35), {Y}(40)"

out_str,err = function_problem2(input_str)

print(out_str)