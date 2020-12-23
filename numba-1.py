#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.  Created on 2020-12-23

 Description (in paragraph form)

 Dependencies:
     numpy
     timeit
     donemusic
     matplotlib

 Changelog:
     Version 1.0 - First Version
================================================================================
"""

from numba import jit, vectorize, float64
import numpy as np
from timeit import default_timer

start = default_timer()

x = np.arange(10000000000,dtype=np.double)

@vectorize([float64(float64,float64)], target='cpu', nopython=True)
# @vectorize([float64(float64,float64)], target='parallel', nopython=True)
def multiply(a,b):
    return a*b

multiply(x, 2)


print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')