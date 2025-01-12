#!/home/anuj.mishra/anaconda3/envs/gwpy/bin/python

import numpy as np

import mpmath
from mpmath import hyp1f1
import math
import scipy
import pandas as pd

from copy import deepcopy

from itertools import product
from functools import partial
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# loading Cython version of the point lens class
import sys
sys.path.append('src/')
import cythonized_pnt_lens_class as pnt_lens_cy

import multiprocessing as mp

import pickle

outdir='./'#./data/'
label = 'point_lens_Ff_lookup_table_Geo_relErr_1p0'

if __name__=='__main__': 
    print('\n### Generating the grid where F(f) values will be computed ###')
    ys = np.logspace(-2, np.log10(5), int(4e4))
    ws = np.logspace(-4, np.log10(1.3e4), int(6e3))

    y_w_grid = []   #np.array([None]*len(input_prms))
    for y in ys[:]:
        wc = pnt_lens_cy.wc_geo_re1p0(y)
        tmp_ws = ws[ws<=1.1*wc]
        y_w_grid.append([y, tmp_ws])

    y_w_grid = np.array(y_w_grid, dtype=object)     

    print('ys: ({:.3f}, {:.3f}), len(ys)={}'.format(ys[0], ys[-1], len(ys)))
    print('ws: ({:.4f}, {:2f}), max_len(ws)={}'.format(ws[0], ws[-1], len(ws)))
    print('Total points in grid: ', np.sum([len(y_w_grid[i][1]) for i in range(len(y_w_grid))]) )
    
    print('\n### Generating the lookup table ###')
    npool = int(sys.argv[1])
    pool = mp.Pool(processes=npool)
    t1=time.time()
    Ff_grid = dict()
    k=0
    for y_ws in tqdm(y_w_grid):
        g_y = y_ws[0]
        g_ws = y_ws[1]
        Ff_eff_partial = partial(pnt_lens_cy.point_Fw_eff, y = g_y)
        tmp_res = np.array(list(pool.map(Ff_eff_partial, g_ws)))
        Ff_grid[str(k)] = dict(y=g_y, ws=g_ws, Ffs_real=np.real(tmp_res), Ffs_imag=np.imag(tmp_res))    
        k+=1
    t2 = time.time() 
    print('\nComputation Time (generation) = {:.2f} s'.format(t2-t1))
    print('\n### Done Generation ###')

    print('\n### Exporting to pickle file ###')
    t1 = time.time()
    with open(outdir + label + '.pkl', 'wb') as f:
        pickle.dump(Ff_grid, f)
    t2 = time.time() 
    print('Computation Time (exporting) = {:.2f} s'.format(t2-t1))  
    print('### Done Exporting ###')

    print('\n### Importing the pickle file and checking ###')
    with open(outdir + label + '.pkl', 'rb') as f:
        Ff_grid = pickle.load(f)

    print('Number of entries in lookup_table: ', len(Ff_grid))  
    print('\nKeys in each entry: ', Ff_grid['0'].keys())
    print('\nFirst entry with values:', Ff_grid['0'])
