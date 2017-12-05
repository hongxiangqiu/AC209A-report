---
title: cfals_mp.py
---

``` python
import multiprocessing as mp
import numpy as np
from scipy.optimize import nnls
import numpy.linalg as la

def get_u_selector(u: int):
    u_selector = np.sort(data_bag.X_csr[u, :].indices)
    yu = np.asarray(data_bag.X_csr[u, u_selector].todense()).reshape(-1)
    #yu = np.concatenate([yu, data_bag.rhs_zeros])
    return u_selector, yu

def get_b_selector(b: int):
    b_selector = np.sort(data_bag.X_csc[:, b].indices)
    yb = np.asarray(data_bag.X_csc[b_selector, b].todense()).reshape(-1)
    #yb = np.concatenate([yb, data_bag.rhs_zeros])
    return b_selector, yb

def _nnls(a, b):
    return nnls(a, b)[0]

def _lasolve(a,b):
    return la.solve(a, b)

def _lalstsq(a, b):
    return la.lstsq(a, b)[0]

def __solve_with_reg2(a, y):
    #return _lalstsq(np.vstack([a, data_bag.reg]), y)
    #return _lalstsq(np.vstack([a, data_bag.reg]), y)
    ata = a.T.dot(a)
    lhs = ata + data_bag.reg
    rhs = y.dot(a)
    return _lasolve(lhs, rhs)

def solve_pu(u: int, QT):
    select = data_bag.u_selectors[u]
    pu = __solve_with_reg2(QT[select, :], data_bag.y_for_us[u])
    return pu

def solve_qb(b: int, P):
    select = data_bag.b_selectors[b]
    qb = __solve_with_reg2(P[select, :], data_bag.y_for_bs[b])
    return qb

def solve_pu2(u):
    QT = np.ctypeslib.as_array(data_bag.QT_mem.get_obj()).reshape(data_bag.Nb, data_bag.h)
    return solve_pu(u, QT=QT)

def solve_qb2(b):
    P = np.ctypeslib.as_array(data_bag.P_mem.get_obj()).reshape(data_bag.Nu, data_bag.h)
    return solve_qb(b, P=P)

class CfALSInitializationDataBag:
    def __init__(self, cfals=None, pickled=None):
        if cfals is None and pickled is None:
            raise ValueError
        elif cfals is not None and pickled is not None:
            raise ValueError
        elif cfals is not None:
            self.X_csc = cfals.X_csc
            self.X_csr = cfals.X_csr
            self.rhs_zeros = cfals.rhs_zeros
        else:
            self.X_csc = pickled['X_csc']
            self.X_csr = pickled['X_csr']
            self.rhs_zeros = pickled['rhs_zeros']

    def get_pickled(self):
        return {    'X_csc': self.X_csc,
                    'X_csr': self.X_csr,
                    'rhs_zeros': self.rhs_zeros
               }

    def get_initializer(self):
        def init():
            global data_pickled
            global data_bag
            data_bag = CfALSInitializationDataBag(pickled = data_pickled)
        return init

class CfALSDataBag:
    def __init__(self, cfals=None, pickled=None):
        self.reg = self.__get_data('reg', cfals=cfals, pickled=pickled)
        self.u_selectors = self.__get_data('u_selectors', cfals=cfals, pickled=pickled)
        self.b_selectors = self.__get_data('b_selectors', cfals=cfals, pickled=pickled)
        self.y_for_us = self.__get_data('y_for_us', cfals=cfals, pickled=pickled)
        self.y_for_bs = self.__get_data('y_for_bs', cfals=cfals, pickled=pickled)
        self.P_mem = self.__get_data('P_mem', cfals=cfals, pickled=pickled)
        self.QT_mem = self.__get_data('QT_mem', cfals=cfals, pickled=pickled)
        self.h = self.__get_data('h', cfals=cfals, pickled=pickled)
        self.Nu = self.__get_data('Nu', cfals=cfals, pickled=pickled)
        self.Nb = self.__get_data('Nb', cfals=cfals, pickled=pickled)

    def __get_data(self, key, cfals=None, pickled=None, pickled_processor=None):
        if cfals is None and pickled is None:
            raise ValueError
        if cfals is not None and pickled is not None:
            raise ValueError
        if cfals is not None:
            return cfals.__dict__[key]
        if pickled_processor is None:
            return pickled[key]
        return pickled_processor(pickled[key])

    def get_pickled(self):
        return {    'reg': self.reg,
                    'u_selectors': self.u_selectors,
                    'b_selectors': self.b_selectors,
                    'y_for_us': self.y_for_us,
                    'y_for_bs': self.y_for_bs,
                    'P_mem': self.P_mem,
                    'QT_mem': self.QT_mem,
                    'h': self.h,
                    'Nu': self.Nu,
                    'Nb': self.Nb,
               }

    def get_initializer(self):
        def init():
            global data_pickled
            global data_bag
            data_bag = CfALSDataBag(pickled = data_pickled)
        return init

class CFALSInitializationExecutor:
    def __init__(self, n_workers:int):
        self.data_bag = None
        self.n_workers = n_workers

    def initialize(self, cfals):
        self.data_bag = CfALSInitializationDataBag(cfals)

        global data_pickled
        data_pickled = self.data_bag.get_pickled()

        init = self.data_bag.get_initializer()
        self.pool = mp.Pool(processes=self.n_workers,initializer=init)

    def execute(self, func, parameters):
        return self.pool.map(func, parameters)

    def close(self):
        self.pool.close()


class CFALSExecutor:
    def __init__(self, n_workers:int):
        self.data_bag = None
        self.n_workers = n_workers

    def initialize(self, cfals):
        self.data_bag = CfALSDataBag(cfals)

        global data_pickled
        data_pickled = self.data_bag.get_pickled()

        init = self.data_bag.get_initializer()
        self.pool = mp.Pool(processes=self.n_workers,initializer=init)

    def execute(self, func, parameters):
        return self.pool.map(func, parameters)


    def close(self):
        self.pool.close()
```
