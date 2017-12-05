---
title: cfals.py
---

``` python
import concurrent.futures
from typing import Tuple, List, Optional, Any

import numpy as np
import scipy.sparse
from . import cfals_mp as alsmp
from functools import partial
import ctypes
import multiprocessing as mp


def _random(*size):
    # all numbers are initialized between 0 and 1, is it a problem?
    return np.random.randn(*size)



def _point_wise_norm_square(x: np.ndarray):
    return np.sum(np.abs(x) ** 2)

class CfALS:
    def __init__(self, x: scipy.sparse.spmatrix, h: int, alpha: float, num_workers = None):
        """
        Arguments
        ==========
        x: scipy.sparse.coo_matrix
            user*restaurant rating residual.
        h: int
            number of hidden variables
        alpha: float
            regularization constant, should be same as the coefficient as previous regressions
        """
        self.X_dok = x.todok()  # type: scipy.sparse.dok_matrix
        self.X_csr = x.tocsr()  # type: scipy.sparse.csr_matrix
        self.X_csc = x.tocsc()  # type: scipy.sparse.csc_matrix
        self.h = h
        self.alpha = alpha
        self.Nu, self.Nb = self.X_dok.shape  # type: Tuple[int,int]
        self.P = None  # type: np.ndarray
        self.QT = None  # type: np.ndarray
        self.reg = np.eye(h) * alpha  # type: np.ndarray
        self.rhs_zeros = np.zeros(h)

        self.Us = list(range(self.Nu))
        self.Bs = list(range(self.Nb))

        self.u_selectors = [None] * self.Nu  # type: List[np.ndarray]
        self.b_selectors = [None] * self.Nb  # type: List[np.ndarray]
        self.y_for_us = [None] * self.Nu  # type: List[np.ndarray]
        self.y_for_bs = [None] * self.Nb  # type: List[np.ndarray]

        ie = alsmp.CFALSInitializationExecutor(num_workers)
        try:
            ie.initialize(cfals=self)
            for u,r in enumerate(ie.execute(alsmp.get_u_selector, self.Us)):
                self.u_selectors[u] = r[0]
                self.y_for_us[u] = r[1]

            for b,r in enumerate(ie.execute(alsmp.get_b_selector, self.Bs)):
                self.b_selectors[b] = r[0]
                self.y_for_bs[b] = r[1]
        except:
            raise
        finally:
            ie.close()

        self.prev_loss = None  # type: Optional[float]
        self.cur_loss = None  # type: Optional[float]

        self.P_mem = mp.Array(ctypes.c_double, self.Nu*self.h)
        self.QT_mem = mp.Array(ctypes.c_double, self.Nb*self.h)
        self.P = np.ctypeslib.as_array(self.P_mem.get_obj()).reshape(self.Nu, self.h)
        self.QT = np.ctypeslib.as_array(self.QT_mem.get_obj()).reshape(self.Nb, self.h)


    def initialize(self):
        self.P[:] = _random(self.Nu, self.h)
        self.QT[:] = _random(self.Nb, self.h)

        self.prev_loss = self.cur_loss
        self.cur_loss = None

    def steps(self, n, executor: alsmp.CFALSExecutor):
        for i in range(n):
            self.step(executor=executor)

    def step(self, executor: alsmp.CFALSExecutor):
        self.prev_loss = self.cur_loss

        #for u,pu in enumerate(executor.execute(partial(alsmp.solve_pu, QT=self.QT), self.Us)):
        for u,pu in enumerate(executor.execute(alsmp.solve_pu2, self.Us)):
            self.P[u,:] = pu

        #for b,qb in enumerate(executor.execute(partial(alsmp.solve_qb, P=self.P), self.Bs)):
        for b,qb in enumerate(executor.execute(alsmp.solve_qb2, self.Bs)):
            self.QT[b,:] = qb

        self.cur_loss = self.loss()

        if self.prev_loss is not None and self.cur_loss > self.prev_loss:
            print("Loss increased")

    def factorize(self,
                  max_iterations: int = 10000,
                  eps: float = 0.001,
                  skip_initialize: bool = False,
                  executor: alsmp.CFALSExecutor = None):
        if executor is None:
            raise ValueError("An instance of Executor must be passed in")
        if not skip_initialize:
            self.initialize()
        i = 0
        while i < max_iterations and (
                        (self.cur_loss is None) or (self.prev_loss is None) or (
                            abs(self.cur_loss - self.prev_loss) > eps)):
            self.step(executor=executor)
            i += 1
        if abs(self.cur_loss - self.prev_loss) >= eps:
            print("Did not converge!")
        return self.P, self.QT.T

    def loss(self):
        reg = self.alpha * (_point_wise_norm_square(self.P) + _point_wise_norm_square(self.QT))
        return self.loss_no_reg() + reg

    def loss_no_reg(self):
        result = 0.0
        for (u, b) in self.X_dok.keys():
            result += (self.get_r(u, b) - self.X_dok[u, b]) ** 2
        return result

    def get_r(self, u: int, b: int) -> float:
        return self.P[u, :].dot(self.QT[b, :])
```
