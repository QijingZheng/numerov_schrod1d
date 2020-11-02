#!/usr/bin/env python

import time
import ase
import numpy as np


class schro1d(object):
    """
    Matrix numerov method for solving 1D Schrodinger equation.
    """

    def __init__(self, x, v, m=1.0, pbc=False):
        """
        """
        self.x = np.asarray(x)    # unit Angstrom
        self.v = np.asarray(v)    # unit eV

        assert self.x.ndim == 1
        assert self.x.shape == self.v.shape
        self.h = self.x[1] - self.x[0]
        self.n = self.x.size
        # make sure that the grid is equal-spaced
        assert np.allclose(self.h, np.diff(self.x))

        self.m = m                # unit electron mass
        self.pbc = pbc            # periodic boundary condition?

        self.gen_matrix()

    def gen_matrix(self):
        # hbar**2 / (2 * m)
        K = ase.units._hbar**2 / \
            (2 * self.m * ase.units._me) / 1E-20 / ase.units._e
        A = np.diag(-2*np.ones(self.n), 0) / self.h**2 +\
            np.diag(np.ones(self.n-1), -1) / self.h**2 +\
            np.diag(np.ones(self.n-1), 1) / self.h**2

        B = np.diag(10*np.ones(self.n), 0) / 12 +\
            np.diag(np.ones(self.n-1), -1) / 12 +\
            np.diag(np.ones(self.n-1), 1) / 12

        V = np.diag(self.v)

        if self.pbc:
            # periodic boundary condition
            A[0, -1] = A[-1, 0] = K / self.h**2
            B[0, -1] = B[-1, 0] = 1. / 12

        self.hamil = -K * np.dot(np.linalg.inv(B), A) + V

        # somehow the scipy sparse matrix diagonalization is too slow...

        # else:
        #     from scipy.sparse import diags
        #     from scipy.sparse.linalg import inv
        #     A = diags(
        #         [-2*np.ones(self.n), np.ones(self.n-1), np.ones(self.n-1)],
        #         [0, -1, 1], format='csc'
        #     ) / self.h**2 * K
        #
        #     B = diags(
        #         [10*np.ones(self.n), np.ones(self.n-1), np.ones(self.n-1)],
        #         [0, -1, 1], format='csc'
        #     ) / 12.
        #
        #     V = diags([v], [0], format='csc')
        #
        #     self.hamil = -inv(B).dot(A) + V

    def knots(self):
        '''
        '''
        return np.sum(
            np.abs(
                np.diff(self.w >= 0, axis=0)
            ),
            axis=0
        )

    def solve(self, n=1):
        e, w = np.linalg.eigh(self.hamil)

        I = np.argsort(e)
        self.e = e[I][:n]
        self.w = w[:, I][:, :n]

        # normalization to 1.0
        from scipy.integrate import simps
        self.w /= np.sqrt(simps(self.w**2, self.x, axis=0))
        # number of knots
        nodes = self.knots()

        print("#" * 20)
        print("# {:5s} {:>12s}".format("nodes", "Energy [eV]"))
        print("#" * 20)
        for ii in range(n):
            print("{:-7d} {:-12.6f}".format(nodes[ii], self.e[ii]))

        return self.e, self.w

    def plot(self):
        '''
        '''
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        figure = plt.figure(
            figsize=(3.0, 3.6),
            dpi=300,
        )
        ax = plt.subplot()

        wfc_clrs = (self.e - self.e.min()) / (self.e.max() - self.e.min())

        # plot the potential
        ax.plot(self.x, self.v, color='k', lw=2.0, ls='-', alpha=0.6)
        # plot the wavefunctions
        for ii in range(self.w.shape[1]):
            ax.axhline(self.e[ii], lw=0.5, ls=':', color='k')
            ax.plot(self.x, self.e[ii] + self.w[:, ii], lw=1.0,
                    color=plt.cm.plasma_r(wfc_clrs[ii]))

        ax.set_xlabel(r'$x$ [$\AA$]', labelpad=5)
        ax.set_ylabel('Energy [eV]', labelpad=5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # # quantum square well
    # x = np.linspace(-10, 10, 2000)
    # v = np.ones_like(x) * 30
    #
    # L = 8.0
    # v[(x >= -L/2) & (x <= L/2)] = 0
    # # L = 2.0
    # # v[(x >= -L/2) & (x <= L/2)] = 1
    #
    # p = schro1d(x, v)
    # e, wfc = p.solve(n=5)
    # np.savetxt('wfc.dat', p.w)
    # print("Exact Eienvalues: ")
    # # Exact solution of square well potential
    # print((np.arange(5) + 1)**2 * np.pi**2 * ase.units._hbar**2 / 2 /
    #         ase.units._me / L**2 / 1E-20 / ase.units._e)
    # p.plot()

    # harmonic potential
    x = np.linspace(-10, 10, 2000)
    v = 0.5 * 0.50 * x**2
    p = schro1d(x, v)
    e, wfc = p.solve(n=5)
    np.savetxt('wfc.dat', p.w)
    p.plot()
