# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_TRLWE.py
# Purpose:     
#
# Author:      Junichi Sakamoto
#
# Created:     2022 Mar. 23
# Copyright:   (c) sakamoto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from __future__ import annotations
import random
import numpy as np

### For 128-bit security
# N = 635
# S = 2**-15
# Q = 32
# P = 16

### For test
N = 16
S = 2**-15
Q = 6
P = 2


def ext_product(A, B):
    """
    External product of the two N-1 degree polynomials.
    Components ordering are:
    [0] = 1, [1] = X, [2] = X^2, ..., [N-1] = X^(N-1)

    Args:
        A (_type_): _description_
        B (_type_): _description_
    """
    ### Mult
    list_X = [0] * (2*len(A))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            list_X[i+j] += a*b

    ### Reduction
    list_C = []
    for i in range(len(A)):
        list_C.append((list_X[i] - list_X[i + len(A)]) % 1)

    return list_C

class TRLWE:

    def __init__(self, N: int, sigma: float, q: int, p: int) -> None:
        """
        Set TRLWE parameters

        Args:
            N (int): Irreducible polynomial degree, which is determined by security parameter.
            sigma (float): Standard deviation of the random noise, which is determined by security parameter.
            q (int): The number of quantization bits.
            p (int): The number of plaintext space bits.
        """

        self.N = N
        self.k = 1
        self.sigma = sigma
        self.q = q
        self.q_ = 1 / (2**self.q)
        self.p = p
        self.p_ = 1 / (2**self.p)
        self.mu = None
        self.sk = None
        self.a = None
        self.e = None
        self.b = None
        self.value = None

    def rand_element(self) -> TRLWE:
        pass

    def rand_plaintext(self) -> "N-1 degree polenomial":
        self.mu = np.array([random.randint(0, (2**self.p) - 1) * self.p_ for i in range(self.N)])
        return self.mu

    def keyGen(self) -> "N-1 degree polenomial":
        self.sk = np.array([0 if random.random() < 0.5 else 1 for i in range(self.N)])
        return self.sk

    def enc(self, mu, s):
        self.a = np.array([random.randint(0, (2**self.q) - 1) * self.q_ for i in range(self.N)])
        self.e = np.array([random.normalvariate(0, self.sigma) for i in range(self.N)])
        self.b = ext_product(self.a, s) + mu + self.e
        self.value = [self.a, self.b]
        return self.value

    def dec(self, c, s):
        mu = c[1] - ext_product(c[0], s)
        for i in range(self.N):
            mu[i] = (round(mu[i]*2**self.p)) % (2**self.p)
            mu[i] = mu[i] / (2**self.p)

        return mu


def main():
    ### Setup
    trlwe = TRLWE(N, S, Q, P)
    sk = trlwe.keyGen()
    mu = trlwe.rand_plaintext()
    print(mu)

    ###Enc
    c = trlwe.enc(mu, sk)
    print(c)

    ###Dec
    MU = trlwe.dec(c, sk)
    print(MU)

    if np.all(mu == MU):
        print("OK")
    else:
        print("NG")

    ### Check additive homomorphic
    mu1 = trlwe.rand_plaintext()
    mu2 = trlwe.rand_plaintext()
    print("{} + {}".format(mu1, mu2))

    ###Enc
    c1 = trlwe.enc(mu1, sk)
    print(c1)
    c2 = trlwe.enc(mu2, sk)
    print(c2)

    ###Add
    c3 = [c1[0] + c2[0], c1[1] + c2[1]]
    print(c3)

    ###Dec
    MU = trlwe.dec(c3, sk)
    print(MU)

    if np.all((mu1 + mu2) % 1 == MU):
        print("OK")
    else:
        print("NG")

    ### Decrypt other key
    sk = trlwe.keyGen()
    MU = trlwe.dec(c3, sk)
    print(MU)
    if np.all((mu1 + mu2) % 1 == MU):
        print("OK")
    else:
        print("NG")


if __name__ == '__main__':
    main()