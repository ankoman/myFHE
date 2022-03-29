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
from myTLWE import Torus

### For 128-bit security
# N = 635
# S = 2**-15
# Q = 32
# P = 16


class TorusRing:
    """
    T_N[X]: Polenomial ring over the Torus.
    T_N[X] = T[X]/(X^N+1)
    X^N+1 is the M-(2N)th cyclotomic polynomial.
    """
    N = 0

    @classmethod
    def init(cls, N: int):
        cls.N = N

    def __init__(self, value: List = None) -> TorusRing:
        list_t = []
        for elem in value:
            list_t.append(elem & 2**Torus.q - 1)
        self.value = np.array(list_t)

    def __add__(self, rhs: TorusRing) -> TorusRing:
        return TorusRing(self.value + rhs.value)

    def __sub__(self, rhs: TorusRing) -> TorusRing:
        return TorusRing(self.value - rhs.value)

    def __repr__(self) -> str:
        return f"TorusRing({self.value})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.value)

    def __eq__(self, x):
        #print(self.value == x.value)
        return np.all(self.value == x.value)

    @staticmethod
    def round(tr: TorusRing):
        list_t = []
        for val in tr.value:
            ## Rounding
            temp = val + (1 << (Torus.q - TRLWE.p - 1))
            temp = temp >> (Torus.q - TRLWE.p)
            list_t.append(int(temp << (Torus.q - TRLWE.p)))
        
        return TorusRing(np.array(list_t))

    @staticmethod
    def rand_element() -> TorusRing:
        return TorusRing(np.array([random.randint(0, (2**TRLWE.p) - 1) << (Torus.q - TRLWE.p) for i in range(TorusRing.N)]))

    @staticmethod
    def ext_product(A: TorusRing, B: "Z_N[X] polynomial"):
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
        for i, a in enumerate(A.value):
            for j, b in enumerate(B):
                list_X[i+j] += a*b

        ### Reduction
        list_C = []
        for i in range(len(A)):
            list_C.append((list_X[i] - list_X[i + len(A)]))

        return TorusRing(np.array(list_C))

class TRLWE:
    """
    TRLWE parameters

    Params.:
        N (int): Irreducible polynomial degree, which is determined by security parameter.
        sigma (float): Standard deviation of the random noise, which is determined by security parameter.
        p (int): The number of plaintext space bits.
    """

    N = 0
    k = 1
    sigma = 0
    p = 0
    p_ = 0

    @classmethod
    def init(cls, N: int, sigma: float, p: int):
        TorusRing.init(N)
        cls.N = N
        cls.sigma = sigma
        cls.p = p
        cls.p_ = 1 / (2**cls.p)

    def __init__(self, value: List = None) -> TRLWE:
        self.value = value

    def __repr__(self) -> str:
        return f"TRLWE({self.value})"

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, rhs: TorusRing) -> TRLWE:
        return TRLWE(self.value + rhs.value)

    @staticmethod
    def rand_element(self) -> TRLWE:
        pass

    @staticmethod
    def rand_plaintext() -> TorusRing:
        return TorusRing.rand_element()

    @staticmethod
    def keyGen() -> "N-1 degree binary polynomial":
        return np.array([0 if random.random() < 0.5 else 1 for i in range(TRLWE.N)])

    @staticmethod
    def enc(mu: TorusRing, s: "binary polynomial"):
        a = TorusRing.rand_element()    # k = 1
        e = np.array([round(random.normalvariate(0, TRLWE.sigma) * 2**Torus.q) for i in range(TRLWE.N)])
        b = TorusRing.ext_product(a, s) + mu + TorusRing(e)
        return TRLWE(np.array(np.append(a, b)))

    @staticmethod
    def dec(c: TRLWE, s: "binary polynomial"):
        mu = c.value[1] - TorusRing.ext_product(c.value[0], s) # k = 1
        return TorusRing.round(mu)


def main():

    N = 2
    S = 2**-15
    Q = 6
    P = 1

    ### Setup
    TRLWE.init(N, S, P)
    sk = TRLWE.keyGen()
    print(f"sk: {sk}")
    mu = TRLWE.rand_plaintext()
    print(f"mu: {mu}")

    ###Enc
    c = TRLWE.enc(mu, sk)
    print(f"c: {c}")

    ###Dec
    mup = TRLWE.dec(c, sk)
    print(f"mu': {mup}")

    if mu == mup:
        print("OK")
    else:
        print("NG")

    ## Check additive homomorphic
    mu1 = TRLWE.rand_plaintext()
    mu2 = TRLWE.rand_plaintext()
    print(f"mu1 + mu2: {mu1} + {mu2}")

    ###Enc
    c1 = TRLWE.enc(mu1, sk)
    c2 = TRLWE.enc(mu2, sk)
    print(f"c1 + c2: {c1} + {c2}")

    ###Add
    c3 = c1 + c2
    print(f"c3: {c3}")

    ###Dec
    mu3 = TRLWE.dec(c3, sk)
    print(f"mu3: {mu3}")

    if (mu1 + mu2) == mu3:
        print("OK")
    else:
        print("NG")


if __name__ == '__main__':
    main()