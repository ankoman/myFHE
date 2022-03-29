# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        myTGSW.py
# Purpose:     
#
# Author:      Junichi Sakamoto
#
# Created:     2022 Mar. 25
# Copyright:   (c) sakamoto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from __future__ import annotations
from myTLWE import N, Torus
from myTLWE import TLWE
import random
import numpy as np


class TGSW:
    """
    Torus GSW

    Class variables:

    """
    B = 0
    Bbit = 0
    l = 0

    @classmethod
    def init(cls, n: int, sigma: float, p: int, B: int, l: int):
        TLWE.n = n
        TLWE.sigma = sigma
        TLWE.p = p
        TLWE.p_ = 1 / (2**TLWE.p)
        cls.B = B
        cls.Bbit = len(bin(cls.B)[2:]) - 1
        cls.l = l

    def __init__(self, value: List = None) -> TGSW:
        self.value = value

    def __add__(self, x: TLWE) -> TGSW:
        pass

    def __repr__(self) -> str:
        return "TGSW({})".format(self.value)

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def rand_element() -> "TGSW":
        pass

    @staticmethod
    def rand_plaintext() -> int:
        return random.randint(0, TLWE.p**2 - 1) << Torus.q

    @staticmethod
    def keyGen() -> "n-dimension binary array":
        return TLWE.keyGen()


    @staticmethod
    def getGTmatrix():
        B = [1 << (Torus.q - (i+1)*TGSW.Bbit) for i in range(TGSW.l)]
        G = []
        for i in range(TLWE.n + 1):
            zeros = [0] * (TGSW.l * TLWE.n)
            zeros[i*TGSW.l:i*TGSW.l] = B
            G.append(zeros)
        G = np.array(G).reshape(TLWE.n + 1, (TLWE.n + 1) * TGSW.l)
        return G.T
    
    @staticmethod
    def Ginv(tlwe: TLWE):
        list_Ginv = []
        for elem in tlwe:
            elem += 2**(Torus.q - TGSW.l*TGSW.Bbit - 1)
            elem = int(elem)
            for i in range(TGSW.l):
                list_Ginv.append(elem >> (Torus.q - TGSW.Bbit*(i+1)) & (TGSW.B - 1))

        return np.array(list_Ginv)

    @staticmethod
    def enc(m: int, s_: "binary array") -> TGSW:
        Z = np.array([TLWE.enc(Torus(0), s_).value for i in range((TLWE.n+1) * TGSW.l)]).reshape((TLWE.n + 1) * TGSW.l, TLWE.n+1)
        ### Make gadget matrix
        GT = TGSW.getGTmatrix()
        return Z + m*GT

    @staticmethod
    def dec(c, s_) -> Torus:
        pass
    
    @staticmethod
    def externalProduct(tlwe: TLWE, tgsw: TGSW) -> TLWE:
        Ginv = TGSW.Ginv(tlwe.value)
        prod = Ginv @ tgsw
        ### Reduction for fixed-point number
        list_red = []
        for elem in prod:
            list_red.append(elem >> Torus.q)
        return TLWE(list_red)


def main():

    ### For test
    N = 625
    S = 2**-15
    Q = 6
    P = 10
    B = 64
    l = 3

    ### Flatten test
    TGSW.init(N, S, P, B, l)
    sk = TLWE.keyGen()
    #print(f"sk: {sk}")
    m = TLWE.rand_plaintext()
    print(f"m: {m} ({m.toInt()})")
    c = TLWE.enc(m, sk)
    #print(f"TLWE: {c}")

    Ginv = TGSW.Ginv(c.value)
    G = TGSW.getGTmatrix()

    a = Ginv @ G
    #print(f"Flatten: {a}")

    mp = TLWE.dec(TLWE(a), sk)
    print(f"m': {mp} ({mp.toInt()})")

    if mp == m:
        print("OK")
    else:
        print("NG")

    ### multiplicative homomorhic test
    m1 = TLWE.rand_plaintext()
    m2 = TGSW.rand_plaintext()
    print(f"m1: {m1} ({m1.toInt()})")
    print(f"m2: {m2} ({m2 >> Torus.q})")

    c1 = TLWE.enc(m1, sk)
    c2 = TGSW.enc(m2, sk)
    #print(f"c1: {c1}")
    #print(f"c2: {c2}")
    c3 = TGSW.externalProduct(c1, c2)
    #print(f"c3: {c3}")

    m3 = TLWE.dec(c3, sk)
    print(f"m3: {m3} ({m3.toInt()})")

    if m3 == Torus(m1.value*m2 >> Torus.q):
        print("OK")
    else:
        print("NG")



if __name__ == '__main__':
    main()