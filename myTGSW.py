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
from myTLWE import Torus
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

    def __add__(self, x: TGSW) -> TGSW:
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
        return random.randint(0, 2**TLWE.p - 1)

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
        for elem in tlwe.value:
            elem += 2**(Torus.q - TGSW.l*TGSW.Bbit - 1)
            for i in range(TGSW.l):
                ### Unsigned representation
                list_Ginv.append(elem >> (Torus.q - TGSW.Bbit*(i+1)) & (TGSW.B - 1))

        return np.array(list_Ginv)

    @staticmethod
    def enc(m: int, s_: "binary array") -> TGSW:
        Z = np.array([TLWE.enc(Torus(0), s_).value for i in range((TLWE.n+1) * TGSW.l)]).reshape((TLWE.n + 1) * TGSW.l, TLWE.n+1)
        ### Make gadget matrix
        GT = TGSW.getGTmatrix()
        return TGSW(Z + m*GT)

    @staticmethod
    def dec(c, s_) -> int:
        """
        Plaintext space p must be less than B.
        """
        mp = TLWE.dec_wo_round(TLWE(c.value[-TGSW.l]), s_)
        mp = mp.value + (1 << (Torus.q - TGSW.Bbit - 1))
        return ((mp << (TGSW.Bbit)) >> Torus.q) & (2**TLWE.p - 1)
    
    @staticmethod
    def externalProduct(tlwe: TLWE, tgsw: TGSW) -> TLWE:
        Ginv = TGSW.Ginv(tlwe)
        prod = Ginv @ tgsw.value

        return TLWE(prod)

    @staticmethod
    def CMUX(sel: TGSW, c0: TLWE, c1: TLWE) -> TLWE:
        """Controlled MUX

        Args:
            sel (TGSW): Selector signal in TGSW form encrypting 0/1.
            c0 (TLWE): CMUX output when sel = 0.
            c1 (TLWE): CMUX output when sel = 1.

        Returns:
            TLWE: _description_
        """
        return TGSW.externalProduct(c1 - c0, sel) + c0


def main():

    ### For test
    N = 2
    S = 2**-15
    P = 2
    B = 64
    l = 3

    ### Flatten test
    TGSW.init(N, S, P, B, l)
    sk = TLWE.keyGen()
    print(f"sk: {sk}")
    m = TLWE.rand_plaintext()
    print(f"m: {m} ({m.toInt()})")
    c = TLWE.enc(m, sk)
    #print(f"TLWE: {c}")

    Ginv = TGSW.Ginv(c)
    G = TGSW.getGTmatrix()

    a = Ginv @ G
    #print(f"Flatten: {a}")

    mp = TLWE.dec(TLWE(a), sk)
    print(f"m': {mp} ({mp.toInt()})")

    if mp == m:
        print("OK\n")
    else:
        print("NG\n")

    ### multiplicative homomorhic test
    m1 = TLWE.rand_plaintext()
    m2 = TGSW.rand_plaintext()
    print(f"m1: {m1} ({m1.toInt()})")
    print(f"m2: {m2}")

    c1 = TLWE.enc(m1, sk)
    c2 = TGSW.enc(m2, sk)
    print(f"c1: {c1}")
    print(f"c2: {c2}")
    c3 = TGSW.externalProduct(c1, c2)
    print(f"c3: {c3}")

    m3 = TLWE.dec(c3, sk)
    print(f"m3: {m3} ({m3.toInt()})")

    if m3 == m1*m2:
        print("OK\n")
    else:
        print("NG\n")

    ### CMUX test
    m1 = TLWE.rand_plaintext()
    m2 = TLWE.rand_plaintext()
    print(f"m1: {m1} ({m1.toInt()})")
    print(f"m2: {m2} ({m2.toInt()})")
    c1 = TLWE.enc(m1, sk)
    c2 = TLWE.enc(m2, sk)
    print(f"c1: {c1}")
    print(f"c2: {c2}")

    sel = random.randint(0, 1)
    print(f"sel: {sel}")
    c_sel = TGSW.enc(sel, sk)
    print(f"c_sel: {c_sel}")

    res = TGSW.CMUX(c_sel, c1, c2)
    print(f"res: {res}")

    selected = TLWE.dec(res, sk)
    print(f"selected: {selected.toInt()}")

    if sel*(m2.value - m1.value) + m1.value == selected.value:
        print("OK\n")
    else:
        print("NG\n")

    ### Dec test
    m1 = TGSW.rand_plaintext()
    print(f"m1: {m1}")
    c1 = TGSW.enc(m1, sk)
    print(f"c1: {c1}")

    mp = TGSW.dec(c1, sk)
    print(f"m': {mp}")

    if mp == m1:
        print("OK\n")
    else:
        print("NG\n")

if __name__ == '__main__':
    main()