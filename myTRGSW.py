# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        myTRGSW.py
# Purpose:
#
# Author:      Junichi Sakamoto
#
# Created:     2022 Mar. 29
# Copyright:   (c) sakamoto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from __future__ import annotations
from array import ArrayType

from soupsieve import select
from myTRLWE import TorusRing, TRLWE, IntRing
from myTLWE import Torus
import random
import numpy as np


class TRGSW:
    """
    Torus ring GSW

    Class variables:

    """
    B = 0
    Bbit = 0
    l = 0

    @classmethod
    def init(cls, N: int, sigma: float, p: int, B: int, l: int):
        TRLWE.init(N, sigma, p)
        cls.B = B
        cls.Bbit = len(bin(cls.B)[2:]) - 1
        cls.l = l

    def __init__(self, value: List = None) -> TRGSW:
        self.value = value

    def __add__(self, x: TRGSW) -> TRGSW:
        pass

    def __repr__(self) -> str:
        return "TRGSW({})".format(self.value)

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def rand_element() -> "TRGSW":
        pass

    @staticmethod
    def rand_plaintext() -> IntRing:
        return IntRing.rand_element(2**TRLWE.p - 1)

    @staticmethod
    def rand_plaintext_bin() -> IntRing:
        r = random.randint(0, 1)
        if r:
            return IntRing.getOne()
        else:
            return IntRing.getZero()

    @staticmethod
    def keyGen() -> IntRing:
        return TRLWE.keyGen()

    @staticmethod
    def getGTmatrix():
        B = [1 << (Torus.q - (i+1)*TRGSW.Bbit) for i in range(TRGSW.l)]
        G = []
        for i in range(TRLWE.k + 1):
            zeros = [0] * (TRGSW.l * TRLWE.k)
            zeros[i*TRGSW.l:i*TRGSW.l] = B
            G.append(zeros)
        G = np.array(G).reshape(TRLWE.k + 1, (TRLWE.k + 1) * TRGSW.l)
        return G.T

    @staticmethod
    def Ginv(trlwe: TRLWE) -> "IntRing array":
        list_Ginv = []
        for torus_poly in trlwe.value:
            for i in range(TRGSW.l):
                list_intpoly = []
                for coeff in torus_poly.value:
                    ### Unsigned representation
                    coeff += 2**(Torus.q - TRGSW.l*TRGSW.Bbit - 1)
                    coeff = int(coeff)
                    list_intpoly.append(coeff >> (Torus.q - TRGSW.Bbit*(i+1)) & (TRGSW.B - 1))
                list_Ginv.append(IntRing(np.array(list_intpoly)))

        return np.array(list_Ginv)

    @staticmethod
    def enc(m: IntRing, s_: IntRing) -> TRGSW:
        """_summary_

        Args:
            m (IntRing): Integer polynomial ring over Zp
            s_ (IntRing): Integer polynomial ring over Binary

        Returns:
            TRGSW: _description_
        """
        Z = np.array([TRLWE.enc(TorusRing.getZero(), s_).value for i in range((TRLWE.k+1) * TRGSW.l)]).reshape((TRLWE.k + 1) * TRGSW.l, TRLWE.k+1)
        ### Make gadget matrix
        GT = TRGSW.getGTmatrix()
        return TRGSW(Z + m@GT)

    @staticmethod
    def dec(c: TRGSW, s_: IntRing):
        """
        Plaintext space p must be less than B.
        """
        mp = TRLWE.dec_wo_round(TRLWE(c.value[-TRGSW.l]), s_)

        ### Round
        list_t = []
        for val in mp.value:
            ## Rounding
            temp = val + (1 << (Torus.q - TRGSW.Bbit - 1))
            temp = temp << TRGSW.Bbit >> Torus.q
            list_t.append(temp & (2**TRLWE.p - 1))

        return IntRing(list_t)

    @staticmethod
    def externalProduct(trlwe: TRLWE, trgsw: TRGSW) -> TRLWE:
        Ginv = TRGSW.Ginv(trlwe)
        prod = Ginv @ trgsw.value   # Prod is TRLWE
        ### Reduction for fixed-point number
        list_red = []
        for elem in prod:
            list_red.append(elem)
        return TRLWE(list_red)

    @staticmethod
    def CMUX(sel: TRGSW, c0: TRLWE, c1: TRLWE) -> TRLWE:
        """Controlled MUX

        Args:
            sel (TGSW): Selector signal in TGSW form encrypting 0/1.
            c0 (TLWE): CMUX output when sel = 0.
            c1 (TLWE): CMUX output when sel = 1.

        Returns:
            TLWE: _description_
        """
        return TRGSW.externalProduct(c1 - c0, sel) + c0


def main():

    ### For test
    N = 2
    S = 2**-25
    P = 5
    B = 64
    l = 3

    ### Flatten test
    TRGSW.init(N, S, P, B, l)
    sk = TRLWE.keyGen()
    print(f"sk: {sk}")
    m = TRLWE.rand_plaintext()
    print(f"m: {m} ({m.toInt()})")
    c = TRLWE.enc(m, sk)
    print(f"c: {c}")

    Ginv = TRGSW.Ginv(c)
    G = TRGSW.getGTmatrix()

    a = Ginv @ G
    print(f"Flatten: {a}")

    mp = TRLWE.dec(TRLWE(a), sk)
    print(f"m': {mp} ({mp.toInt()})")

    if mp == m:
        print("OK\n")
    else:
        print("NG\n")

    ### Homomorphic multiplication test
    m1 = TRLWE.rand_plaintext()
    m2 = TRGSW.rand_plaintext()
    print(f"m1: {m1} ({m1.toInt()})")
    print(f"m2: {m2}")

    c1 = TRLWE.enc(m1, sk)
    c2 = TRGSW.enc(m2, sk)
    print(f"c1: {c1}")
    print(f"c2: {c2}")
    c3 = TRGSW.externalProduct(c1, c2)
    print(f"c3: {c3}")

    m3 = TRLWE.dec(c3, sk)
    print(f"m3: {m3} ({m3.toInt()})")

    if m3 == m2*m1:
        print("OK\n")
    else:
        print("NG\n")

    ### CMUX test
    m1 = TRLWE.rand_plaintext()
    m2 = TRLWE.rand_plaintext()
    print(f"m1: {m1} ({m1.toInt()})")
    print(f"m2: {m2} ({m2.toInt()})")
    c1 = TRLWE.enc(m1, sk)
    c2 = TRLWE.enc(m2, sk)
    print(f"c1: {c1}")
    print(f"c2: {c2}")

    sel = TRGSW.rand_plaintext_bin()
    print(f"sel: {sel}")
    c_sel = TRGSW.enc(sel, sk)
    print(f"c_sel: {c_sel}")

    res = TRGSW.CMUX(c_sel, c1, c2)
    print(f"res: {res}")

    selected = TRLWE.dec(res, sk)
    print(f"selected: {selected.toInt()}")

    sel = sel.value[0]
    if sel and selected == m2:
        print("OK")
    elif sel == 0 and selected == m1:
        print("OK")
    else:
        print("NG")

    ### Dec test
    m1 = TRGSW.rand_plaintext()
    print(f"m1: {m1}")
    c1 = TRGSW.enc(m1, sk)
    print(f"c1: {c1}")

    mp = TRGSW.dec(c1, sk)
    print(f"m': {mp}")

    if mp == m1:
        print("OK\n")
    else:
        print("NG\n")

if __name__ == '__main__':
    main()
