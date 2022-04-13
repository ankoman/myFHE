# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        myTFHE.py
# Purpose:
#
# Author:      Junichi Sakamoto
#
# Created:     2022 Apr. 1
# Copyright:   (c) sakamoto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from __future__ import annotations
from myTRGSW import TRGSW
from myTRLWE import TorusRing, TRLWE, IntRing
from myTLWE import Torus, TLWE
import random
import numpy as np
import time


class TFHE:
    """
    TFHE

    """

    base = 0
    basebit = 0
    t = 0

    @classmethod
    def init(cls, n: int, s: float, N: int, S: float, P: int, B: int, l: int, base: int, t: int):
        cls.base = base
        cls.t = t
        cls.basebit = len(bin(cls.base)[2:]) - 1

        TRGSW.init(N, S, P, B, l)   # with TRLWE
        TLWE.init(n, s, P)

    def __init__(self, value: List = None):
        raise Exception("Not implemented")

    @staticmethod
    def X_pow(exp: int) -> IntRing:
        """
        Below exponentiation must be implemented by binary method to decrease its computational cost.
        """
        X = [0 for i in range(TorusRing.N - 1)]
        X.insert(1, 1)
        X = IntRing(X)
        return X**exp

    @staticmethod
    def BlindRotate(tlwe: TLWE, tv: TRLWE, bsk: TRGSW, skr):
        a = tlwe.value[:-1]
        b = tlwe.value[-1]
        print("\n####Start BlindRotate")

        minusb_t = 2*TRLWE.N - (b  >> (Torus.q - TRLWE.Nbit - 1)).value # b~ = floor(2N*b) mod 2N
        qi = tv * TFHE.X_pow(minusb_t)
        print(f"X^-b_t = X^{minusb_t} = {qi}\n")

        for ai, bski in zip(a, bsk):
            a_t = (ai.value + (1 << (Torus.q - TRLWE.Nbit - 2))) >> (Torus.q - TRLWE.Nbit - 1) # a~ = round(2N*b) mod 2N
            qi = TRGSW.CMUX(bski, qi, qi * TFHE.X_pow(a_t))
            print(f"a_t = {a_t}")


            print(f'qi: {qi}')
            res = TRLWE.dec(qi, skr)
            print(f"res: {res}")

        print("####End BlindRotate\n")
        return qi

    @staticmethod
    def Rotate(tlwe: TLWE, s):
        a = tlwe.value[:-1]
        b = tlwe.value[-1]

        b_t = 2*TRLWE.N - (b >> (Torus.q - TRLWE.Nbit - 1)).value  # b~ = floor(2N*b) mod 2N
        Xi = TFHE.X_pow(b_t)
        #print(f"X^-b_t = X^{b_t} = {Xi}\n")

        for ai, si in zip(a, s):
            a_t = (ai.value + (1 << (Torus.q - TRLWE.Nbit - 2))) >> (Torus.q - TRLWE.Nbit - 1)  # a~ = round(2N*b) mod 2N
            if si:
                #print(f"{ai} ({ai.toInt()}) * {si}, {a_t}")
                b_t += a_t
                Xi = Xi * TFHE.X_pow(a_t)
        #print(b_t)
        return Xi

    @staticmethod
    def genBootstrappingKey(sk: list[int], skr: IntRing):
        bsk = [TRGSW.enc(IntRing.getOne() * sk[i], skr) for i in range(TLWE.n)]
        return bsk


    @staticmethod
    def genKeySwitchingKey(sk_lvl0: list[int], sk_lvl1: list[int]) -> list[list[TLWE]]:
        list_ksk = []
        for si in sk_lvl1:
            list_row = []
            for j in range(TFHE.t):
                ksk = TLWE.enc(Torus(si*(1 << (Torus.q - (j+1)*TFHE.basebit))), sk_lvl0)
                list_row.append(ksk)
            list_ksk.append(np.array(list_row))

        return np.array(list_ksk)
                
    @staticmethod
    def keySwitch_naive(tlwe_lvl1: TLWE, ksk: list[list[TLWE]]):
        list_t = [Torus(0)] * TLWE.n
        list_t.append(tlwe_lvl1.value[-1])
        res = TLWE(np.array(list_t))

        round_offset = 1 << (Torus.q - (1 + TFHE.basebit * TFHE.t))
        for i in range(TRLWE.N):
            ai = tlwe_lvl1.value[i].value + round_offset
            for j in range(TFHE.t):
                aij = (ai >> (Torus.q - (j+1)*TFHE.basebit)) & (2**TFHE.basebit - 1)
                res -= ksk[i][j]*aij
        
        return res

    @staticmethod
    def SampleExtractIndex0(trlwe: TRLWE) -> TLWE:
        list_a = []
        list_a.append(Torus(trlwe.value[0].value[0]))
        for i in range(TRLWE.N - 1, 0, -1):
            list_a.append(Torus((2**Torus.q) - trlwe.value[0].value[i]))
        list_a.append(Torus(trlwe.value[-1].value[0]))

        return TLWE(list_a)

    @staticmethod
    def HomNAND(inA: TLWE, inB: TLWE):
        pass


def main():

    ### Setup
    N = 4
    n = N
    S = 0#2**-25
    s = 0#2**-15
    P = 2
    B = 64
    l = 3
    base = 4
    t = 8

    TFHE.init(n, s, N, S, P, B, l, base, t)

    ### Test rotate
    # sk = TRLWE.keyGen()
    # print(f"sk: {sk}")
    # tv = TorusRing([i for i in range(TorusRing.N)])
    # tv = TRLWE.enc(tv, sk, True)
    # print(f"tv: {tv}")



    sk = TLWE.keyGen()
    skr = TRGSW.keyGen()
    mu = TLWE.rand_plaintext()
    tv = TRLWE.rand_plaintext()
    #tv = IntRing(np.array([i << 30 for i in range(TRLWE.N)])) # must be in Tp
    bsk = TFHE.genBootstrappingKey(sk, skr)
    print(f"mu: {mu} ({mu.toInt()})")
    print(f"exp: {(mu.value >> (Torus.q - TRLWE.Nbit - 1))}")
    print(f"tv: {tv}")
    print(f"sk: {sk}")
    print(f"skr: {skr}")
    #print(f"bsk: {bsk}")


    ###BR
    c = TLWE.enc(mu, sk)
    print("c: {}".format(c))
    c_tv = TRLWE.enc(tv, IntRing.getZero(), explicit=True)
    print(f"c_tv: {c_tv}")

    c_X_pow_mu = TFHE.BlindRotate(c, c_tv, bsk, skr)
    res = TRLWE.dec(c_X_pow_mu, skr)
    print(f"c_X_pow_mu: {c_X_pow_mu}")
    print(f"res: {res}")

    ##SE
    extracted = TFHE.SampleExtractIndex0(c_X_pow_mu)
    constant_term =TLWE.dec(extracted, skr.value)
    print(constant_term)
  
    ##KS
    ksk = TFHE.genKeySwitchingKey(sk, skr.value)
    #print(ksk)

    res = TFHE.keySwitch_naive(extracted, ksk)
    res = TLWE.dec(res, sk)
    print(res)


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("\nelapsed_time:{0}".format(elapsed_time) + "[sec]")
