# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_TLWE.py
# Purpose:     temporary test code
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
P = 13

class Torus:
    """
    Discritized torus, where the quantization is done with 32 bits.
    Using int (32 bits) variables, mod 1 operation becomes overflows.

    Class variables:
        q (int) = 32: The number of quantization bits.
    """
    q = 32
    q_ = 1 / (2**q)
    mask = 2**q - 1

    # @classmethod
    # def init(cls, q):
    #     cls.__q = q

    @classmethod
    def rand_element(cls) -> Torus:
        return Torus(random.randint(0, 2**cls.q - 1))
        #return random.random()

    def __init__(self, value: int = 0) -> None:
        self.value = value & self.mask

    def __add__(self, x) -> Torus:
        if type(x) == int or type(x) == float or type(x) == np.int64:
            return Torus(self.value + x)
        elif type(x) == Torus:
            return Torus(self.value + x.value)
        else:
            raise Exception("Undefined type")

    def __sub__(self, x: Torus) -> Torus:
        if type(x) == int or type(x) == float:
            return Torus(self.value - x)
        elif type(x) == Torus:
            return Torus(self.value - x.value)
        else:
            raise Exception("Undefined type")

    def __mul__(self, x) -> TLWE:
        if type(x) == int or type(x) == float or type(x) == np.int64:
            #print(f'{self.value}*{x}')
            return Torus(np.int64((np.uint64(self.value) * np.uint64(x)) & np.uint64(self.mask)))
        elif type(x) == Torus:
            return Torus(self.value * x.value)
        else:
            raise Exception("Undefined type")

    def __rshift__(self, x: int) -> Torus:
        if type(x) == int:
            return Torus(self.value >> x)
        else:
            raise Exception("Undefined type")

    def __lshift__(self, x: int) -> Torus:
        if type(x) == int:
            return Torus(self.value << x)
        else:
            raise Exception("Undefined type")

    def __and__(self, x: int) -> Torus:
        if type(x) == int:
            return Torus(self.value & x)
        else:
            raise Exception("Undefined type")

    def __eq__(self, x: Torus):
        if type(x) == Torus:
            return self.value == x.value
        else:
            raise Exception("Undefined type")
     
    
    def __repr__(self) -> str:
        return "Torus({})".format(self.value)

    def __str__(self) -> str:
        return self.__repr__()

    def toInt(self) -> int:
        return (Torus.round(self) >> (Torus.q - TLWE.p)).value

    @staticmethod
    def round(tr: Torus) -> Torus:
        ## Rounding
        temp = tr + (1 << (Torus.q - TLWE.p - 1))
        temp = temp >> (Torus.q - TLWE.p)
        temp = temp << (Torus.q - TLWE.p)
        return temp

class TLWE:
    """
    Torus LWE

    Class variables:
        p (int): The number of plaintext space bits.
        n (int): Vector space dimension, which is determined by security parameter.
        sigma (float): Standard deviation of the random noise, which is determined by security parameter.
    """
    n = 0
    sigma = 0
    p = 0
    p_ = 0

    @classmethod
    def init(cls, n: int, sigma: float, p: int):
        cls.n = n
        cls.sigma = sigma
        cls.p = p
        cls.p_ = 1 / (2**cls.p)

    def __init__(self, value: List = None) -> TLWE:
        self.value = value

    def __add__(self, x: TLWE) -> TLWE:
        return TLWE(self.value + x.value)

    def __sub__(self, x: TLWE) -> TLWE:
        return TLWE(self.value - x.value)

    def __rsub__(self, x: TLWE) -> TLWE:
        return TLWE(self.value - x.value)

    def __mul__(self, rhs: int) -> TLWE:
        if type(rhs) == int or type(rhs) == np.int64:
            return TLWE(np.array([elem * rhs for elem in self.value]))
        else:
            raise Exception("IntRing rshift type exception")

    def __repr__(self) -> str:
        return "TLWE({})".format(self.value)

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def rand_element() -> "TLWE":
        sk = TLWE.keyGen()
        mu = TLWE.rand_plaintext()
        c = TLWE.enc(mu, sk)
        return c

    @staticmethod
    def rand_plaintext() -> Torus:
        #return Torus(random.randint(0, (2**TLWE.p) - 1) * TLWE.p_)
        return Torus(random.randint(0, (2**TLWE.p) - 1) << (Torus.q - TLWE.p))

    @staticmethod
    def keyGen() -> "n-dimension binary array":
        sk = np.array([0 if random.random() < 0.5 else 1 for i in range(TLWE.n)])
        return sk

    @staticmethod
    def enc(mu: Torus, s_: "binary array") -> TLWE:
        #a_ = np.array([random.randint(0, (2**mu.q) - 1) * mu.q_ for i in range(TLWE.n)])
        a_ = np.array([Torus(random.randint(0, (2**Torus.q) - 1)) for i in range(TLWE.n)])
        e = round(random.normalvariate(0, TLWE.sigma) * 2**Torus.q)
        b = np.dot(a_, s_) + mu.value + e
        return TLWE(np.array(np.append(a_, b)))

    @staticmethod
    def dec(c: TLWE, s_: "binary array") -> Torus:
        mu = TLWE.dec_wo_round(c, s_) 

        ### round
        mu = mu + (1 << (Torus.q - TLWE.p - 1))
        mu = mu >> (Torus.q - TLWE.p)
        mu = mu << (Torus.q - TLWE.p)

        return mu

    @staticmethod
    def dec_wo_round(c: TLWE, s_: "binary array") -> Torus:
        mu = c.value[-1] - np.dot(c.value[:-1], s_)

        return mu

def main():
    ### Setup
    #Torus.init(Q)
    TLWE.init(N, S, P)
    sk = TLWE.keyGen()
    mu = TLWE.rand_plaintext()
    print("mu: {}".format(mu))
    print("sk: {}".format(sk))


    ###Enc
    c = TLWE.enc(mu, sk)
    print("c: {}".format(c))

    ###Dec
    res = TLWE.dec(c, sk)
    print("res: {}".format(res))

    if mu == res:
        print("OK")
    else:
        print("NG")


    ###Addition
    mu1 = TLWE.rand_plaintext()
    mu2 = TLWE.rand_plaintext()
    print("{} + {}".format(mu1, mu2))

    ###Enc
    c1 = TLWE.enc(mu1, sk)
    print(f"c1: {c1}")
    c2 = TLWE.enc(mu2, sk)
    print(f"c2: {c2}")

    ###Add
    c3 = c1 + c2
    print(f"c3: {c3}")
    
    ###Dec
    MU = TLWE.dec(c3, sk)
    print(MU)

    if mu1 + mu2 == MU:
        print("OK")
    else:
        print("NG")

    
if __name__ == '__main__':
    main()
