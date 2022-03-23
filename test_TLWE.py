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

class Torus:

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def rand_element(self) -> None:
        return Torus(random.random())

    def __add__(self, x, y):
        pass
    
    def __repr__(self) -> str:
        return "Torus({})".format(self.value)

    def __str__(self) -> str:
        return self.__repr__()

class TLWE:
    
    def __init__(self, n: int, sigma: float, q: int, p: int) -> None:
        """
        Set TLWE settings

        Args:
            n (int): Vector space dimension, which is determined by security parameter.
            sigma (float): Standard deviation of the random noise, which is determined by security parameter.
            q (int): The number of quantization bits.
            p (int): The number of plaintext space bits.
        """

        self.n = n
        self.sigma = sigma
        self.q = q
        self.q_ = 1 / (2**self.q)
        self.p = p
        self.p_ = 1 / (2**self.p)
        self.mu = None
        self.sk = None
        self.a_ = None
        self.e = None
        self.b = None
        self.value = None

    def rand_element(self) -> "TLWE":
        pass    

    def rand_plaintext(self) -> int:
        self.mu = random.randint(0, (2**self.p) - 1) * self.p_
        return self.mu

    def keyGen(self) -> "n-dimension binary array":
        self.sk = np.array([0 if random.random() < 0.5 else 1 for i in range(self.n)])
        return self.sk
        
    def enc(self, mu, s_):
        self.a_ = np.array([random.randint(0, (2**self.q) - 1) * self.q_ for i in range(self.n)])
        self.e = random.normalvariate(0, self.sigma)
        self.b = np.dot(self.a_, s_) + mu + self.e
        self.value = [self.a_, self.b]
        return self.value
    
    def dec(self, c, s_):
        mu = c[1] - np.dot(c[0], s_)
        mu = (round(mu*2**self.p)) % (2**self.p)
        mu = mu / (2**self.p)
        return mu

def main():
    ### Setup
    tlwe = TLWE(N, S, Q, P)
    sk = tlwe.keyGen()
    mu = tlwe.rand_plaintext()
    print(mu)

    ###Enc
    c = tlwe.enc(mu, sk)
    print(c)

    ###Dec
    MU = tlwe.dec(c, sk)
    print(MU)

    if mu == MU:
        print("OK")
    else:
        print("NG")


    ###Addition
    mu1 = tlwe.rand_plaintext()
    mu2 = tlwe.rand_plaintext()
    print("{} + {}".format(mu1, mu2))

    ###Enc
    c1 = tlwe.enc(mu1, sk)
    print(c1)
    c2 = tlwe.enc(mu2, sk)
    print(c2)

    ###Add
    c3 = [c1[0] + c2[0], c1[1] + c2[1]]
    print(c3)
    
    ###Dec
    MU = tlwe.dec(c3, sk)
    print(MU)

    if (mu1 + mu2) % 1 == MU:
        print("OK")
    else:
        print("NG")

    
if __name__ == '__main__':
    main()
