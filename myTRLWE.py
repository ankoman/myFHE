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
import math
import numpy as np
from myTLWE import TLWE, Torus

### For 128-bit security
# N = 635
# S = 2**-15
# Q = 32
# P = 16

class IntRing():
    """
    Integer Ring
    Components ordering are:
    [0] = 1, [1] = X, [2] = X^2, ..., [N-1] = X^(N-1)
    """
    N = 0

    @classmethod
    def init(cls, N: int):
        cls.N = N

    def __init__(self, value: List = None) -> IntRing:
        # list_t = []
        # for elem in value:
        #     list_t.append(elem & 2**TRLWE.p - 1)
        # self.value = np.array(list_t)
        self.value = value

    def __add__(self, rhs: IntRing) -> IntRing:
        return IntRing(self.value + rhs.value)

    def __sub__(self, rhs: IntRing) -> IntRing:
        return IntRing(self.value - rhs.value)
    
    def __mul__(self, rhs) -> IntRing:
        if type(rhs) == int or type(rhs) == np.int64:
            return IntRing(np.array([coeff * rhs for coeff in self.value]))
        elif type(rhs) == np.ndarray:
            return IntRing(self.value*rhs)
        elif type(rhs) == TorusRing:
            res = TorusRing.ext_product(rhs, self.value)
           # print(f"{self} * {rhs} = {res}")
            return res
        elif type(rhs) == IntRing:
            res = TorusRing.ext_product(rhs, self.value)
            #print(f"{self} * {rhs} = {res}")
            return IntRing(res.value)
        else:
            print(type(rhs))
            raise Exception("IntRing mul type exception")

    def __pow__(self, rhs) -> TorusRing:
        t = self
        if type(rhs) == int:
            for i in range(rhs - 1):
                t = TorusRing.ext_product(t, self.value)
            return t
        else:
            print(type(rhs))
            raise Exception("IntRing pow type exception")

    def __matmul__(self, rhs) -> "IntRing array":
        list_ans = []
        for row in rhs:
            list_row = []
            for elem in row:
                list_row.append(self*elem)
            list_ans.append(list_row)
        return np.array(list_ans)

    def __lshift__(self, rhs) -> IntRing:
        if type(rhs) == int:
            return IntRing(np.array([coeff << rhs for coeff in self.value]))
        else:
            raise Exception("IntRing lshift type exception")

    def __rshift__(self, rhs) -> IntRing:
        if type(rhs) == int:
            return IntRing(np.array([coeff >> rhs for coeff in self.value]))
        else:
            raise Exception("IntRing rshift type exception")

    def __repr__(self) -> str:
        return f"IntRing({self.value})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __eq__(self, x):
        return np.all(self.value == x.value)

    @staticmethod
    def rand_element(coeff_max) -> IntRing:
        return IntRing(np.array([random.randint(0, coeff_max) for i in range(IntRing.N)]))

    @staticmethod
    def getZero():
        return IntRing(np.array([0 for i in range(IntRing.N)]))

    @staticmethod
    def getOne():
        one = np.array([0 for i in range(IntRing.N)])
        one[0] = 1
        return IntRing(one)

class TorusRing:
    """
    T_N[X]: Polenomial ring over the Torus.
    T_N[X] = T[X]/(X^N+1)
    X^N+1 is the M-(2N)th cyclotomic polynomial.
    Components ordering are:
    [0] = 1, [1] = X, [2] = X^2, ..., [N-1] = X^(N-1)
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

    def __mul__(self, rhs) -> TorusRing:
        if type(rhs) == IntRing:
            return TorusRing.ext_product(rhs, self.value)
        else:
            print(type(rhs))
            raise Exception("TorusRing pow type exception")

    def __pow__(self, rhs) -> TorusRing:
        t = self
        if type(rhs) == int:
            for i in range(rhs - 1):
                t = TorusRing.ext_product(t, self.value)
            return t
        else:
            print(type(rhs))
            raise Exception("TorusRing mul type exception")

    def __rshift__(self, rhs) -> TorusRing:
        if type(rhs) == int:
            return TorusRing(np.array([coeff >> rhs for coeff in self.value]))
        else:
            raise Exception("IntRing rshift type exception")

    def __repr__(self) -> str:
        return f"TorusRing({self.value})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.value)

    def __eq__(self, x):
        return np.all(self.value == x.value)

    def toInt(self) -> "integer array":
        return np.array([elem >> (Torus.q - TRLWE.p) for elem in TorusRing.round(self).value])

    @staticmethod
    def getZero():
        return TorusRing(np.array([0 for i in range(TorusRing.N)]))

    @staticmethod
    def getOne():
        return TorusRing(np.array([0 for i in range(TorusRing.N - 1)].insert(0, 1)))

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
    def ext_product(A: TorusRing, B: IntRing):
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
    Nbit = 0
    k = 1
    sigma = 0
    p = 0
    p_ = 0

    @classmethod
    def init(cls, N: int, sigma: float, p: int):
        TorusRing.init(N)
        IntRing.init(N)
        cls.N = N
        cls.Nbit = int(math.log2(cls.N))
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

    def __sub__(self, rhs: TorusRing) -> TRLWE:
        return TRLWE(self.value - rhs.value)

    @staticmethod
    def rand_element(self) -> TRLWE:
        pass

    @staticmethod
    def rand_plaintext() -> TorusRing:
        return TorusRing.rand_element()

    @staticmethod
    def keyGen() -> IntRing:
        return IntRing.rand_element(1)

    @staticmethod
    def enc(mu: TorusRing, s: IntRing, explicit: bool = False):
        if explicit:
            a = TorusRing.getZero()
            e = np.array([0 for i in range(TRLWE.N)])
        else:
            a = TorusRing.rand_element()    # k = 1
            e = np.array([round(random.normalvariate(0, TRLWE.sigma) * 2**Torus.q) for i in range(TRLWE.N)])
        b = TorusRing.ext_product(a, s) + mu + TorusRing(e)
        return TRLWE(np.array(np.append(a, b)))

    @staticmethod
    def dec(c: TRLWE, s: IntRing):
        return TorusRing.round(TRLWE.dec_wo_round(c, s))

    @staticmethod
    def dec_wo_round(c: TRLWE, s: IntRing):
        mu = c.value[1] - TorusRing.ext_product(c.value[0], s)  # k = 1
        return mu

def main():

    N = 16
    S = 2**-25
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
        print("OK\n")
    else:
        print("NG\n")


    ### Test rotate
    from myTRGSW import TRGSW
    TRGSW.init(N, S, P, 64, 3)
    sk = TRLWE.keyGen()
    print(f"sk: {sk}")
    tv = TorusRing([i for i in range(TorusRing.N)])
    tv = TRLWE.enc(tv, sk, True)
    print(f"tv: {tv}")

    X = [0 for i in range(TorusRing.N - 1)]
    X.insert(1, 1)
    X = IntRing(X)
    X = TRGSW.enc(X, sk)
    print(f"X: {X}")

    print(tv*X**32)



if __name__ == '__main__':
    main()