# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_myTLWE.py
# Purpose:     
#
# Author:      Junichi Sakamoto
#
# Created:     2022 Mar. 24
# Copyright:   (c) sakamoto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import unittest
from myTLWE import Torus
from myTLWE import TLWE

LOOP = 100000
N = 16
S = 2**-15
Q = 6
P = 11

class TestTLWE(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        TLWE.init(N, S, P)

    def test_encryption(self):
        for i in range(LOOP):
            ### Gen test vector
            sk = TLWE.keyGen()
            mu = TLWE.rand_plaintext()
            ### Enc and Dec
            c = TLWE.enc(mu, sk)
            res = TLWE.dec(c, sk)

            self.assertEqual(mu, res)
        print(f"PASS: {LOOP} test_encryption")

    def test_HomAddition(self):
        for i in range(LOOP):
            ### Gen test vector
            sk = TLWE.keyGen()
            mu1 = TLWE.rand_plaintext()
            mu2 = TLWE.rand_plaintext()
            ### Enc and Dec
            c1 = TLWE.enc(mu1, sk)
            c2 = TLWE.enc(mu2, sk)
            c3 = c1 + c2
            res = TLWE.dec(c3, sk)

            self.assertEqual(mu1 + mu2, res)
        print(f"PASS: {LOOP} test_HomAddition")

if __name__ == '__main__':
    unittest.main()