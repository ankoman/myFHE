# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_myTRLWE.py
# Purpose:
#
# Author:      Junichi Sakamoto
#
# Created:     2022 Mar. 29
# Copyright:   (c) sakamoto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import unittest
import sys
from myTRLWE import TorusRing
from myTRLWE import TRLWE

LOOP = 10000
N = 32
S = 2**-25
P = 1


class TestTRLWE(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        TRLWE.init(N, S, P)

    def test_encryption(self):
        for i in range(LOOP):
            ### Gen test vector
            sk = TRLWE.keyGen()
            mu = TRLWE.rand_plaintext()
            ### Enc and Dec
            c = TRLWE.enc(mu, sk)
            res = TRLWE.dec(c, sk)

            self.assertEqual(mu, res)
        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")


    def test_HomAddition(self):
        for i in range(LOOP):
            ### Gen test vector
            sk = TRLWE.keyGen()
            mu1 = TRLWE.rand_plaintext()
            mu2 = TRLWE.rand_plaintext()
            ### Enc and Dec
            c1 = TRLWE.enc(mu1, sk)
            c2 = TRLWE.enc(mu2, sk)
            c3 = c1 + c2
            res = TRLWE.dec(c3, sk)

            self.assertEqual(mu1 + mu2, res)
        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")



if __name__ == '__main__':
    unittest.main()
