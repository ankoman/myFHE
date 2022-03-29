# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_myTGSW.py
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
from myTLWE import Torus
from myTLWE import TLWE
from myTGSW import TGSW

LOOP = 10000
N = 32
S = 2**-15
P = 6
B = 64
l = 3



class TestTGSW(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        TGSW.init(N, S, P, B, l)

    def test_Flatten(self):
        for i in range(LOOP):
            ### Gen test vector
            sk = TLWE.keyGen()
            m = TLWE.rand_plaintext()
            c = TLWE.enc(m, sk)
            ### Flatten
            Ginv = TGSW.Ginv(c.value)
            G = TGSW.getGTmatrix()
            flatten = Ginv @ G
            res = TLWE.dec(TLWE(flatten), sk)
            self.assertEqual(m, res)

        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")

    def test_HomMultiplication(self):
        for i in range(LOOP):
            ### Gen test vector
            sk = TLWE.keyGen()
            m1 = TLWE.rand_plaintext()
            m2 = TGSW.rand_plaintext()
            ### Enc and Dec
            c1 = TLWE.enc(m1, sk)
            c2 = TGSW.enc(m2, sk)
            c3 = TGSW.externalProduct(c1, c2)
            m3 = TLWE.dec(c3, sk)
            self.assertEqual(Torus(m1.value*m2), m3)

        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")

    def test_CMUX(self):
        sk = TLWE.keyGen()
        m1 = TLWE.rand_plaintext()
        m2 = TLWE.rand_plaintext()
        c1 = TLWE.enc(m1, sk)
        c2 = TLWE.enc(m2, sk)
        sel = TGSW.rand_plaintext() % 2
        c_sel = TGSW.enc(sel, sk)

        res = TGSW.CMUX(c_sel, c1, c2)
        selected = TLWE.dec(res, sk)

        ans = sel*(m2.value - m1.value) + m1.value
        self.assertEqual(ans, selected.value)

        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")

if __name__ == '__main__':
    unittest.main()
