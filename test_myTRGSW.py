# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_myTRGSW.py
# Purpose:
#
# Author:      Junichi Sakamoto
#
# Created:     2022 Mar. 30
# Copyright:   (c) sakamoto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import unittest
import sys
from myTRLWE import Torus, TRLWE
from myTRGSW import TRGSW

LOOP = 10000
N = 32
S = 2**-25
P = 8
B = 64
l = 3


class TestTGSW(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        TRGSW.init(N, S, P, B, l)

    def test_Flatten(self):
        for i in range(LOOP):
            sk = TRLWE.keyGen()
            m = TRLWE.rand_plaintext()
            c = TRLWE.enc(m, sk)
            Ginv = TRGSW.Ginv(c)
            G = TRGSW.getGTmatrix()
            a = Ginv @ G
            mp = TRLWE.dec(TRLWE(a), sk)
            self.assertEqual(mp, m)

        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")

    def test_HomMultiplication(self):
        for i in range(LOOP):
            sk = TRLWE.keyGen()
            m1 = TRLWE.rand_plaintext()
            m2 = TRGSW.rand_plaintext()
            c1 = TRLWE.enc(m1, sk)
            c2 = TRGSW.enc(m2, sk)
            c3 = TRGSW.externalProduct(c1, c2)
            m3 = TRLWE.dec(c3, sk)
            self.assertEqual(m2*m1, m3)

        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")

    def test_CMUX(self):
        for i in range(LOOP):
            sk = TRLWE.keyGen()
            m1 = TRLWE.rand_plaintext()
            m2 = TRLWE.rand_plaintext()
            c1 = TRLWE.enc(m1, sk)
            c2 = TRLWE.enc(m2, sk)
            sel = TRGSW.rand_plaintext_bin()
            c_sel = TRGSW.enc(sel, sk)
            res = TRGSW.CMUX(c_sel, c1, c2)
            selected = TRLWE.dec(res, sk)
            sel = sel.value[0]

            ans = m2 if sel else m1
            self.assertEqual(ans, selected)

        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")


if __name__ == '__main__':
    unittest.main()
