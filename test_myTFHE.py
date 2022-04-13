# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test_myTFHE.py
# Purpose:
#
# Author:      Junichi Sakamoto
#
# Created:     2022 Apr. 11
# Copyright:   (c) sakamoto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from __future__ import annotations
from myTRGSW import TRGSW
from myTRLWE import TorusRing, TRLWE, IntRing
from myTLWE import Torus, TLWE
from myTFHE import TFHE
import random
import numpy as np
import time
import unittest
import sys

LOOP = 10000
N = 8
n = 4
S = 2**-25
s = 2**-15
P = 2
B = 64
l = 3
base = 4
t = 8


class TestTGSW(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        TFHE.init(n, s, N, S, P, B, l, base, t)

    def test_sampleExtractIndex0(self):
        for i in range(LOOP):

            sk = TRGSW.keyGen()
            mu = TRLWE.rand_plaintext()
            c = TRLWE.enc(mu, sk)
            extracted = TFHE.SampleExtractIndex0(c)
            constant_term = TLWE.dec(extracted, sk.value)
            self.assertEqual(constant_term, Torus(mu.value[0]))

        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")

    def test_keySwitch(self):
        for i in range(LOOP):
            sk = TLWE.keyGen()
            skr = TRLWE.keyGen()
            mu = TRLWE.rand_plaintext()
            c = TRLWE.enc(mu, skr)
            extracted = TFHE.SampleExtractIndex0(c)
            ksk = TFHE.genKeySwitchingKey(sk, skr.value)
            res = TFHE.keySwitch_naive(extracted, ksk)
            res = TLWE.dec(res, sk)
            self.assertEqual(res, Torus(mu.value[0]))

        print(f"PASS: {LOOP} {sys._getframe().f_code.co_name}")

if __name__ == '__main__':
    start = time.time()
    unittest.main()
    elapsed_time = time.time() - start
    print("\nelapsed_time:{0}".format(elapsed_time) + "[sec]")
