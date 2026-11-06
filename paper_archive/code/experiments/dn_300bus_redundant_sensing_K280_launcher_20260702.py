# -*- coding: utf-8 -*-
"""Launcher for the K=280 redundant-sensing recovery run."""
import dn_300bus_redundant_sensing_attack_20260702 as atk

atk.K_ATTACK = 280
atk.N_TRAIN = 260000
atk.N_VAL = 7000
atk.N_FINAL = 10000
atk.EPOCHS = 16
atk.LR = 4e-5
atk.OUT_NAME = '300bus_redundant_sensing_K280_attack_20260702.txt'
atk.main()
