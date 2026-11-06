import os
import numpy as np
import torch
import dn_300bus_redundant_sensing_attack_20260702 as m

m.K_ATTACK = 299
m.OUT_NAME = "300bus_redundant_sensing_K299_warmstart_attack_20260702.txt"
m.N_TRAIN = 160000
m.N_VAL = 5000
m.N_FINAL = 8000
m.EPOCHS = 8
m.LR = 3e-5


def load_k280_model(seed, lib):
    n_topos, _, n_bus = lib["V"].shape
    model = m.base.NRE300(n_topos, n_bus).to(m.DEVICE)
    ckpt = os.path.join(m.ROOT, f"nre_300bus_ipc_K280_miss30_redundant_seed{seed}_20260702.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(m.ROOT, f"nre_300bus_ipc_seed{seed}.pt")
    obj = torch.load(ckpt, map_location=m.DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return model


def exact_curve_k299(lib):
    rows = []
    for k, seed in [(150, 150404), (180, 180404), (220, 220404), (240, 240404), (260, 260404), (280, 280404), (299, 299404)]:
        _, y, y_map, _ = m.make_dataset(lib, 6000, seed, k, return_ll=True)
        rows.append((k, int(k - int(k * m.MISS_RATE)), float(np.mean(y_map == y))))
    return rows

m.load_old_model = load_k280_model
m.exact_curve = exact_curve_k299
m.main()
