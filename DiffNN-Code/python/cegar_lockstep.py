#!/usr/bin/env python3
# cegar_lockstep.py — build {over,under}‑approximate .nnet files for                      

import argparse, logging, sys, pathlib, itertools
from copy import deepcopy
import numpy as np
import common

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(message)s",
                    stream=sys.stderr)
set_verbose = lambda v: logging.getLogger().setLevel(logging.INFO if v else logging.WARNING)

def to_numpy(net):
    Ws, Bs = [], []
    for l, (W, b) in enumerate(zip(net["weights"], net["biases"])):
        W, b = np.asarray(W, np.float32), np.asarray(b, np.float32)
        logging.info(" layer %d parsed: %d → %d", l, W.shape[1], W.shape[0])
        Ws.append(W); Bs.append(b)
    net["weights"], net["biases"] = Ws, Bs
    return net

# ------------expand nnet to pos/neg/increasing/decreasing columns -> perform abstraction
sign = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))

def downstream_signs(Ws): # Ws is the list of weight matricies
    S = [None] * len(Ws) 
    S[-1] = sign(Ws[-1].sum(0)) # start at the output layer 
    for l in range(len(Ws) - 2, -1, -1):
        S[l] = sign(S[l+1] @ Ws[l])
    return S

def split_pos_neg(W):
    for j in range(W.shape[1]):
        col = W[:, j]
        mask = col >= 0
        if mask.any():            yield "pos", j, np.where(mask,  col, 0.) #using yield... how good is that?
        if (~mask).any():         yield "neg", j, np.where(~mask, col, 0.)

def expand(net):
    sizes, Ws, Bs, buckets = [net["layerSizes"][0]], [], [], []
    incdec = downstream_signs(net["weights"])
    for l in range(net["numLayers"] - 1):
        W, b = net["weights"][l], net["biases"][l]
        cols, bias, lab = [], [], []
        for pn, j, col in split_pos_neg(W):
            lab.append((pn, "inc" if incdec[l][j] > 0 else "dec"))
            cols.append(col[:, None])
            bias.append(0.)
        Ws.append(np.hstack(cols)); Bs.append(np.asarray(bias, np.float32))
        buckets.append(lab); sizes.append(len(bias))
    Ws.append(net["weights"][-1].copy()); Bs.append(net["biases"][-1].copy())
    sizes.append(net["layerSizes"][-1])
    return sizes, Ws, Bs, buckets

def merge_cols(W, b, j, k, op):
    agg = np.maximum if op == "max" else np.minimum
    Wm = np.column_stack([np.delete(W, (j, k), 1), agg(W[:, j], W[:, k])])
    bm = np.append(np.delete(b, (j, k)), b[j] + b[k])
    return Wm, bm

# used for initial abstraction generation
def run(net, x):
    a = np.asarray(x, np.float32)
    for W, b in zip(net["weights"][:-1], net["biases"][:-1]):
        a = np.maximum(0, W.T @ a + b)
    return net["weights"][-1].T @ a + net["biases"][-1]

# used for refinement
def forward_all(net, x):
    outs = [np.asarray(x, np.float32)]
    for W, b in zip(net["weights"][:-1], net["biases"][:-1]):
        outs.append(np.maximum(0, W.T @ outs[-1] + b))
    outs.append(net["weights"][-1].T @ outs[-1] + net["biases"][-1])
    return outs

# -------------------------- LOCK‑STEP ---------------------
def lockstep(netA, netB, mode="over"):
    op = "max" if mode == "over" else "min"
    szA, WA, bA, buckA = expand(netA)
    szB, WB, bB, buckB = expand(netB)
    L = len(WA) - 1
    changed = True
    while changed:
        changed = False
        for l in range(L):
            pair = next(((j, k) for j in range(len(buckA[l])-1)
                                   for k in range(j+1, len(buckA[l]))
                                   if buckA[l][j] == buckA[l][k]), None)
            if not pair: continue
            j, k = pair
            WA[l], bA[l] = merge_cols(WA[l], bA[l], j, k, op)
            WB[l], bB[l] = merge_cols(WB[l], bB[l], j, k, op)
            keep = [c for c in range(len(buckA[l])) if c not in (j, k)]
            buckA[l] = buckB[l] = [buckA[l][c] for c in keep] + [buckA[l][j]]
            szA[l+1] = WA[l].shape[1]; szB[l+1] = WB[l].shape[1]
            changed = True

    def pack(tpl, sz, Ws, Bs):
        out = tpl.copy()
        out.update({"layerSizes": sz,
                    "numLayers": len(sz)-1,
                    "maxLayerSize": max(sz),
                    "weights": [w.tolist() for w in Ws],
                    "biases":  [b.tolist() for b in Bs]})
        return out
    return pack(netA, szA, WA, bA), pack(netB, szB, WB, bB)


# -------------------------------- INDICATOR Guided ---------------------
# ---view secrtion 4.1  Generating an Initial Abstraction of the CEGAR paper

def indicator_abstraction(net, P, Q, X, kind="over", verbose=False):
    net_bar = deepcopy(net)
    if "buckets" not in net_bar:
        _, _, _, net_bar["buckets"] = expand(net_bar)
    if "map" not in net_bar:
        net_bar["map"] = [list(range(len(b))) for b in net_bar["buckets"]]
    rounds = 0
    while all(Q(run(net_bar, x)) for x in X):
        l, j, k = _best_pair(net_bar) or (None, None, None)
        if l is None: break
        _merge_inplace(net_bar, l, j, k, kind)
        rounds += 1
        if verbose:
            print(f"   merge {rounds}: L{l} ({j},{k})")
    if verbose: print("indicator‑guided merges:", rounds)
    return net_bar

# helpers local to indicator_abstraction
def _best_pair(net_bar):
    best = None; best_d = np.inf
    for i, W in enumerate(net_bar["weights"][:-1]):
        groups = {}
        for c, lab in enumerate(net_bar["buckets"][i]):
            groups.setdefault(lab, []).append(c)
        for cols in groups.values():
            for j, k in itertools.combinations(cols, 2):
                d = np.max(np.abs(W[:, j] - W[:, k]))
                if d < best_d: best_d, best = d, (i, j, k)
    return best

def _merge_inplace(net_bar, i, j, k, kind):
    W, b = net_bar["weights"][i], net_bar["biases"][i]
    agg = np.maximum if kind == "over" else np.minimum
    W[:, j], b[j] = agg(W[:, j], W[:, k]), max(b[j], b[k]) if kind == "over" else min(b[j], b[k])
    net_bar["weights"][i] = np.delete(W, k, 1)
    net_bar["biases"][i]  = np.delete(b, k)
    del net_bar["buckets"][i][k]; del net_bar["map"][i][k]
    for l in range(i+1, len(net_bar["weights"])):
        net_bar["weights"][l] = np.delete(net_bar["weights"][l], k, 0)

def refine(net, net_bar, x):
    Ac, Aa = forward_all(net, x), forward_all(net_bar, x)
    best, score = None, 0.
    for l, (Wc, Wa, mp) in enumerate(zip(net["weights"][:-1], net_bar["weights"][:-1], net_bar["map"])):
        for jc, ja in enumerate(mp):
            for pc, pa in enumerate(mp):
                s = abs(Wc[pc, jc] - Wa[pa, ja]) * abs(Ac[l+1][jc] - Aa[l+1][ja])
                if s > score: score, best = s, (l, ja, jc)
    if best: _split(net_bar, *best)

def _split(net_bar, l, ja, jc):
    W, b = net_bar["weights"][l], net_bar["biases"][l]
    net_bar["weights"][l] = np.insert(W, ja, W[:, ja], 1)
    net_bar["biases"][l]  = np.insert(b, ja, b[ja])
    lab = net_bar["buckets"][l][ja]
    net_bar["buckets"][l].insert(ja, lab); net_bar["map"][l].insert(ja, jc)
    for i in range(l+1, len(net_bar["weights"])):
        net_bar["weights"][i] = np.insert(net_bar["weights"][i], ja,
                                          net_bar["weights"][i][ja, :], 0)

def main(orig, quant, out_dir, verbose=False):
    set_verbose(verbose)
    nA, nB = to_numpy(common.read_network(orig)), to_numpy(common.read_network(quant))

    overA,  overB  = lockstep(nA, nB, "over")
    underA, underB = lockstep(nA, nB, "under")

      # TODO: implement indicator-guided abstractoin
    # currently, our lockstep creates an inital abstraction to saturation creating only four neurons per layer that represent pos/inc, pos/dec, neg/inc, neg/dec
    # the bottom was my attempt to use indicator-guided abstraction as shown in section 4.1 of the CEGAR paper but it is not working yet
    
    # P = lambda x: True          # input property  – accept every input
    # Q = lambda y: True          # output property – accept every output
    # min_vals = np.asarray(net_orig["minVals"], dtype=np.float32)
    # max_vals = np.asarray(net_orig["maxVals"], dtype=np.float32)
    # X = [min_vals, max_vals]   
    # over_orig  = indicator_guided_abstraction(net_orig,  P, Q, X, kind="over",  verbose=verbose)
    # over_quant = indicator_guided_abstraction(net_quant, P, Q, X, kind="over",  verbose=verbose)
    # under_orig  = indicator_guided_abstraction(net_orig,  P, Q, X, kind="under", verbose=verbose)
    # under_quant = indicator_guided_abstraction(net_quant, P, Q, X, kind="under", verbose=verbose)

    out = pathlib.Path(out_dir)
    common.write_network(overA,  out / f"over_{pathlib.Path(orig).stem}.nnet")
    common.write_network(underA, out / f"under_{pathlib.Path(orig).stem}.nnet")
    common.write_network(overB,  out / f"over_{pathlib.Path(quant).stem}.nnet")
    common.write_network(underB, out / f"under_{pathlib.Path(quant).stem}.nnet")
    print("✓ written to", out.resolve())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("quantised")
    ap.add_argument("--out", "-o", default="build")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    main(args.original, args.quantised, args.out, args.verbose)
