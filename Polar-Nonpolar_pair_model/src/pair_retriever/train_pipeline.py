# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from pair_retriever.graph import (
    GraphCache,
    build_cif_index,
    calc_opcount,
    load_structure_from_index,
    norm_id,
    safe_reduced_formula,
)
from pair_retriever.model import GNNEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df = df.dropna(subset=["Polar_mpid", "NPolar_mpid"]).copy()
    df["Polar_mpid"] = df["Polar_mpid"].astype(str).map(norm_id)
    df["NPolar_mpid"] = df["NPolar_mpid"].astype(str).map(norm_id)
    return df


def build_pos_by_polar(df_pos: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in df_pos.itertuples(index=False):
        out.setdefault(row.Polar_mpid, []).append(row.NPolar_mpid)
    return out


def build_formula_map(mpids: List[str], cif_index: Dict[str, str]) -> Dict[str, Optional[str]]:
    out = {}
    for mid in tqdm(mpids, desc="Build formula map"):
        s = load_structure_from_index(mid, cif_index)
        out[mid] = safe_reduced_formula(s) if s is not None else None
    return out


def build_opcount_map(mpids: List[str], cif_index: Dict[str, str], symprec: float) -> Dict[str, Optional[int]]:
    out = {}
    for mid in tqdm(mpids, desc=f"Build opcount map (symprec={symprec})"):
        s = load_structure_from_index(mid, cif_index)
        out[mid] = calc_opcount(s, symprec) if s is not None else None
    return out


def choose_candidate_pool(df_test: pd.DataFrame, use_hard_testpool: bool, hard_pool_index: Optional[str]) -> List[str]:
    if use_hard_testpool:
        if not hard_pool_index or not os.path.isfile(hard_pool_index):
            raise FileNotFoundError(f"Hard pool index not found: {hard_pool_index}")
        idx = pd.read_csv(hard_pool_index, dtype=str)
        col_mid = None
        for c in idx.columns:
            if c.lower().strip() in ["material_id", "mpid", "mid"]:
                col_mid = c
                break
        if col_mid is None:
            raise ValueError(f"Cannot find material id column in {hard_pool_index}")
        return sorted(set(idx[col_mid].astype(str).map(norm_id).tolist()))

    return sorted(set(df_test["NPolar_mpid"].tolist()))


def info_nce_loss(zp: torch.Tensor, zn: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (zp @ zn.t()) / float(temperature)
    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)


def sym_hinge_loss(op_p: torch.Tensor, op_n: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    delta = op_n - op_p
    return torch.relu(float(margin) - delta).mean()


def train_epoch(
    encoder: nn.Module,
    graph_cache: GraphCache,
    cif_index: Dict[str, str],
    pos_by_polar_train: Dict[str, List[str]],
    train_nonpolar_pool: List[str],
    formula_map_train: Dict[str, Optional[str]],
    op_map_train: Optional[Dict[str, Optional[int]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_polars: int,
    temperature: float,
    neg_per_polar_extra: int,
    use_same_formula_neg: bool,
    use_sym_hinge_loss: bool,
    sym_margin: float,
    lambda_sym_loss: float,
) -> float:
    encoder.train()
    polars = list(pos_by_polar_train.keys())
    random.shuffle(polars)

    total_loss = 0.0
    n_steps = 0

    for st in range(0, len(polars), batch_polars):
        batch_polars_list = polars[st: st + batch_polars]
        if len(batch_polars_list) < 8:
            continue

        batch_pos = [random.choice(pos_by_polar_train[p]) for p in batch_polars_list]

        extra_negs = []
        if neg_per_polar_extra > 0:
            for p in batch_polars_list:
                fml = formula_map_train.get(p) if use_same_formula_neg else None
                if use_same_formula_neg and fml is not None:
                    pool = [
                        n for n in train_nonpolar_pool
                        if (formula_map_train.get(n) == fml) and (n not in pos_by_polar_train.get(p, []))
                    ]
                    if len(pool) < 5:
                        pool = [n for n in train_nonpolar_pool if n not in pos_by_polar_train.get(p, [])]
                else:
                    pool = [n for n in train_nonpolar_pool if n not in pos_by_polar_train.get(p, [])]

                if len(pool) == 0:
                    pool = train_nonpolar_pool

                k = min(neg_per_polar_extra, len(pool))
                extra_negs.append(random.sample(pool, k))
        else:
            extra_negs = [[] for _ in batch_polars_list]

        step_mpids = set(batch_polars_list) | set(batch_pos)
        for negs in extra_negs:
            step_mpids |= set(negs)
        step_mpids = sorted(step_mpids)

        graphs = {}
        for mid in step_mpids:
            g = graph_cache.get(mid, cif_index)
            if g is not None:
                graphs[mid] = g

        kept_polars, kept_pos, kept_negs = [], [], []
        for p, npos, negs in zip(batch_polars_list, batch_pos, extra_negs):
            if (p in graphs) and (npos in graphs):
                kept_polars.append(p)
                kept_pos.append(npos)
                kept_negs.append([n for n in negs if n in graphs])

        if len(kept_polars) < 8:
            continue

        batch_p = next(iter(DataLoader([graphs[p] for p in kept_polars], batch_size=len(kept_polars), shuffle=False))).to(device)
        batch_n = next(iter(DataLoader([graphs[n] for n in kept_pos], batch_size=len(kept_pos), shuffle=False))).to(device)

        zp = encoder(batch_p)
        zn = encoder(batch_n)

        loss = info_nce_loss(zp, zn, temperature)

        if neg_per_polar_extra > 0:
            neg_graphs = []
            neg_owner = []
            for i, negs in enumerate(kept_negs):
                for n in negs:
                    neg_graphs.append(graphs[n])
                    neg_owner.append(i)

            if len(neg_graphs) > 0:
                batch_neg = next(iter(DataLoader(neg_graphs, batch_size=len(neg_graphs), shuffle=False))).to(device)
                zneg = encoder(batch_neg)

                losses = []
                for i in range(zp.size(0)):
                    idx = [j for j, owner in enumerate(neg_owner) if owner == i]
                    if len(idx) == 0:
                        continue
                    zi = zp[i:i+1]
                    posi = zn[i:i+1]
                    negi = zneg[idx]
                    cand = torch.cat([posi, negi], dim=0)
                    logits = (zi @ cand.t()).squeeze(0) / float(temperature)
                    target = torch.tensor([0], device=device)
                    losses.append(F.cross_entropy(logits[None, :], target))

                if len(losses) > 0:
                    loss = loss + 0.5 * torch.stack(losses).mean()

        if use_sym_hinge_loss and (op_map_train is not None):
            op_p = torch.tensor(
                [float(op_map_train.get(p) or 0.0) for p in kept_polars],
                device=device,
                dtype=torch.float32,
            )
            op_n = torch.tensor(
                [float(op_map_train.get(n) or 0.0) for n in kept_pos],
                device=device,
                dtype=torch.float32,
            )
            loss_sym = sym_hinge_loss(op_p, op_n, margin=sym_margin)
            loss = loss + float(lambda_sym_loss) * loss_sym

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        n_steps += 1

    return total_loss / max(1, n_steps)


@torch.no_grad()
def embed_mpids(
    encoder: nn.Module,
    graph_cache: GraphCache,
    cif_index: Dict[str, str],
    mpids: List[str],
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    encoder.eval()
    graphs, keep = [], []

    for mid in mpids:
        g = graph_cache.get(mid, cif_index)
        if g is not None:
            graphs.append(g)
            keep.append(mid)

    emb: Dict[str, np.ndarray] = {}
    for st in range(0, len(graphs), batch_size):
        batch_graphs = graphs[st: st + batch_size]
        mids = keep[st: st + batch_size]
        batch = next(iter(DataLoader(batch_graphs, batch_size=len(batch_graphs), shuffle=False))).to(device)
        z = encoder(batch).detach().cpu().numpy()
        for m, vec in zip(mids, z):
            emb[m] = vec.astype(np.float32)

    return emb


def eval_retrieval(
    test_polars: List[str],
    pos_by_polar_test: Dict[str, List[str]],
    cand_pool: List[str],
    emb: Dict[str, np.ndarray],
    ks: List[int],
    op_map: Optional[Dict[str, Optional[int]]] = None,
    use_sym_bias: bool = False,
    use_sym_score_bias: bool = True,
    lambda_sym_score: float = 0.15,
    ops_norm: float = 192.0,
    clamp_delta: float = 1.0,
) -> dict:
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    n_total = 0
    n_eval = 0

    cand_ok = [c for c in cand_pool if c in emb]
    if len(cand_ok) == 0:
        return {"n_eval": 0, "reason": "no candidates embedded"}

    zc = np.vstack([emb[c] for c in cand_ok]).astype(np.float32)

    for p in tqdm(test_polars, desc="Evaluate"):
        n_total += 1
        if p not in emb:
            continue

        gt = set(pos_by_polar_test.get(p, []))
        if len(gt) == 0:
            continue

        zp = emb[p].astype(np.float32)
        scores = (zc @ zp).astype(np.float32)

        if use_sym_bias and (op_map is not None) and use_sym_score_bias:
            op_p = op_map.get(p)
            if op_p is not None:
                bias = []
                for c in cand_ok:
                    op_n = op_map.get(c)
                    if op_n is None:
                        bias.append(0.0)
                    else:
                        d = (float(op_n) - float(op_p)) / float(ops_norm)
                        d = max(-clamp_delta, min(clamp_delta, d))
                        bias.append(float(lambda_sym_score) * d)
                scores = scores + np.asarray(bias, dtype=np.float32)

        order = np.argsort(-scores)
        ranked = [cand_ok[i] for i in order]

        if not any(n in gt for n in ranked):
            continue

        n_eval += 1
        for k in ks:
            if any(n in gt for n in ranked[:k]):
                hits[k] += 1

        rr = 0.0
        for i, n in enumerate(ranked, start=1):
            if n in gt:
                rr = 1.0 / float(i)
                break
        rr_sum += rr

    out = {
        "tag": "closed_world_testpool",
        "n_total_queries_considered": int(n_total),
        "n_eval_queries_with_GT_in_pool": int(n_eval),
        "coverage_GT_in_pool": float(n_eval) / float(n_total + 1e-12),
        "MRR": float(rr_sum) / float(n_eval + 1e-12),
    }
    for k in ks:
        out[f"Hit@{k}"] = float(hits[k]) / float(n_eval + 1e-12)
    return out


def run_training(cfg: dict) -> None:
    set_seed(int(cfg["train"]["seed"]))

    run_dir = Path(cfg["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_dir = run_dir / "graph_cache"

    train_pos_csv = cfg["paths"]["train_pos_csv"]
    test_pos_csv = cfg["paths"]["test_pos_csv"]
    struct_dir = cfg["paths"]["struct_dir"]

    if not os.path.isfile(train_pos_csv):
        raise FileNotFoundError(f"train_pos_csv not found: {train_pos_csv}")
    if not os.path.isfile(test_pos_csv):
        raise FileNotFoundError(f"test_pos_csv not found: {test_pos_csv}")

    df_tr = load_split_pairs(train_pos_csv)
    df_te = load_split_pairs(test_pos_csv)

    print(f"[INFO] train_pos_pairs={len(df_tr)} | test_pos_pairs={len(df_te)}")
    print(f"[INFO] train_polars={df_tr['Polar_mpid'].nunique()} | test_polars={df_te['Polar_mpid'].nunique()}")

    cif_index = build_cif_index(struct_dir)
    print(f"[INFO] cif_index_size={len(cif_index)}")

    graph_cache = GraphCache(
        cache_dir=str(graph_cache_dir),
        cutoff=float(cfg["graph"]["cutoff"]),
        max_neighbors=int(cfg["graph"]["max_neighbors"]),
    )

    pos_by_polar_train = build_pos_by_polar(df_tr)
    pos_by_polar_test = build_pos_by_polar(df_te)

    train_polars = sorted(pos_by_polar_train.keys())
    test_polars = sorted(pos_by_polar_test.keys())
    train_nonpolars = sorted(set(df_tr["NPolar_mpid"].tolist()))

    cand_pool = choose_candidate_pool(
        df_test=df_te,
        use_hard_testpool=bool(cfg["eval"]["use_hard_testpool"]),
        hard_pool_index=cfg["paths"].get("hard_pool_index"),
    )
    print(f"[INFO] eval candidate_pool size={len(cand_pool)} | USE_HARD_TESTPOOL={cfg['eval']['use_hard_testpool']}")

    mpids_formula = sorted(set(train_polars) | set(train_nonpolars))
    formula_map_train = build_formula_map(mpids_formula, cif_index)

    op_map_train = None
    op_map_eval = None
    if bool(cfg["train"]["use_sym_hinge_loss"]):
        op_map_train = build_opcount_map(mpids_formula, cif_index, float(cfg["train"]["symprec_opcount"]))
    if bool(cfg["eval"]["use_sym_score_bias"]):
        mpids_eval = sorted(set(test_polars) | set(cand_pool))
        op_map_eval = build_opcount_map(mpids_eval, cif_index, float(cfg["train"]["symprec_opcount"]))

    device_name = cfg["train"]["device"]
    if str(device_name).lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    encoder = GNNEncoder(
        hidden=int(cfg["model"]["hidden"]),
        layers=int(cfg["model"]["layers"]),
        out_dim=int(cfg["model"]["emb_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        z_max=int(cfg["model"]["z_max"]),
        cutoff=float(cfg["graph"]["cutoff"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    print(f"[INFO] DEVICE={device} | epochs={cfg['train']['epochs']} | batch_polars={cfg['train']['batch_polars']}")

    for ep in range(1, int(cfg["train"]["epochs"]) + 1):
        loss = train_epoch(
            encoder=encoder,
            graph_cache=graph_cache,
            cif_index=cif_index,
            pos_by_polar_train=pos_by_polar_train,
            train_nonpolar_pool=train_nonpolars,
            formula_map_train=formula_map_train,
            op_map_train=op_map_train,
            optimizer=optimizer,
            device=device,
            batch_polars=int(cfg["train"]["batch_polars"]),
            temperature=float(cfg["train"]["temperature"]),
            neg_per_polar_extra=int(cfg["train"]["neg_per_polar_extra"]),
            use_same_formula_neg=bool(cfg["train"]["use_same_formula_neg"]),
            use_sym_hinge_loss=bool(cfg["train"]["use_sym_hinge_loss"]),
            sym_margin=float(cfg["train"]["sym_margin"]),
            lambda_sym_loss=float(cfg["train"]["lambda_sym_loss"]),
        )
        if (ep == 1) or (ep % 5 == 0) or (ep == int(cfg["train"]["epochs"])):
            print(f"[TRAIN] epoch={ep:03d}/{cfg['train']['epochs']}  loss={loss:.4f}")

    enc_path = run_dir / "encoder.pt"
    torch.save(encoder.state_dict(), enc_path)
    print(f"[OK] Saved encoder: {enc_path}")

    emb_mpids = sorted(set(test_polars) | set(cand_pool))
    emb = embed_mpids(
        encoder=encoder,
        graph_cache=graph_cache,
        cif_index=cif_index,
        mpids=emb_mpids,
        batch_size=int(cfg["eval"]["embed_batch_size"]),
        device=device,
    )

    ks = list(cfg["eval"]["ks"])

    m_base = eval_retrieval(
        test_polars=test_polars,
        pos_by_polar_test=pos_by_polar_test,
        cand_pool=cand_pool,
        emb=emb,
        ks=ks,
        op_map=None,
        use_sym_bias=False,
    )
    with open(run_dir / "metrics_base.json", "w", encoding="utf-8") as f:
        json.dump(m_base, f, indent=2, ensure_ascii=False)

    print("\n========== BASE ==========")
    print(m_base)

    if bool(cfg["eval"]["use_sym_score_bias"]) and (op_map_eval is not None):
        m_sym = eval_retrieval(
            test_polars=test_polars,
            pos_by_polar_test=pos_by_polar_test,
            cand_pool=cand_pool,
            emb=emb,
            ks=ks,
            op_map=op_map_eval,
            use_sym_bias=True,
            use_sym_score_bias=bool(cfg["eval"]["use_sym_score_bias"]),
            lambda_sym_score=float(cfg["eval"]["lambda_sym_score"]),
            ops_norm=float(cfg["train"]["ops_norm"]),
            clamp_delta=float(cfg["eval"]["clamp_delta"]),
        )
        with open(run_dir / "metrics_symbias.json", "w", encoding="utf-8") as f:
            json.dump(m_sym, f, indent=2, ensure_ascii=False)

        print("\n====== SYM SCORE BIAS ======")
        print(m_sym)

    with open(run_dir / "used_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("\n============================================================")
    print("[DONE] Outputs in:", run_dir.resolve())
    print("============================================================")