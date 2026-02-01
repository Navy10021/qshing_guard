from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import extract_urls, normalize_url, registrable_domain

# tqdm optional
try:
    from tqdm.auto import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


def _read_csv(path: str) -> pd.DataFrame:
    """Robust CSV reader with encoding fallback."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, encoding="cp949")


def _progress_map(series: pd.Series, fn, desc: str) -> pd.Series:
    """Map with tqdm progress bar if available."""
    if not _HAS_TQDM:
        return series.apply(fn)

    values = series.tolist()
    out = []
    for v in tqdm(values, desc=desc, unit="row"):
        out.append(fn(v))
    return pd.Series(out, index=series.index)


def _detect_kisa_url_col(df: pd.DataFrame) -> str:
    # Expected columns often include localized headers (e.g., date, homepage).
    for c in df.columns:
        cl = str(c).lower()
        if "홈페이지" in str(c) or cl in ("url", "homepage", "site") or "url" in cl:
            return c
    if len(df.columns) >= 1:
        return df.columns[-1]
    raise ValueError(f"Cannot find URL column in KISA CSV. columns={list(df.columns)}")


def _build_from_kisa(kisa_csv: str) -> pd.DataFrame:
    df = _read_csv(kisa_csv)
    url_col = _detect_kisa_url_col(df)

    out = pd.DataFrame({
        "source": "kisa",
        "label": 1,
        "text": None,
        "url_raw": df[url_col].astype(str),
    })
    out["url_norm"] = _progress_map(out["url_raw"], normalize_url, "Normalize URLs (kisa)")
    out = out.dropna(subset=["url_norm"])
    return out


def _build_from_kakao(kakao_csv: str) -> pd.DataFrame:
    df = _read_csv(kakao_csv)
    if "content" not in df.columns:
        raise ValueError(f"Cannot find 'content' column in kakao csv. columns={list(df.columns)}")
    if "class" not in df.columns:
        raise ValueError(f"Cannot find 'class' column in kakao csv. columns={list(df.columns)}")

    tmp = df[["content", "class"]].copy()
    tmp["label"] = tmp["class"].apply(lambda x: 1 if int(x) == 2 else 0)

    # URL extraction (progress)
    if _HAS_TQDM:
        urls_list = []
        for txt in tqdm(tmp["content"].tolist(), desc="Extract URLs (kakao)", unit="row"):
            urls_list.append(extract_urls(txt))
        tmp["urls"] = urls_list
    else:
        tmp["urls"] = tmp["content"].apply(extract_urls)

    tmp = tmp.explode("urls")
    tmp = tmp.rename(columns={"content": "text", "urls": "url_raw"})
    tmp = tmp.dropna(subset=["url_raw"])

    out = pd.DataFrame({
        "source": "kakao",
        "label": tmp["label"].astype(int),
        "text": tmp["text"].astype(str),
        "url_raw": tmp["url_raw"].astype(str),
    })
    out["url_norm"] = _progress_map(out["url_raw"], normalize_url, "Normalize URLs (kakao)")
    out = out.dropna(subset=["url_norm"])
    return out


def _read_normal_csv(normal_csv: str) -> pd.DataFrame:
    """
    Support:
    1) header with 'url' or 'domain' columns
    2) single-column list (first col as url/domain)
    3) headerless 2-col: [rank, domain]
    """
    df = _read_csv(normal_csv)

    if len(df.columns) == 2 and (str(df.columns[0]).startswith("Unnamed") or str(df.columns[0]).isdigit()):
        df = pd.read_csv(normal_csv, header=None)

    cols_lower = [str(c).lower() for c in df.columns]
    if len(df.columns) == 2 and ("url" not in cols_lower) and ("domain" not in cols_lower):
        df = df.rename(columns={df.columns[0]: "rank", df.columns[1]: "domain"})
        return df

    return df


def _pick_normal_url_series(df: pd.DataFrame) -> pd.Series:
    cols = [str(c).lower() for c in df.columns]

    for key in ("url", "urls", "homepage"):
        if key in cols:
            return df[df.columns[cols.index(key)]].astype(str)

    if "domain" in cols:
        dom = df[df.columns[cols.index("domain")]].astype(str)
        return dom.apply(lambda d: f"https://{d.strip()}" if isinstance(d, str) else "")

    if "domain" in df.columns:
        dom = df["domain"].astype(str)
        return dom.apply(lambda d: f"https://{d.strip()}" if isinstance(d, str) else "")

    s = df[df.columns[0]].astype(str)

    sample = s.dropna().astype(str).head(50).tolist()
    domainish = 0
    for v in sample:
        vv = v.strip().lower()
        if vv and "://" not in vv and "." in vv and " " not in vv:
            domainish += 1
    if sample and domainish / max(1, len(sample)) >= 0.7:
        return s.apply(lambda d: f"https://{d.strip()}" if isinstance(d, str) else "")

    return s


def _build_from_normal(normal_csv: str, limit: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    df = _read_normal_csv(normal_csv)
    url_series = _pick_normal_url_series(df)

    out = pd.DataFrame({
        "source": "normal",
        "label": 0,
        "text": None,
        "url_raw": url_series.astype(str),
    })

    out = out.dropna(subset=["url_raw"])
    if limit is not None and limit > 0 and len(out) > limit:
        out = out.sample(n=limit, random_state=seed).reset_index(drop=True)

    out["url_norm"] = _progress_map(out["url_raw"], normalize_url, "Normalize URLs (normal)")
    out = out.dropna(subset=["url_norm"])
    return out


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """URL-level split stratified by label."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    if val_size <= 0:
        val_df = train_df.iloc[:0].copy()
        return train_df, val_df, test_df

    val_frac = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_frac,
        random_state=seed,
        stratify=train_df["label"],
    )
    return train_df, val_df, test_df


def _group_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Group-aware split:
      - Same group never appears across splits
      - Tries to keep label distribution close (greedy assignment on groups)
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["_group"] = df[group_col].fillna("UNKNOWN_GROUP").astype(str)

    gstats = df.groupby("_group")["label"].agg(["count", "sum"]).reset_index()
    gstats = gstats.rename(columns={"sum": "pos"})
    gstats["neg"] = gstats["count"] - gstats["pos"]

    # deterministic shuffle then size-desc
    gstats = gstats.sample(frac=1.0, random_state=seed).sort_values("count", ascending=False).reset_index(drop=True)

    total = int(gstats["count"].sum())
    target_test = int(round(total * test_size))
    target_val = int(round(total * val_size))
    target_train = total - target_test - target_val

    total_pos = int(gstats["pos"].sum())
    pos_ratio = total_pos / max(1, total)

    tgt = {
        "train": {"n": target_train, "pos": int(round(target_train * pos_ratio))},
        "val": {"n": target_val, "pos": int(round(target_val * pos_ratio))},
        "test": {"n": target_test, "pos": int(round(target_test * pos_ratio))},
    }
    cur = {
        "train": {"n": 0, "pos": 0, "groups": []},
        "val": {"n": 0, "pos": 0, "groups": []},
        "test": {"n": 0, "pos": 0, "groups": []},
    }

    def score(split_name: str, add_n: int, add_pos: int) -> float:
        n2 = cur[split_name]["n"] + add_n
        p2 = cur[split_name]["pos"] + add_pos
        return (n2 - tgt[split_name]["n"]) ** 2 + (p2 - tgt[split_name]["pos"]) ** 2

    for _, r in gstats.iterrows():
        g = r["_group"]
        add_n = int(r["count"])
        add_pos = int(r["pos"])

        scores = {k: score(k, add_n, add_pos) for k in ("train", "val", "test")}
        best = min(scores.values())
        cand = [k for k, v in scores.items() if v == best]
        pick = cand[int(rng.integers(0, len(cand)))]
        cur[pick]["n"] += add_n
        cur[pick]["pos"] += add_pos
        cur[pick]["groups"].append(g)

    train_groups = set(cur["train"]["groups"])
    val_groups = set(cur["val"]["groups"])
    test_groups = set(cur["test"]["groups"])

    train_df = df[df["_group"].isin(train_groups)].drop(columns=["_group"])
    val_df = df[df["_group"].isin(val_groups)].drop(columns=["_group"])
    test_df = df[df["_group"].isin(test_groups)].drop(columns=["_group"])

    return train_df, val_df, test_df


def _audit_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    def _inter(a: pd.Series, b: pd.Series) -> int:
        return int(len(set(a.dropna().astype(str)) & set(b.dropna().astype(str))))

    if "url_norm" in train_df.columns and "url_norm" in val_df.columns and "url_norm" in test_df.columns:
        uv = _inter(train_df["url_norm"], val_df["url_norm"])
        ut = _inter(train_df["url_norm"], test_df["url_norm"])
        vt = _inter(val_df["url_norm"], test_df["url_norm"])
        print(f"[AUDIT] url_norm overlap train-val={uv}, train-test={ut}, val-test={vt}")

    if "domain_reg" in train_df.columns and "domain_reg" in val_df.columns and "domain_reg" in test_df.columns:
        dv = _inter(train_df["domain_reg"], val_df["domain_reg"])
        dt = _inter(train_df["domain_reg"], test_df["domain_reg"])
        vt = _inter(val_df["domain_reg"], test_df["domain_reg"])
        print(f"[AUDIT] domain_reg overlap train-val={dv}, train-test={dt}, val-test={vt}")


def _apply_sampling_controls(df: pd.DataFrame, seed: int,
                            phish_limit: int,
                            balance_ratio: float,
                            total_limit: int) -> pd.DataFrame:
    """
    Apply sampling controls in order:
      1) phish_limit (cap label=1)
      2) balance_ratio (benign/phish)
      3) total_limit (cap total with stratified sampling)
    """
    out = df

    if phish_limit and phish_limit > 0:
        ph = out[out.label == 1]
        be = out[out.label == 0]
        if len(ph) > phish_limit:
            ph = ph.sample(n=phish_limit, random_state=seed)
        out = pd.concat([ph, be], ignore_index=True)
        print(f"[STEP] phish_limit={phish_limit} -> phish={int((out.label==1).sum())}, benign={int((out.label==0).sum())}")

    if balance_ratio and balance_ratio > 0:
        ph = out[out.label == 1]
        be = out[out.label == 0]
        target_benign = int(len(ph) * balance_ratio)
        if target_benign <= 0:
            target_benign = 1
        if len(be) > target_benign:
            be = be.sample(n=target_benign, random_state=seed)
        out = pd.concat([ph, be], ignore_index=True)
        print(f"[STEP] balance_ratio={balance_ratio} -> phish={len(ph)}, benign={len(be)}")

    if total_limit and total_limit > 0 and len(out) > total_limit:
        total = len(out)

        def _n_for_group(g: pd.DataFrame) -> int:
            n = int(round(len(g) / total * total_limit))
            return max(1, n)

        out2 = (
            out.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=_n_for_group(g), random_state=seed))
            .reset_index(drop=True)
        )

        if len(out2) > total_limit:
            out2 = out2.sample(n=total_limit, random_state=seed).reset_index(drop=True)

        out = out2
        print(f"[STEP] total_limit={total_limit} -> total={len(out)}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kisa_csv", type=str, required=True)
    ap.add_argument("--kakao_csv", type=str, required=True)
    ap.add_argument("--normal_csv", type=str, default=None)

    ap.add_argument("--normal_limit", type=int, default=0,
                    help="If >0, sample at most N normal URLs (before merge)")
    ap.add_argument("--phish_limit", type=int, default=0,
                    help="If >0, sample at most N phishing URLs (after merge & dedup)")
    ap.add_argument("--total_limit", type=int, default=0,
                    help="If >0, cap total rows after merge (stratified by label)")
    ap.add_argument("--balance_ratio", type=float, default=0.0,
                    help="If >0, enforce benign/phish ratio after sampling. "
                         "Example 1.0 -> 1:1, 0.5 -> benign is half of phish.")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dedup_by_url_norm", action="store_true",
                    help="Deduplicate rows by url_norm (keep first)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # ✅ leakage-safe split option
    ap.add_argument("--split_by", type=str, default="domain", choices=["url", "domain"],
                    help="Split granularity. domain = group split by registrable domain(eTLD+1).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP] build kisa...")
    kisa_df = _build_from_kisa(args.kisa_csv)
    print(f"  kisa: {len(kisa_df)}")

    print("[STEP] build kakao...")
    kakao_df = _build_from_kakao(args.kakao_csv)
    print(f"  kakao: {len(kakao_df)}")

    dfs = [kisa_df, kakao_df]

    if args.normal_csv:
        print("[STEP] build normal...")
        lim = args.normal_limit if args.normal_limit and args.normal_limit > 0 else None
        normal_df = _build_from_normal(args.normal_csv, limit=lim, seed=args.seed)
        print(f"  normal: {len(normal_df)} (limit={args.normal_limit})")
        dfs.append(normal_df)

    print("[STEP] merge...")
    merged = pd.concat(dfs, ignore_index=True)

    merged = merged.dropna(subset=["url_norm"])
    merged["label"] = merged["label"].astype(int)

    if args.dedup_by_url_norm:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["url_norm"], keep="first")
        print(f"[STEP] dedup by url_norm: {before} -> {len(merged)}")

    merged = _apply_sampling_controls(
        merged,
        seed=args.seed,
        phish_limit=args.phish_limit,
        balance_ratio=args.balance_ratio,
        total_limit=args.total_limit,
    )

    # domain grouping key for leakage-safe splitting
    merged["domain_reg"] = _progress_map(merged["url_norm"], registrable_domain, "Compute eTLD+1 (domain_reg)")
    merged["domain_reg"] = merged["domain_reg"].fillna("UNKNOWN_DOMAIN")

    merged["meta"] = [json.dumps({"source": s}, ensure_ascii=False) for s in merged["source"].tolist()]

    print("[STEP] split...")
    if args.split_by == "domain":
        train_df, val_df, test_df = _group_stratified_split(
            merged, group_col="domain_reg", test_size=args.test_size, val_size=args.val_size, seed=args.seed
        )
    else:
        train_df, val_df, test_df = stratified_split(
            merged, test_size=args.test_size, val_size=args.val_size, seed=args.seed
        )

    _audit_leakage(train_df, val_df, test_df)

    merged.to_csv(out_dir / "manifest.csv", index=False)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # optional parquet (keep existing behavior)
    try:
        merged.to_parquet(out_dir / "manifest.parquet", index=False)
        train_df.to_parquet(out_dir / "train.parquet", index=False)
        val_df.to_parquet(out_dir / "val.parquet", index=False)
        test_df.to_parquet(out_dir / "test.parquet", index=False)
    except Exception:
        pass

    print("[OK] manifest written to:", out_dir)
    print(
        "  total:", len(merged),
        "label1(phish):", int((merged.label == 1).sum()),
        "label0(benign):", int((merged.label == 0).sum()),
    )
    print("  train/val/test:", len(train_df), len(val_df), len(test_df))


if __name__ == "__main__":
    main()
