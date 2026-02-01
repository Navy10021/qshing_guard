from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..models.context_gan import ContextAttacker
from ..models.context_encoder import ContextEncoder
from ..models.detector import DetectorConfig, QuishingDetector
from ..train.datasets import QRImageDataset, FusionDataset, QRDatasetConfig
from ..features.url_vectorizer import fit_url_vectorizer, transform_url_vectorizer
from ..utils.backgrounds import BackgroundSampler
from ..utils.context_ops import context_attack
from ..utils.qr_decode import decode_qr_tensor, normalize_url_for_match


@dataclass
class DifficultySchedule:
    def strength(self, r: int) -> float:
        if r < 5:
            return 0.35
        if r < 10:
            return 0.70
        return 1.00


class ReplayBuffer:
    def __init__(self, max_items: int = 4096):
        self.max_items = max_items
        self.img: List[torch.Tensor] = []
        self.y: List[torch.Tensor] = []
        self.url: List[Optional[torch.Tensor]] = []
        self.lex: List[Optional[torch.Tensor]] = []
        self.ctx: List[Optional[torch.Tensor]] = []

    def add(
        self,
        img: torch.Tensor,
        y: torch.Tensor,
        url_vec: Optional[torch.Tensor],
        lex: Optional[torch.Tensor],
        ctx: Optional[torch.Tensor],
    ):
        img = img.detach().cpu()
        y = y.detach().cpu()
        url_vec = url_vec.detach().cpu() if url_vec is not None else None
        lex = lex.detach().cpu() if lex is not None else None
        ctx = ctx.detach().cpu() if ctx is not None else None

        B = img.shape[0]
        for i in range(B):
            self.img.append(img[i : i + 1])
            self.y.append(y[i : i + 1])
            self.url.append(url_vec[i : i + 1] if url_vec is not None else None)
            self.lex.append(lex[i : i + 1] if lex is not None else None)
            self.ctx.append(ctx[i : i + 1] if ctx is not None else None)

        if len(self.img) > self.max_items:
            k = len(self.img) - self.max_items
            self.img = self.img[k:]
            self.y = self.y[k:]
            self.url = self.url[k:]
            self.lex = self.lex[k:]
            self.ctx = self.ctx[k:]

    def sample(self, b: int, device: torch.device):
        if len(self.img) == 0:
            return None
        idx = np.random.randint(0, len(self.img), size=b)
        x = torch.cat([self.img[i] for i in idx], dim=0).to(device)
        y = torch.cat([self.y[i] for i in idx], dim=0).view(-1).long().to(device)

        url0 = self.url[0]
        if url0 is None:
            ctx = torch.cat([self.ctx[i] for i in idx], dim=0).to(device) if self.ctx[0] is not None else None
            return x, y, None, None, ctx

        url = torch.cat([self.url[i] for i in idx], dim=0).to(device)
        lex = torch.cat([self.lex[i] for i in idx], dim=0).to(device)
        ctx = torch.cat([self.ctx[i] for i in idx], dim=0).to(device) if self.ctx[0] is not None else None
        return x, y, url, lex, ctx


def _freeze(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def _unfreeze(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = True


def _make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sample_w = weights[labels]
    return WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)


def _ctx_from_batch(ctx_batch, x_batch, device: torch.device):
    if ctx_batch is None:
        return torch.zeros((x_batch.size(0), 7), device=device, dtype=torch.float32)
    return ctx_batch.to(device)


def _decode_mask(
    x_adv: torch.Tensor,
    payloads: Optional[list[str]],
    image_size: int,
    subset: int = 0,
    require_payload_match: bool = False,
) -> torch.Tensor:
    """Return boolean mask where True means 'decodable' (and payload-matched if enabled).

    x_adv: (B,3,H,W) RGB in [0,1].
    payloads: list[str] length B when require_payload_match=True, else can be None.
    subset=0 -> check all; subset>0 -> check only first subset, assume rest True.
    """
    B = x_adv.size(0)
    mask = torch.ones((B,), dtype=torch.bool, device=x_adv.device)
    n_check = B if subset <= 0 else min(B, subset)

    for i in range(n_check):
        decoded = decode_qr_tensor(x_adv[i], size=image_size)
        if decoded is None or len(decoded) == 0:
            mask[i] = False
            continue
        if require_payload_match:
            gt = payloads[i] if payloads is not None else ""
            if normalize_url_for_match(decoded) != normalize_url_for_match(gt):
                mask[i] = False
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--detector_mode", choices=["qr", "fusion"], default="fusion")
    ap.add_argument("--fusion_mode", choices=["gated", "concat"], default="gated")
    ap.add_argument("--use_context", action="store_true")
    ap.add_argument("--background_dir", default="assets/backgrounds")

    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--k_attack", type=int, default=1)
    ap.add_argument("--k_defense", type=int, default=1)
    ap.add_argument("--attacker_pool", type=int, default=3)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--lr_attacker", type=float, default=2e-4)
    ap.add_argument("--lr_detector", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--balance_sampler", action="store_true")
    ap.add_argument("--max_features", type=int, default=8000)

    ap.add_argument("--lambda_preserve", type=float, default=0.20)
    ap.add_argument("--lambda_adv", type=float, default=0.80)
    ap.add_argument("--lambda_cons", type=float, default=0.20)
    ap.add_argument("--lambda_replay", type=float, default=0.20)

    # decode constraints (non-differentiable filter)
    ap.add_argument("--decode_filter", action="store_true")
    ap.add_argument("--decode_subset", type=int, default=0, help="0=check all; >0=check first N per batch")
    ap.add_argument("--decode_min_keep", type=int, default=8, help="minimum decodable samples to keep a batch step")
    ap.add_argument("--decode_resample", type=int, default=1, help="resample attempts if too few decodable")

    # NEW: require decoded payload == GT payload (normalized)
    ap.add_argument("--payload_match", action="store_true")
    ap.add_argument("--payload_col", type=str, default="url_norm", help="CSV column to use as GT payload string")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    need_payloads = bool(args.decode_filter and args.payload_match)
    cfg_ds = QRDatasetConfig(
        image_size=args.image_size,
        augment=False,
        use_context=bool(args.use_context),
        return_payload=need_payloads,
        payload_col=args.payload_col,
    )

    if args.detector_mode == "qr":
        train_ds = QRImageDataset(args.train_csv, cfg=cfg_ds, seed=args.seed, require_label=True)
        val_ds = QRImageDataset(args.val_csv, cfg=cfg_ds, seed=args.seed, require_label=True)
        det_cfg = DetectorConfig(mode="qr", use_context=bool(args.use_context))
        train_labels = train_ds.labels
        fusion = False
    else:
        df_tr = pd.read_csv(args.train_csv)
        df_va = pd.read_csv(args.val_csv)
        bundle_tr = fit_url_vectorizer(df_tr["url_norm"].astype(str).tolist(), max_features=args.max_features)
        X_tr = bundle_tr.X
        X_va = transform_url_vectorizer(bundle_tr.vectorizer, df_va["url_norm"].astype(str).tolist())
        train_ds = FusionDataset(args.train_csv, url_vec=X_tr, cfg=cfg_ds, seed=args.seed, require_label=True)
        val_ds = FusionDataset(args.val_csv, url_vec=X_va, cfg=cfg_ds, seed=args.seed, require_label=True)
        det_cfg = DetectorConfig(
            mode="fusion",
            fusion_mode=args.fusion_mode,
            url_dim=X_tr.shape[1],
            lex_dim=train_ds.lex.shape[1],
            use_context=bool(args.use_context),
        )
        train_labels = train_ds.labels
        fusion = True

    if args.balance_sampler:
        sampler = _make_sampler(train_labels)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    detector = QuishingDetector(det_cfg).to(device)
    attackers = [ContextAttacker(style_dim=16).to(device) for _ in range(args.attacker_pool)]
    ctx_enc = ContextEncoder().to(device)
    bg = BackgroundSampler(args.background_dir)

    opt_det = torch.optim.AdamW(detector.parameters(), lr=args.lr_detector, weight_decay=1e-4)
    opt_atk = [torch.optim.AdamW(a.parameters(), lr=args.lr_attacker, weight_decay=1e-4) for a in attackers]

    schedule = DifficultySchedule()
    replay = ReplayBuffer(max_items=2048)

    def forward_detector(x, url_vec, lex, ctx):
        return detector(x, url_vec=url_vec, lex=lex, ctx=ctx)

    def _attack_batch(x, ctx_clean, strength, G):
        style = torch.rand((x.shape[0], 16), device=device)
        p = G(x, style)
        bg_t = bg.sample(x.shape[0], size=x.shape[-1], device=device) if bg.available() else None
        x_adv = context_attack(x, bg_t, p, difficulty=strength)
        ctx_adv = ctx_enc(x_adv) if args.use_context else ctx_clean
        return x_adv, ctx_adv

    def _maybe_decode_filter(x_adv, y, url_vec, lex, ctx_adv, payloads):
        if not args.decode_filter:
            return x_adv, y, url_vec, lex, ctx_adv, payloads, None

        mask = _decode_mask(
            x_adv,
            payloads=payloads,
            image_size=args.image_size,
            subset=args.decode_subset,
            require_payload_match=bool(args.payload_match),
        ).to(device)
        kept = int(mask.sum().item())
        payloads2 = [payloads[i] for i in range(len(payloads)) if bool(mask[i].item())] if payloads is not None else None
        return (
            x_adv[mask],
            y[mask],
            (url_vec[mask] if url_vec is not None else None),
            (lex[mask] if lex is not None else None),
            (ctx_adv[mask] if ctx_adv is not None else None),
            payloads2,
            kept,
        )

    def _unpack_batch(batch):
        if not fusion:
            if args.use_context:
                if need_payloads:
                    x, ctx, y, payloads = batch
                else:
                    x, ctx, y = batch
                    payloads = None
                ctx_clean = _ctx_from_batch(ctx, x, device)
            else:
                if need_payloads:
                    x, y, payloads = batch
                else:
                    x, y = batch
                    payloads = None
                ctx_clean = None
            url_vec = lex = None
            return x.to(device), y.to(device), url_vec, lex, ctx_clean, payloads

        # fusion
        if need_payloads:
            x, url_vec, lex, ctx, y, payloads = batch
        else:
            x, url_vec, lex, ctx, y = batch
            payloads = None
        x = x.to(device)
        y = y.to(device)
        url_vec = url_vec.to(device)
        lex = lex.to(device)
        ctx_clean = ctx.to(device)  # tensor (maybe zeros)
        return x, y, url_vec, lex, ctx_clean, payloads

    def eval_clean_and_adv(round_idx: int):
        detector.eval()
        strength = schedule.strength(round_idx)
        correct_clean = 0
        correct_adv = 0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, url_vec, lex, ctx_clean, payloads = _unpack_batch(batch)

                logits = forward_detector(x, url_vec, lex, ctx_clean)
                pred = logits.argmax(dim=1)
                correct_clean += (pred == y).sum().item()

                x_adv, ctx_adv = _attack_batch(x, ctx_clean, strength, attackers[0])
                x_adv, y2, url2, lex2, ctx2, payloads2, _ = _maybe_decode_filter(x_adv, y, url_vec, lex, ctx_adv, payloads)
                if x_adv.size(0) > 0:
                    logits_adv = forward_detector(x_adv, url2, lex2, ctx2)
                    pred_adv = logits_adv.argmax(dim=1)
                    correct_adv += (pred_adv == y2).sum().item()
                    n += x_adv.size(0)

        detector.train()
        return correct_clean / max(1, len(val_ds)), correct_adv / max(1, n)

    for r in range(args.rounds):
        strength = schedule.strength(r)
        print(
            f"\n[Round {r+1}/{args.rounds}] difficulty_strength={strength:.2f} "
            f"mode={args.detector_mode} use_context={args.use_context} "
            f"decode_filter={args.decode_filter} payload_match={args.payload_match}"
        )

        # Phase 1: attacker
        _freeze(detector)
        for a in attackers:
            _unfreeze(a)

        it = iter(train_loader)
        for step in range(args.k_attack):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            x, y, url_vec, lex, ctx_clean, payloads = _unpack_batch(batch)

            gi = int(np.random.randint(0, len(attackers)))
            G = attackers[gi]
            opt = opt_atk[gi]
            opt.zero_grad(set_to_none=True)

            kept = None
            for attempt in range(max(1, args.decode_resample)):
                x_adv, ctx_adv = _attack_batch(x, ctx_clean, strength, G)
                x_f, y_f, url_f, lex_f, ctx_f, payloads_f, kept = _maybe_decode_filter(
                    x_adv, y, url_vec, lex, ctx_adv, payloads
                )
                if (not args.decode_filter) or x_f.size(0) >= args.decode_min_keep:
                    break

            if args.decode_filter and x_f.size(0) == 0:
                print(f"  [Attack {step+1}/{args.k_attack}] G{gi} skipped: 0 valid adv samples")
                continue

            logits_adv = forward_detector(x_f, url_f, lex_f, ctx_f)
            loss_det = F.cross_entropy(logits_adv, y_f)

            loss_preserve = F.l1_loss(F.avg_pool2d(x_f, kernel_size=8), F.avg_pool2d(x[: x_f.size(0)], kernel_size=8))
            loss_attack = (-loss_det) + args.lambda_preserve * loss_preserve
            loss_attack.backward()
            opt.step()

            replay.add(x_f, y_f, url_f, lex_f, ctx_f if fusion else None)
            if args.decode_filter:
                print(f"  [Attack {step+1}/{args.k_attack}] G{gi} keep={x_f.size(0)} loss_attack={loss_attack.item():.4f} loss_det={loss_det.item():.4f}")
            else:
                print(f"  [Attack {step+1}/{args.k_attack}] G{gi} loss_attack={loss_attack.item():.4f} loss_det={loss_det.item():.4f}")

        # Phase 2: defender
        for a in attackers:
            _freeze(a)
        _unfreeze(detector)

        it = iter(train_loader)
        for step in range(args.k_defense):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            x, y, url_vec, lex, ctx_clean, payloads = _unpack_batch(batch)

            opt_det.zero_grad(set_to_none=True)

            logits_clean = forward_detector(x, url_vec, lex, ctx_clean)
            loss_clean = F.cross_entropy(logits_clean, y)

            gi = int(np.random.randint(0, len(attackers)))
            x_adv, ctx_adv = _attack_batch(x, ctx_clean, strength, attackers[gi])
            x_f, y_f, url_f, lex_f, ctx_f, payloads_f, kept = _maybe_decode_filter(
                x_adv, y, url_vec, lex, ctx_adv, payloads
            )

            if args.decode_filter and x_f.size(0) == 0:
                loss = loss_clean
                loss.backward()
                opt_det.step()
                print(f"  [Defense {step+1}/{args.k_defense}] clean_only (0 valid adv)")
                continue

            logits_adv = forward_detector(x_f, url_f, lex_f, ctx_f)
            loss_adv = F.cross_entropy(logits_adv, y_f)

            probs_clean = torch.softmax(logits_clean.detach(), dim=1)
            probs_adv = torch.softmax(logits_adv, dim=1)
            loss_cons = F.mse_loss(probs_adv, probs_clean[: probs_adv.size(0)])

            rep = replay.sample(x_f.size(0), device=device)
            if rep is not None:
                x_rep, y_rep, url_rep, lex_rep, ctx_rep = rep
                logits_rep = forward_detector(x_rep, url_rep, lex_rep, ctx_rep)
                loss_rep = F.cross_entropy(logits_rep, y_rep)
            else:
                loss_rep = torch.tensor(0.0, device=device)

            loss = loss_clean + args.lambda_adv * loss_adv + args.lambda_cons * loss_cons + args.lambda_replay * loss_rep
            loss.backward()
            opt_det.step()

            if args.decode_filter:
                print(f"  [Defense {step+1}/{args.k_defense}] keep={x_f.size(0)} loss={loss.item():.4f} clean={loss_clean.item():.4f} adv={loss_adv.item():.4f} rep={loss_rep.item():.4f}")
            else:
                print(f"  [Defense {step+1}/{args.k_defense}] loss={loss.item():.4f} clean={loss_clean.item():.4f} adv={loss_adv.item():.4f} rep={loss_rep.item():.4f}")

        acc_c, acc_a = eval_clean_and_adv(r)
        print(f"  [VAL] acc_clean={acc_c:.4f} acc_adv(valid)={acc_a:.4f}")

        torch.save({"round": r, "model": detector.state_dict(), "cfg": det_cfg.__dict__}, out_dir / f"detector_round{r+1}.pt")
        for i, G in enumerate(attackers):
            torch.save({"round": r, "model": G.state_dict()}, out_dir / f"attacker{i}_round{r+1}.pt")


if __name__ == "__main__":
    main()
