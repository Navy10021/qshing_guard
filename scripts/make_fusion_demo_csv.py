# scripts/make_fusion_demo_csv.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def scan_images(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        return []
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def norm_slash(s: str) -> str:
    return str(s).replace("\\", "/")


def resolve_path(p: str, base_dir: Path) -> Path:
    """
    mapping_csv의 qr_path가
    - 절대경로면 그대로
    - 상대경로면 base_dir 기준으로 붙여서 resolve
    """
    p = norm_slash(p).strip()
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()


def pick_cols(df: pd.DataFrame) -> pd.DataFrame:
    # qr_path column normalize
    if "qr_path" not in df.columns:
        for c in ("qr_img_path", "qr_image_path", "image_path", "img_path", "path"):
            if c in df.columns:
                df = df.rename(columns={c: "qr_path"})
                break

    # url_norm column normalize
    if "url_norm" not in df.columns:
        for c in ("url", "url_raw", "url_normed", "url_clean"):
            if c in df.columns:
                df = df.rename(columns={c: "url_norm"})
                break

    if "qr_path" not in df.columns or "url_norm" not in df.columns:
        raise ValueError(
            f"[mapping_csv] must contain qr_path and url_norm (or recognizable aliases). "
            f"Got columns: {list(df.columns)}"
        )
    return df


def build_from_mapping_with_fallback(
    mapping_csv: Path,
    input_dir: Path,
    recursive: bool,
    prefer_col: str,
    limit: int,
) -> pd.DataFrame:
    df = pd.read_csv(mapping_csv)
    df = pick_cols(df)

    # 스캔된 이미지 목록
    scanned = scan_images(input_dir, recursive=recursive)
    scanned_norm = {norm_slash(p.as_posix()): p for p in scanned}
    scanned_names = {p.name: p for p in scanned}

    out_rows = []
    base_dir = Path(".").resolve()

    # 1) 우선 mapping_csv의 qr_path가 실제 존재하는지 확인해서 수집
    for _, r in df.iterrows():
        qr_raw = str(r["qr_path"])
        url = str(r[prefer_col]) if (prefer_col and prefer_col in df.columns) else str(r["url_norm"])

        # 후보 1: mapping_csv에 적힌 경로 그대로(상대/절대 모두)
        cand1 = resolve_path(qr_raw, base_dir)
        if cand1.exists():
            out_rows.append({"qr_path": cand1.as_posix(), "url_norm": url})
        else:
            # 후보 2: input_dir 기준 상대경로 (예: qr_images/xxx.png 대신 xxx.png만 들어있는 경우)
            cand2 = resolve_path(qr_raw, input_dir)
            if cand2.exists():
                out_rows.append({"qr_path": cand2.as_posix(), "url_norm": url})
            else:
                # 후보 3: 파일명만 매칭 (폴더가 달라진 경우)
                name = Path(norm_slash(qr_raw)).name
                if name in scanned_names:
                    out_rows.append({"qr_path": scanned_names[name].as_posix(), "url_norm": url})

        if len(out_rows) >= limit:
            break

    if len(out_rows) > 0:
        return pd.DataFrame(out_rows).drop_duplicates(subset=["qr_path"]).head(limit).reset_index(drop=True)

    # 2) 여기까지 왔다는 건:
    # - input_dir에 이미지가 없거나
    # - mapping_csv qr_path 경로가 현재 파일시스템과 전혀 안 맞음
    # 그러면 scan_images 결과가 있으면 그걸로만 demo를 만들되 url은 mapping에서 이름 매칭해서 붙임
    if len(scanned) > 0:
        df["qr_name"] = df["qr_path"].astype(str).map(lambda x: Path(norm_slash(x)).name)
        name_to_url = {}
        for _, r in df.iterrows():
            name_to_url[str(r["qr_name"])] = str(r[prefer_col]) if (prefer_col and prefer_col in df.columns) else str(r["url_norm"])

        rows = []
        for p in scanned[:limit]:
            url = name_to_url.get(p.name, f"https://example.com/demo/{p.stem}")
            rows.append({"qr_path": p.as_posix(), "url_norm": url})
        return pd.DataFrame(rows)

    # 3) 아무 것도 못 찾으면 명확하게 안내
    raise RuntimeError(
        "Could not find any QR image files.\n"
        f"- input_dir scanned: {input_dir} -> 0 images\n"
        f"- mapping_csv: {mapping_csv}\n"
        "Fix options:\n"
        "  (1) Run QR generation first and point --input_dir to the folder where images were created\n"
        "  (2) Ensure mapping_csv has correct qr_path paths that exist in your filesystem\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing QR images (can be empty; fallback to mapping paths)")
    ap.add_argument("--out_csv", type=str, default="samples/fusion_demo.csv")
    ap.add_argument("--recursive", action="store_true", help="Scan images recursively")
    ap.add_argument("--limit", type=int, default=20, help="Max number of rows to output (demo용)")
    ap.add_argument("--mapping_csv", type=str, required=True, help="CSV containing qr_path and url_norm (e.g., test_with_qr.csv)")
    ap.add_argument("--prefer_url_col", type=str, default="", help="If mapping_csv has multiple URL cols, choose this one")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    mapping_csv = Path(args.mapping_csv)
    if not mapping_csv.exists():
        raise FileNotFoundError(f"mapping_csv not found: {mapping_csv}")

    df_out = build_from_mapping_with_fallback(
        mapping_csv=mapping_csv,
        input_dir=input_dir,
        recursive=args.recursive,
        prefer_col=args.prefer_url_col,
        limit=max(1, args.limit),
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    print("[OK] Wrote:", out_csv)
    print("[OK] Rows:", len(df_out))
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
