from __future__ import annotations

import argparse
from typing import Optional

import cv2
from pyzbar.pyzbar import decode, ZBarSymbol


def decode_qr_image(path: str) -> Optional[str]:
    img = cv2.imread(path)
    if img is None:
        return None
    return decode_qr_array(img)


def decode_qr_array(img_bgr) -> Optional[str]:
    """Decode QR from an in-memory BGR image.

    NOTE:
    - Restrict symbols to QRCODE to avoid noisy zbar warnings from other barcode decoders (e.g., Databar).
    """
    codes = decode(img_bgr, symbols=[ZBarSymbol.QRCODE])
    if not codes:
        return None
    data = codes[0].data
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", type=str, help="Path to QR image")
    args = ap.parse_args()

    result = decode_qr_image(args.image_path)
    print(result if result is not None else "")


if __name__ == "__main__":
    main()
