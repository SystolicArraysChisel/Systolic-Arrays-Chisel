"""Train a 64-16-10 MLP on sklearn's digits, quantize it for the accelerator and
write host/cpp/demo/digits_model.json (read by host/cpp/src/run_mlp.cpp).

Hidden layer: bias + requant + leaky ReLU (slope 1/8). Output: bias, raw int32 logits.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

LEAK_SHIFT = 3  # leaky slope 1/8
QMAX = 127


def leaky(z):
    return np.where(z >= 0, z, z / 8)


def train(x, y, rng, epochs=300, lr=0.1, batch=64):
    w1, b1 = rng.normal(0, 0.3, (64, 16)), np.zeros(16)
    w2, b2 = rng.normal(0, 0.3, (16, 10)), np.zeros(10)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), batch):
            xb, yb = x[order[i:i + batch]], y[order[i:i + batch]]
            z1 = xb @ w1 + b1
            h = leaky(z1)
            z2 = h @ w2 + b2
            p = np.exp(z2 - z2.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            p[np.arange(len(yb)), yb] -= 1  # d(cross-entropy)/d(logits)
            p /= len(yb)
            dh = (p @ w2.T) * np.where(z1 >= 0, 1, 1 / 8)
            w2 -= lr * h.T @ p
            b2 -= lr * p.sum(0)
            w1 -= lr * xb.T @ dh
            b1 -= lr * dh.sum(0)
    return w1, b1, w2, b2


def rhu(v):  # round half up, like the hardware
    return np.floor(v + 0.5).astype(np.int64)


def quant_w(w):  # symmetric per-tensor int8
    s = np.abs(w).max() / QMAX
    return rhu(w / s), s


def int_forward(xq, w1q, b1q, shift, w2q, b2q):
    acc = xq @ w1q + b1q
    h = np.clip((acc + (1 << (shift - 1))) >> shift, -QMAX, QMAX)  # requant (shift > 0)
    h = np.where(h >= 0, h, h >> LEAK_SHIFT)  # leaky ReLU, floor
    return h @ w2q + b2q  # raw int32 logits


def main():
    d = load_digits()
    x = d.data / 16.0  # pixels 0..16 -> 0..1
    xtr, xte, ytr, yte = train_test_split(x, d.target, test_size=0.25, random_state=0)
    w1, b1, w2, b2 = train(xtr, ytr, np.random.default_rng(0))
    float_pred = (leaky(xte @ w1 + b1) @ w2 + b2).argmax(1)

    s_x = 1 / QMAX  # int8 input: round(pixel / 16 * 127)
    w1q, s_w1 = quant_w(w1)
    b1q = rhu(b1 / (s_x * s_w1))
    xq = rhu(xte / s_x)
    acc1 = rhu(xtr / s_x) @ w1q + b1q
    # smallest shift for which no training activation saturates
    shift = next(s for s in range(1, 31) if np.abs((acc1 + (1 << (s - 1))) >> s).max() <= QMAX)
    s_h = s_x * s_w1 * 2**shift  # scale of the int8 hidden layer
    w2q, s_w2 = quant_w(w2)
    b2q = rhu(b2 / (s_h * s_w2))
    int_pred = int_forward(xq, w1q, b1q, shift, w2q, b2q).argmax(1)
    print(f"float accuracy   {(float_pred == yte).mean():.4f}")
    print(f"integer accuracy {(int_pred == yte).mean():.4f}  (shift1={shift}, leak_shift={LEAK_SHIFT})")

    out = Path(__file__).resolve().parent.parent / "host/cpp/demo/digits_model.json"
    model = dict(w1=w1q, b1=b1q, shift1=shift, leak_shift=LEAK_SHIFT, w2=w2q, b2=b2q,
                 x=xq, labels=yte, expected=int_pred)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({k: v.tolist() if hasattr(v, "tolist") else v for k, v in model.items()},
                              separators=(",", ":")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
