"""Train a 64-16-10 MLP to recognise drawn shapes and symbols, quantize it for the
accelerator and write host/cpp/demo/shapes_model.json.

No dataset: the shapes are drawn here, the way the demo GUI draws them (280 px canvas,
28 px round pen), with random size, proportions, rotation and wobble. Each drawing then
goes through the GUI's conversion to network input (Canvas::bitmap and digits::features):
crop to the ink, scale the longer side to 32 pixels, count the inked pixels (>= 128) of
each 4x4 block. Same network and integer pipeline as train_digits.py.
"""
import numpy as np
from PIL import Image, ImageDraw

from train_digits import DEMO_DIR, export, train

CLASSES = ["circle", "square", "triangle", "star", "heart",
           "arrow", "check", "cross", "lightning", "smiley"]
CANVAS, PEN = 280, 28  # as in host/cpp/gui/widgets.cpp
TRAIN_PER_CLASS, TEST_PER_CLASS = 2000, 300


# ---- Shapes: lists of strokes, each an (n, 2) array of points in about [-1, 1] ----

def arc(cx, cy, rx, ry, a0, a1, n=60):
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def poly(*pts):
    return np.array(pts, dtype=float)


def shape(name, rng):
    u = lambda lo, hi: rng.uniform(lo, hi)
    if name == "circle":
        a0 = u(0, 360)
        return [arc(0, 0, 1, 1, a0, a0 + u(345, 380))]
    if name == "square":
        p = poly((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))
        return [p + rng.normal(0, 0.05, p.shape)]
    if name == "triangle":
        apex = u(-0.3, 0.3)
        p = poly((apex, -1), (1, 1), (-1, 1), (apex, -1))
        return [p + rng.normal(0, 0.05, p.shape)]
    if name == "star":
        a = np.radians(-90 + 144 * np.arange(6))
        return [np.stack([np.cos(a), np.sin(a)], 1)]
    if name == "heart":
        t = np.linspace(0, 2 * np.pi, 80)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        return [np.stack([x, y], 1) / 16]
    if name == "arrow":
        head = u(0.35, 0.6)
        strokes = [poly((-1, 0), (1, 0)), poly((1 - head, -head), (1, 0), (1 - head, head))]
        turn = rng.integers(4) * np.pi / 2  # right, down, left or up
        c, s = np.cos(turn), np.sin(turn)
        return [p @ np.array([[c, s], [-s, c]]) for p in strokes]
    if name == "check":
        knee = u(-0.5, -0.1)
        return [poly((-1, u(-0.1, 0.3)), (knee, 1), (1, -1))]
    if name == "cross":
        return [poly((-1, -1), (1, 1)), poly((1, -1), (-1, 1))]
    if name == "lightning":
        return [poly((0.3, -1), (-0.5, 0.1), (0.4, 0.0), (-0.3, 1))]
    if name == "smiley":
        eye_y, eye_x = u(-0.45, -0.25), u(0.3, 0.45)
        return [arc(0, 0, 1, 1, 0, 360),
                poly((-eye_x, eye_y), (-eye_x, eye_y + 0.08)),
                poly((eye_x, eye_y), (eye_x, eye_y + 0.08)),
                arc(0, u(0.0, 0.15), 0.55, 0.45, 20, 160)]
    raise ValueError(name)


def resample(p, step=0.03):
    """Points every `step` along the polyline, so wobble and rendering are smooth."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], max(2, int(s[-1] / step)))
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def hand_drawn(strokes, rng):
    """Random proportions, rotation, shear and a slow wobble, like a mouse drawing."""
    rot = np.radians(rng.uniform(-15, 15))
    c, s = np.cos(rot), np.sin(rot)
    m = np.array([[c, -s], [s, c]]) @ np.array([[rng.uniform(0.75, 1.25), rng.uniform(-0.15, 0.15)],
                                                 [0, rng.uniform(0.75, 1.25)]])
    out = []
    for p in strokes:
        p = resample(p) @ m.T
        t = np.linspace(0, 1, len(p))[:, None]
        for _ in range(2):
            freq, phase = rng.uniform(0.5, 3), rng.uniform(0, 2 * np.pi, 2)
            p = p + rng.uniform(0.01, 0.04) * np.sin(2 * np.pi * freq * t + phase)
        out.append(p)
    return out


def render(strokes, rng):
    """Draws the strokes on a CANVAS x CANVAS image with the GUI's pen, at a random size."""
    pts = np.concatenate(strokes)
    lo, hi = pts.min(0), pts.max(0)
    size = rng.uniform(110, 240)
    scale = size / max(hi - lo)
    centre = rng.uniform(CANVAS / 2 - 20, CANVAS / 2 + 20, 2)
    img = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(img)
    r = PEN / 2
    for p in strokes:
        q = (p - (lo + hi) / 2) * scale + centre
        draw.line([tuple(v) for v in q], fill=255, width=PEN, joint="curve")
        for x, y in (q[0], q[-1]):  # round caps, like Qt::RoundCap
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return img


def features(img):
    """Canvas::bitmap followed by digits::features, as 64 values in 0..1."""
    a = np.asarray(img)
    ys, xs = np.nonzero(a)
    crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = crop.size
    # QSize::scaled(32, 32, Qt::KeepAspectRatio), integer arithmetic as in Qt
    rw = h and 32 * w // h
    nw, nh = (rw, 32) if rw <= 32 else (32, 32 * h // w)
    small = np.asarray(crop.resize((max(nw, 1), max(nh, 1)), Image.BOX))
    box = np.zeros((32, 32), np.uint8)
    dy, dx = (32 - small.shape[0]) // 2, (32 - small.shape[1]) // 2
    box[dy:dy + small.shape[0], dx:dx + small.shape[1]] = small
    counts = (box >= 128).reshape(8, 4, 8, 4).sum((1, 3))
    return counts.reshape(64) / 16.0


def dataset(per_class, rng):
    x, y = [], []
    for label, name in enumerate(CLASSES):
        for _ in range(per_class):
            x.append(features(render(hand_drawn(shape(name, rng), rng), rng)))
            y.append(label)
    order = rng.permutation(len(y))
    return np.array(x)[order], np.array(y)[order]


def main():
    rng = np.random.default_rng(0)
    xtr, ytr = dataset(TRAIN_PER_CLASS, rng)
    xte, yte = dataset(TEST_PER_CLASS, rng)
    params = train(xtr, ytr, rng, classes=len(CLASSES), epochs=60)
    export(DEMO_DIR / "shapes_model.json", "Shapes and symbols", CLASSES, params, xtr, xte, yte,
           test_rows=300)


if __name__ == "__main__":
    main()
