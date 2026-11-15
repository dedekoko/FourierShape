"""
fourier_shape.py — يرسم محيط أي شكل من صورة بسلسلة دوائر دوّارة (Fourier epicycles).

الفكرة:
    1. نستخرج محيط الشكل من الصورة (أكبر كفاف).
    2. نعيد أخذ نقاطه بتباعد متساوٍ على طول القوس، فتصير "سرعة المرور" ثابتة
       (وإلا تغيّرت المعاملات مع أن الشكل واحد، وهي نقطة عنوان الفيديو).
    3. نعامل كل نقطة عددًا مركّبًا z = x + iy ونحسب تحويل فورييه السريع (FFT).
       كل معامل = دائرة: حجمه نصف قطرها، وتردّده سرعة دورانها، وزاويته بدايتها.
    4. نرتّب الدوائر من الأكبر إلى الأصغر، ونرسم الشكل بعدد متزايد منها
       (1، 3، 5، 10، 20، 50، 100 ...) فيظهر كيف يكتمل الشكل.

الاستعمال:
    python fourier_shape.py                      # شكل كمان مولَّد داخليًّا
    python fourier_shape.py صورة.png             # أي صورة فيها شكل واحد واضح
    python fourier_shape.py صورة.png --out ناتج.mp4 --stages 1 3 5 10 20 50 100 --show

المتطلبات:
    pip install opencv-python numpy
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# لماذا هنا لا في وحدة منفصلة: السكربت مستقل ويُشغَّل أيضًا بـ python -I
# (لا يضيف مجلد السكربت إلى sys.path)، فنبقي دالّة النصوص داخله.
LANG = "ar"
_STRINGS = None


def t(key, **params):
    """يرجع نص الواجهة لمفتاح من locales/<LANG>.json بعد تعويض {name}.

    يُحمَّل الملف مرة واحدة نسبةً إلى مسار السكربت (يعمل من أي مجلد).
    إن غاب المفتاح يرجع المفتاح نفسه ويكتب تحذيرًا في stderr دون إيقاف البرنامج.
    """
    global _STRINGS
    if _STRINGS is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "locales", LANG + ".json")
        try:
            with open(path, encoding="utf-8") as f:
                _STRINGS = json.load(f)
        except (OSError, ValueError) as e:
            print(f"warning: cannot load {path}: {e}", file=sys.stderr)
            _STRINGS = {}
    text = _STRINGS.get(key)
    if text is None:
        print(f"warning: missing string key: {key}", file=sys.stderr)
        return key
    return text.format(**params) if params else text

# ألوان (BGR) — قريبة من ألوان الفيديو
BG = (20, 22, 12)
GHOST = (45, 52, 62)
TRACE = (90, 110, 240)
CIRCLE = (140, 170, 150)
ARM = (200, 220, 210)
TEXT = (90, 110, 240)


def make_default_shape(w=600, h=800):
    """يولّد صورة سوداء عليها شكل كمان أبيض (للتجربة دون صورة خارجية).

    يرجع: مصفوفة uint8 بقناة واحدة.
    """
    img = np.zeros((h, w), np.uint8)
    cx = w // 2
    cv2.ellipse(img, (cx, 600), (190, 160), 0, 0, 360, 255, -1)   # الجسم السفلي
    cv2.ellipse(img, (cx, 400), (140, 120), 0, 0, 360, 255, -1)   # الجسم العلوي
    cv2.rectangle(img, (cx - 95, 430), (cx + 95, 560), 255, -1)   # الخصر
    for s in (-1, 1):                                               # تقعّر الخصر
        cv2.ellipse(img, (cx + s * 175, 495), (70, 55), 0, 0, 360, 0, -1)
    cv2.rectangle(img, (cx - 22, 110), (cx + 22, 300), 255, -1)   # الذراع
    cv2.ellipse(img, (cx, 95), (38, 60), 0, 0, 360, 255, -1)      # الرأس
    for y in (65, 95, 125):                                         # المفاتيح
        cv2.rectangle(img, (cx - 58, y - 8), (cx + 58, y + 8), 255, -1)
    return img


def to_gray_on_white(img):
    """يحوّل أي صورة مقروءة (رمادية، ملوّنة، أو بقناة شفافية) إلى رمادية.

    لماذا: صور PNG ذات الخلفية الشفافة تكون بكسلات خلفيتها سوداء غالبًا،
    فإن حُوّلت إلى رمادي مباشرة اختفى الفرق بين الشكل الأسود والخلفية.
    لذلك نركّب الصورة على خلفية بيضاء حسب قناة الشفافية أولًا.
    """
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        a = img[:, :, 3]
        # لماذا: إن كانت في الصورة شفافية حقيقية فقناتها أدقّ فاصل بين الشكل
        # والخلفية، ولا تتأثر بالخطوط والألوان داخل الشكل (مثل حدود المحافظات).
        if a.min() < 128 and a.max() > 128:
            return 255 - a
        alpha = img[:, :, 3:4].astype(float) / 255.0
        rgb = img[:, :, :3].astype(float) * alpha + 255.0 * (1 - alpha)
        img = rgb.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extract_contour(gray):
    """يستخرج أكبر كفاف من صورة رمادية.

    يحدّد العتبة تلقائيًّا (Otsu)، ويعكس الصورة إن كانت الخلفية فاتحة،
    ويرجع مصفوفة نقاط (n, 2) من نوع float.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # لماذا: إن كانت أغلب الحواف بيضاء فالخلفية فاتحة والشكل داكن، فنعكس.
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127:
        bw = 255 - bw
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        sys.exit(t("error.no_shape"))
    c = max(contours, key=cv2.contourArea)
    return c[:, 0, :].astype(float), bw


def resample_by_arclength(pts, n):
    """يعيد أخذ n نقطة متساوية التباعد على طول المحيط المغلق.

    لماذا: معاملات فورييه تتبع طريقة المرور على المحيط لا الشكل وحده؛
    التباعد المتساوي يثبّت السرعة فتصير المعاملات وصفًا للشكل فقط.
    """
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(t, s, closed[:, 0])
    y = np.interp(t, s, closed[:, 1])
    return x + 1j * y


def fourier_terms(z):
    """يحسب الدوائر من نقاط المحيط المركّبة.

    يرجع: (المركز c0، قائمة (تردّد k، معامل مركّب c) مرتّبة تنازليًّا بالحجم).
    """
    n = len(z)
    coeffs = np.fft.fft(z) / n
    freqs = np.fft.fftfreq(n, d=1.0 / n).astype(int)   # ..., -2, -1, 0, 1, 2, ...
    c0 = coeffs[freqs == 0][0]
    terms = [(k, c) for k, c in zip(freqs, coeffs) if k != 0]
    terms.sort(key=lambda kc: -abs(kc[1]))
    return c0, terms


def chain_points(c0, terms, t):
    """يرجع مراكز سلسلة الدوائر عند الزمن t (من 0 إلى 2π)، آخرها رأس القلم."""
    pos = [c0]
    for k, c in terms:
        pos.append(pos[-1] + c * np.exp(1j * k * t))
    return pos


def curve(c0, terms, ts):
    """يرجع نقاط الشكل المعاد بناؤه بهذه الدوائر عند كل الأزمنة ts."""
    out = np.full(len(ts), c0, complex)
    for k, c in terms:
        out += c * np.exp(1j * k * ts)
    return out


def to_px(z, scale, off):
    """يحوّل نقطة مركّبة إلى إحداثيات بكسل صحيحة على اللوحة."""
    return (int(round(z.real * scale + off[0])), int(round(z.imag * scale + off[1])))


def render(z, ghost_mask, stages, out_path, size=(720, 900), frames_per_stage=150,
           fps=30, show=False):
    """يرسم الفيديو: لكل مرحلة عدد دوائر، تدور السلسلة دورة كاملة وترسم الشكل.

    z: نقاط المحيط المركّبة. ghost_mask: قناع الشكل الأصلي (ظلّ خلفي).
    stages: أعداد الدوائر. out_path: ملف mp4 الناتج. show: عرض نافذة حيّة.
    """
    W, H = size
    c0, terms = fourier_terms(z)

    # ملاءمة الشكل في اللوحة مع هامش للدوائر
    xs, ys = z.real, z.imag
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    scale = 0.62 * min(W, H) / span
    center = (xs.min() + xs.max()) / 2 + 1j * (ys.min() + ys.max()) / 2
    off = (W / 2 - center.real * scale, H * 0.47 - center.imag * scale)

    # ظلّ الشكل الأصلي بالمقياس نفسه
    M = np.float32([[scale, 0, off[0]], [0, scale, off[1]]])
    ghost = cv2.warpAffine(ghost_mask, M, (W, H)) > 127

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    ts_all = np.linspace(0, 2 * np.pi, frames_per_stage)

    for n in stages:
        sel = terms[:n]
        full = curve(c0, sel, np.linspace(0, 2 * np.pi, 800))
        for i, t in enumerate(ts_all):
            img = np.full((H, W, 3), BG, np.uint8)
            img[ghost] = GHOST
            # الأثر المرسوم حتى الآن
            done = full[: max(2, int(len(full) * (i + 1) / len(ts_all)))]
            pts = np.array([to_px(p, scale, off) for p in done], np.int32)
            cv2.polylines(img, [pts], False, TRACE, 3, cv2.LINE_AA)
            # سلسلة الدوائر
            chain = chain_points(c0, sel, t)
            for (k, c), a, b in zip(sel, chain[:-1], chain[1:]):
                r = abs(c) * scale
                if r > 1.5:
                    cv2.circle(img, to_px(a, scale, off), int(r), CIRCLE, 1, cv2.LINE_AA)
                cv2.line(img, to_px(a, scale, off), to_px(b, scale, off), ARM, 1, cv2.LINE_AA)
            cv2.circle(img, to_px(chain[-1], scale, off), 5, (255, 255, 255), -1, cv2.LINE_AA)
            label = str(n)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
            cv2.putText(img, label, ((W - tw) // 2, H - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.6, TEXT, 3, cv2.LINE_AA)
            writer.write(img)
            if show:
                cv2.imshow(t("window.title"), img)
                if cv2.waitKey(1) & 0xFF == 27:   # Esc للخروج
                    writer.release()
                    return
        # لقطة ختامية: الشكل المعاد بناؤه مملوءًا
        img = np.full((H, W, 3), BG, np.uint8)
        pts = np.array([to_px(p, scale, off) for p in full], np.int32)
        cv2.fillPoly(img, [pts], (60, 75, 190), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, TRACE, 3, cv2.LINE_AA)
        cv2.putText(img, f"{n}", ((W - tw) // 2, H - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, TEXT, 3, cv2.LINE_AA)
        for _ in range(fps // 2):
            writer.write(img)
    writer.release()
    if show:
        cv2.destroyAllWindows()


def main():
    """نقطة الدخول: يقرأ الخيارات، يجهّز الشكل، ويُنتج الفيديو."""
    ap = argparse.ArgumentParser(description=t("cli.description"))
    ap.add_argument("image", nargs="?", help=t("cli.image_help"))
    ap.add_argument("--out", default="fourier_shape.mp4", help=t("cli.out_help"))
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100],
                    help=t("cli.stages_help"))
    ap.add_argument("--points", type=int, default=1024, help=t("cli.points_help"))
    ap.add_argument("--show", action="store_true", help=t("cli.show_help"))
    a = ap.parse_args()

    if a.image:
        # لماذا imdecode: cv2.imread لا يقرأ المسارات العربية في ويندوز
        data = np.fromfile(a.image, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(t("error.image_read"))
        gray = to_gray_on_white(img)
    else:
        gray = make_default_shape()

    pts, mask = extract_contour(gray)
    z = resample_by_arclength(pts, a.points)
    render(z, mask, a.stages, a.out, show=a.show)
    print(t("done.message", out=a.out))


if __name__ == "__main__":
    main()
