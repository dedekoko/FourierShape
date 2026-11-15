"""
fourier_shape.py — يرسم محيط أي شكل من صورة بسلسلة دوائر دوّارة (Fourier epicycles).

خطوات البرنامج، ولكل خطوة دالّة باسمها، وقسم بالرقم نفسه في docs/شرح-طريقة-العمل.md:
    الخطوة 2  فصل الشكل وتتبّع حدّه ........ extract_contour()
    الخطوة 3  أخذ نقاط متساوية التباعد ...... sample_points()
    الخطوة 4  حساب المركز .................. compute_center()
    الخطوة 6  قياس الفجوات ................. measure_gaps()
              (قبل الدائرة الأولى: الفجوة = بُعد النقطة عن المركز)
    الخطوة 6  قياس نصف قطر دائرة وزاويتها .. fit_circle()
    الخطوة 8  اختيار الدائرة التالية ........ find_next_circle()
    الخطوة 10 طرح الدائرة من الفجوات ........ remove_circle_from_gaps()
    الخطوات معًا: الدوائر واحدة بعد أخرى .... build_circles()

    الدائرة الأولى = find_next_circle(الفجوات عن المركز)
    الدائرة الثانية = find_next_circle(ما تبقّى من الفجوات بعد طرح الأولى)
    ... وهكذا.

الاستعمال:
    python fourier_shape.py                      # شكل كمان مولَّد داخليًّا
    python fourier_shape.py صورة.png             # أي صورة فيها شكل واحد واضح
    python fourier_shape.py صورة.png --out ناتج.mp4 --stages 1 3 5 10 20 50 100 --show
    python fourier_shape.py صورة.png --explain 5 # يطبع حساب أوّل 5 دوائر خطوة بخطوة

المتطلبات:
    pip install opencv-python numpy
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

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


# ════════════════════════════════════════════════════════════════════════════
# الخطوة 2: فصل الشكل عن الخلفية وتتبّع حدّه          (الشرح: القسم 2)
# ════════════════════════════════════════════════════════════════════════════

def extract_contour(gray):
    """يفصل الشكل عن الخلفية ويرجع حدّه نقطةً نقطة.

    المدخل: صورة رمادية (الشكل بلون والخلفية بلون آخر).
    السلوك: يحدّد عتبة السطوع تلقائيًّا (Otsu)، ويعدّ لون حوافّ الصورة لون
    الخلفية فيعكس إن لزم، ثم يأخذ أكبر قطعة متّصلة فقط ويتتبّع حدّها.
    يرجع: (نقاط الحدّ بالترتيب كمصفوفة (n, 2)، قناع الشكل).
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


# ════════════════════════════════════════════════════════════════════════════
# الخطوة 3: أخذ نقاط متساوية التباعد على المحيط        (الشرح: القسم 3 و14)
# ════════════════════════════════════════════════════════════════════════════

def sample_points(contour, n):
    """يضع n نقطة بمسافات متساوية تمامًا على المحيط المغلق.

    المدخل: نقاط الحدّ بالترتيب (n, 2)، وعدد النقاط المطلوب n (نحن نختاره).
    يرجع: n عددًا مركّبًا z_j = x_j + i·y_j، أوّلها نقطة بداية القلم.
    لماذا التباعد المتساوي: ليمشي القلم بسرعة ثابتة؛ لو تزاحمت النقاط في مكان
    وتباعدت في آخر لتغيّرت الدوائر مع أن الشكل لم يتغيّر.
    """
    closed = np.vstack([contour, contour[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])            # المسافة المقطوعة عند كل بكسل
    targets = np.linspace(0, s[-1], n, endpoint=False)  # مسافات متساوية على المحيط
    x = np.interp(targets, s, closed[:, 0])
    y = np.interp(targets, s, closed[:, 1])
    return x + 1j * y


# ════════════════════════════════════════════════════════════════════════════
# الخطوة 4: حساب المركز، نقطة انطلاق السلسلة            (الشرح: القسم 4)
# ════════════════════════════════════════════════════════════════════════════

def compute_center(z):
    """يحسب المركز: متوسط كل نقاط المحيط.

    المدخل: النقاط المركّبة z.
    يرجع: عددًا مركّبًا واحدًا c0. هو ثابت لا يدور، ومنه تخرج الدائرة الأولى.
    """
    return z.mean()


# ════════════════════════════════════════════════════════════════════════════
# الدائرة: ذراع بطول ثابت تدور بسرعة ثابتة               (الشرح: القسم 5)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Circle:
    """دائرة واحدة في السلسلة.

    k: سرعة الدوران، أي عدد اللفّات خلال رسم الشكل مرة واحدة (السالبة عكس الموجبة).
    coef: عدد مركّب يجمع نصف القطر وزاوية البداية معًا.
    """
    k: int
    coef: complex

    @property
    def radius(self):
        """نصف قطر الدائرة: طول الذراع."""
        return abs(self.coef)

    @property
    def start_angle_deg(self):
        """زاوية بداية الذراع بالدرجات."""
        return float(np.degrees(np.angle(self.coef)))

    def arm_at(self, t):
        """متّجه الذراع عند الزمن t (من 0 إلى 2π): من مركز الدائرة إلى طرفها."""
        return self.coef * np.exp(1j * self.k * t)


def point_times(n):
    """يرجع زمن كل نقطة: النقطة j يمرّ بها القلم عند الزمن 2π·j/n."""
    return 2 * np.pi * np.arange(n) / n


# ════════════════════════════════════════════════════════════════════════════
# الخطوة 6: قياس الفجوات                                   (الشرح: القسم 6 و10)
# ════════════════════════════════════════════════════════════════════════════

def measure_gaps(z, center, circles=()):
    """يقيس الفجوة عند كل نقطة: من طرف آخر ذراع إلى النقطة الحقيقية.

    المدخل: النقاط z، المركز، والدوائر المستعملة حتى الآن.
    - بلا دوائر: الفجوة = بُعد النقطة عن المركز (z − c0)؛ منها تُقاس الدائرة الأولى.
    - بعد الدائرة الأولى: الفجوة = ما عجزت عنه الأولى؛ منها تُقاس الثانية. وهكذا.
    يرجع: فجوة مركّبة لكل نقطة (اتجاهها وطولها).
    """
    t = point_times(len(z))
    gaps = z - center
    for c in circles:
        gaps = gaps - c.arm_at(t)
    return gaps


def remove_circle_from_gaps(gaps, circle):
    """يطرح ذراع دائرة جديدة من الفجوات، فيبقى ما لم تسدّه بعد.

    هو نفس measure_gaps بعد إضافة الدائرة، لكن دون إعادة طرح كل الدوائر السابقة.
    """
    return gaps - circle.arm_at(point_times(len(gaps)))


# ════════════════════════════════════════════════════════════════════════════
# الخطوة 6: قياس نصف قطر دائرة وزاوية بدايتها            (الشرح: القسم 6 و7)
# ════════════════════════════════════════════════════════════════════════════

def useful_lengths(gaps, k, start_angle):
    """الطول النافع لكل فجوة: الجزء منها الواقع على اتجاه الذراع في لحظتها.

    المدخل: الفجوات، سرعة الذراع k، وزاوية بدايتها (راديان).
    عند النقطة j تكون الذراع قد دارت بزاوية k·2π·j/n فوق زاوية البداية.
    يرجع: طولًا نافعًا لكل نقطة (قد يكون سالبًا إن كانت الفجوة عكس الذراع).
    للشرح والطباعة فقط؛ fit_circle تصل إلى المتوسط مباشرة.
    """
    direction = np.exp(1j * (k * point_times(len(gaps)) + start_angle))
    return (gaps * np.conj(direction)).real


def fit_circle(gaps, k):
    """يقيس الدائرة التي تسدّ الفجوات أفضل ما يمكن وهي تلفّ بسرعة k.

    نصف قطرها = متوسط الطول النافع للفجوات على اتجاه ذراعها، وزاويتها هي
    الاتجاه الذي يجعل هذا المتوسط أكبر ما يمكن. المعادلة تعطي الاثنين معًا:
        c_k = (1/N) · Σ gap_j · e^(−i·k·2π·j/N)
    يرجع: Circle(k, c_k).
    """
    return Circle(int(k), complex(np.mean(gaps * np.exp(-1j * k * point_times(len(gaps))))))


# ════════════════════════════════════════════════════════════════════════════
# الخطوة 8: اختيار الدائرة التالية                          (الشرح: القسم 8 و9)
# ════════════════════════════════════════════════════════════════════════════

def find_next_circle(gaps, used_speeds):
    """يجرّب كل سرعة لم تُستعمل بعد، ويرجع الدائرة الأكبر منها.

    لماذا الأكبر: تسدّ أكبر جزء من الفجوات، فيكون الرسم بعدد الدوائر هذا
    أقرب ما يمكن إلى الأصل.
    كيف: fit_circle لكل سرعة k على حدة تعطي ما يعطيه تحويل فورييه السريع (FFT)
    للفجوات دفعة واحدة، فنستعمله لأنه أسرع بكثير والنتيجة واحدة.
    """
    n = len(gaps)
    all_circles = np.fft.fft(gaps) / n                     # fit_circle لكل k معًا
    speeds = np.fft.fftfreq(n, d=1.0 / n).astype(int)      # ..., -2, -1, 0, 1, 2, ...
    sizes = np.abs(all_circles)
    sizes[speeds == 0] = -1                                # السرعة 0 هي المركز، لا دائرة
    for k in used_speeds:
        sizes[speeds == k] = -1
    best = int(np.argmax(sizes))
    return Circle(int(speeds[best]), complex(all_circles[best]))


# ════════════════════════════════════════════════════════════════════════════
# الخطوات معًا: بناء الدوائر واحدة بعد أخرى              (الشرح: القسم 10 و11)
# ════════════════════════════════════════════════════════════════════════════

def build_circles(z, count, log=None):
    """يبني المركز وأوّل count دائرة، كل دائرة من الفجوات التي تركتها السابقة.

    1. المركز = compute_center.
    2. الفجوات الأولى = بُعد النقاط عن المركز (measure_gaps بلا دوائر).
    3. كرّر: الدائرة التالية = find_next_circle(الفجوات)، ثم اطرحها من الفجوات.
    log: دالّة طباعة اختيارية تستقبل (رقم الدائرة، الدائرة، متوسط الفجوة قبلها).
    يرجع: (المركز، قائمة الدوائر بالترتيب من الأكبر إلى الأصغر).
    """
    count = min(count, len(z) - 1)                       # أقصى عدد = عدد النقاط − 1
    center = compute_center(z)
    gaps = measure_gaps(z, center)
    circles = []
    for i in range(count):
        circle = find_next_circle(gaps, {c.k for c in circles})
        if log:
            log(i + 1, circle, float(np.abs(gaps).mean()))
        circles.append(circle)
        gaps = remove_circle_from_gaps(gaps, circle)
    return center, circles


# ════════════════════════════════════════════════════════════════════════════
# الرسم: مواضع الأذرع والشكل المعاد بناؤه                  (الشرح: القسم 15)
# ════════════════════════════════════════════════════════════════════════════

def chain_points(center, circles, t):
    """يرجع مواضع مفاصل السلسلة عند الزمن t: المركز ثم طرف كل ذراع، آخرها القلم."""
    pos = [center]
    for c in circles:
        pos.append(pos[-1] + c.arm_at(t))
    return pos


def curve(center, circles, ts):
    """يرجع موضع القلم عند كل زمن في ts: المركز + مجموع الأذرع."""
    out = np.full(len(ts), center, complex)
    for c in circles:
        out += c.arm_at(ts)
    return out


def to_px(z, scale, off):
    """يحوّل نقطة مركّبة إلى إحداثيات بكسل صحيحة على اللوحة."""
    return (int(round(z.real * scale + off[0])), int(round(z.imag * scale + off[1])))


def render(z, ghost_mask, center, circles, stages, out_path, size=(720, 900),
           frames_per_stage=150, fps=30, show=False):
    """يرسم الفيديو: لكل مرحلة عدد دوائر، تدور السلسلة دورة كاملة وترسم الشكل.

    z: نقاط المحيط المركّبة. ghost_mask: قناع الشكل الأصلي (ظلّ خلفي).
    center وcircles: ناتج build_circles. stages: أعداد الدوائر في كل مرحلة.
    out_path: ملف mp4 الناتج. show: عرض نافذة حيّة.
    """
    W, H = size

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
        sel = circles[:n]
        full = curve(center, sel, np.linspace(0, 2 * np.pi, 800))
        for i, t in enumerate(ts_all):
            img = np.full((H, W, 3), BG, np.uint8)
            img[ghost] = GHOST
            # الأثر المرسوم حتى الآن
            done = full[: max(2, int(len(full) * (i + 1) / len(ts_all)))]
            pts = np.array([to_px(p, scale, off) for p in done], np.int32)
            cv2.polylines(img, [pts], False, TRACE, 3, cv2.LINE_AA)
            # سلسلة الدوائر
            chain = chain_points(center, sel, t)
            for c, a, b in zip(sel, chain[:-1], chain[1:]):
                r = c.radius * scale
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
    ap.add_argument("--explain", type=int, default=0, metavar="N", help=t("cli.explain_help"))
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

    contour, mask = extract_contour(gray)                    # الخطوة 2
    z = sample_points(contour, a.points)                     # الخطوة 3

    def explain(i, circle, mean_gap):
        """يطبع سطرًا لكل دائرة من أوّل N: الفجوة قبلها، سرعتها، نصف قطرها، زاويتها."""
        if i <= a.explain:
            print(t("explain.circle", i=i, gap=mean_gap, k=circle.k,
                    r=circle.radius, angle=circle.start_angle_deg))

    if a.explain:
        c0 = compute_center(z)
        print(t("explain.center", n=len(z), x=c0.real, y=c0.imag))
    # الخطوات 4 إلى 10: المركز ثم الدوائر واحدة بعد أخرى من الفجوات
    center, circles = build_circles(z, max(a.stages), log=explain if a.explain else None)
    render(z, mask, center, circles, a.stages, a.out, show=a.show)
    print(t("done.message", out=a.out))


if __name__ == "__main__":
    main()
