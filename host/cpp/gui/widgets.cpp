#include "widgets.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <algorithm>

#include "tpu_host/digits.hpp"

namespace digits = tpu_host::digits;

namespace {
constexpr int CANVAS = 280;
constexpr int PEN = 28;  // stroke width: about 4 of 32 pixels, like sklearn's digits

// Ink amount (0..255) as a shade of the system palette: none = base, full = text.
QColor shade(const QPalette& pal, int ink) {
    const QColor b = pal.color(QPalette::Base), t = pal.color(QPalette::Text);
    const double f = ink / 255.0;
    return QColor::fromRgbF(float(b.redF() + (t.redF() - b.redF()) * f),
                            float(b.greenF() + (t.greenF() - b.greenF()) * f),
                            float(b.blueF() + (t.blueF() - b.blueF()) * f));
}

void frame(QPainter& p, const QWidget* w) {
    p.setPen(w->palette().color(QPalette::Mid));
    p.setBrush(Qt::NoBrush);
    p.drawRect(w->rect().adjusted(0, 0, -1, -1));
}

int inkOf(std::int8_t x) { return std::clamp(2 * x, 0, 255); }
}  // namespace

// ---- Canvas -----------------------------------------------------------------

Canvas::Canvas(QWidget* parent) : QWidget(parent), ink_(CANVAS, CANVAS, QImage::Format_Grayscale8) {
    setFixedSize(CANVAS, CANVAS);
    setCursor(Qt::CrossCursor);
    clear();
}

void Canvas::clear() {
    ink_.fill(0);
    empty_ = true;
    showingInput_ = false;
    update();
}

void Canvas::showInput(std::span<const std::int8_t> x) {
    ink_.fill(0);
    QPainter p(&ink_);
    const int cell = CANVAS / digits::SIDE;
    const int off = (CANVAS - cell * digits::SIDE) / 2;
    for (int i = 0; i < digits::PIXELS; ++i) {
        const int v = inkOf(x[static_cast<std::size_t>(i)]);
        p.fillRect(off + i % digits::SIDE * cell, off + i / digits::SIDE * cell, cell, cell,
                   QColor(v, v, v));
    }
    empty_ = false;
    showingInput_ = true;
    update();
}

std::vector<std::uint8_t> Canvas::bitmap() const {
    int x0 = CANVAS, y0 = CANVAS, x1 = -1, y1 = -1;
    for (int y = 0; y < CANVAS; ++y) {
        const uchar* line = ink_.constScanLine(y);
        for (int x = 0; x < CANVAS; ++x) {
            if (line[x] > 0) {
                x0 = std::min(x0, x), x1 = std::max(x1, x);
                y0 = std::min(y0, y), y1 = std::max(y1, y);
            }
        }
    }
    std::vector<std::uint8_t> out(digits::BITMAP * digits::BITMAP, 0);
    if (x1 < 0) {
        return out;
    }
    // Like sklearn's digits: the longer side fills the 32x32 box, centred.
    const QImage digit =
        ink_.copy(QRect(QPoint(x0, y0), QPoint(x1, y1)))
            .scaled(digits::BITMAP, digits::BITMAP, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    const int dx = (digits::BITMAP - digit.width()) / 2;
    const int dy = (digits::BITMAP - digit.height()) / 2;
    for (int y = 0; y < digit.height(); ++y) {
        const uchar* line = digit.constScanLine(y);
        for (int x = 0; x < digit.width(); ++x) {
            out[static_cast<std::size_t>((y + dy) * digits::BITMAP + x + dx)] = line[x];
        }
    }
    return out;
}

void Canvas::paintEvent(QPaintEvent*) {
    QPainter p(this);
    // The ink image as shades between the base and text colours.
    QImage shown(ink_.size(), QImage::Format_RGB32);
    QRgb lut[256];
    for (int i = 0; i < 256; ++i) {
        lut[i] = shade(palette(), i).rgb();
    }
    for (int y = 0; y < ink_.height(); ++y) {
        const uchar* src = ink_.constScanLine(y);
        auto* dst = reinterpret_cast<QRgb*>(shown.scanLine(y));
        for (int x = 0; x < ink_.width(); ++x) {
            dst[x] = lut[src[x]];
        }
    }
    p.drawImage(0, 0, shown);
    if (empty_) {
        p.setPen(palette().color(QPalette::PlaceholderText));
        p.drawText(rect(), Qt::AlignCenter, tr("Draw here"));
    }
    frame(p, this);
}

void Canvas::mousePressEvent(QMouseEvent* e) {
    if (e->button() != Qt::LeftButton) {
        return;
    }
    if (showingInput_) {
        clear();
    }
    drawing_ = true;
    last_ = e->position().toPoint();
    drawTo(last_);
}

void Canvas::mouseMoveEvent(QMouseEvent* e) {
    if (drawing_) {
        drawTo(e->position().toPoint());
    }
}

void Canvas::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton && drawing_) {
        drawing_ = false;
        emit strokeFinished();
    }
}

void Canvas::drawTo(QPoint pt) {
    QPainter p(&ink_);
    p.setRenderHint(QPainter::Antialiasing);
    p.setPen(QPen(Qt::white, PEN, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.drawLine(last_, pt);
    last_ = pt;
    empty_ = false;
    update();
}

// ---- InputPreview -----------------------------------------------------------

InputPreview::InputPreview(QWidget* parent) : QWidget(parent) { setFixedSize(66, 66); }

void InputPreview::setInput(std::vector<std::int8_t> x) {
    x_ = std::move(x);
    update();
}

void InputPreview::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), palette().color(QPalette::Base));
    if (x_.size() == digits::PIXELS) {
        const int cell = (width() - 2) / digits::SIDE;
        for (int i = 0; i < digits::PIXELS; ++i) {
            p.fillRect(1 + i % digits::SIDE * cell, 1 + i / digits::SIDE * cell, cell, cell,
                       shade(palette(), inkOf(x_[static_cast<std::size_t>(i)])));
        }
    }
    frame(p, this);
}
