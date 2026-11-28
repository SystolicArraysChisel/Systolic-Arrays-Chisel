// The demo's painted widgets: the drawing area and the network input preview.
// Both use the system palette, so they follow the Windows look.
#pragma once
#include <QImage>
#include <QPoint>
#include <QWidget>
#include <cstdint>
#include <span>
#include <vector>

// Where the digit is drawn with the mouse: black ink on the window's base colour.
class Canvas : public QWidget {
    Q_OBJECT
   public:
    explicit Canvas(QWidget* parent = nullptr);
    void clear();
    // Shows an 8x8 network input (a test-set digit) instead of a drawing.
    void showInput(std::span<const std::int8_t> x);
    bool isEmpty() const { return empty_; }
    // The drawing cropped to its ink, scaled to 32x32 (0 blank, 255 ink).
    std::vector<std::uint8_t> bitmap() const;

   signals:
    void strokeFinished();

   protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;

   private:
    void drawTo(QPoint p);
    QImage ink_;  // ink amount per pixel (white = full), the size of the widget
    QPoint last_;
    bool drawing_ = false;
    bool empty_ = true;
    bool showingInput_ = false;  // the next stroke starts a new drawing
};

// The 8x8 image the TPU receives, in the same colours as the canvas.
class InputPreview : public QWidget {
    Q_OBJECT
   public:
    explicit InputPreview(QWidget* parent = nullptr);
    void setInput(std::vector<std::int8_t> x);  // empty: nothing shown

   protected:
    void paintEvent(QPaintEvent*) override;

   private:
    std::vector<std::int8_t> x_;
};
