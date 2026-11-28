// Second screen: draw, the chosen model classifies the drawing on the TPU.
#pragma once
#include <QComboBox>
#include <QLabel>
#include <QProgressBar>
#include <QWidget>
#include <cstdint>
#include <memory>
#include <vector>

#include "tpu_host/digits.hpp"
#include "tpu_host/tpu_driver.hpp"

class Canvas;
class InputPreview;
class FpgaProgrammer;
class QProgressDialog;
class QGridLayout;

// "Simulator" followed by the serial ports, listed again each time it opens.
class TargetBox : public QComboBox {
    Q_OBJECT
   public:
    static inline const QString SIMULATOR = QStringLiteral("Simulator");
    explicit TargetBox(QWidget* parent = nullptr);
    void showPopup() override;

   private:
    void refresh();
};

class DemoPage : public QWidget {
    Q_OBJECT
   public:
    explicit DemoPage(QWidget* parent = nullptr);
    ~DemoPage() override;
    void setModel(const QString& path);

   signals:
    void backRequested();

   private:
    void connectTarget();
    void disconnectTarget();
    void classify(const std::vector<std::int8_t>& x);
    void showExample();
    void clearResult();
    void buildScores();  // one row per class of model_
    void showPrediction(const QString& name);
    void programFpga();
    void programmingFinished(bool ok, const QString& message);
    void setStatus(const QString& text, bool ok);

    QLabel* title_;
    Canvas* canvas_;
    InputPreview* preview_;
    QLabel* digit_;
    QGridLayout* bars_;
    std::vector<QLabel*> score_labels_;
    std::vector<QProgressBar*> scores_;
    TargetBox* target_box_;
    QComboBox* array_box_;
    QLabel* status_icon_;
    QLabel* status_;
    FpgaProgrammer* programmer_;
    QProgressDialog* progress_ = nullptr;

    QString model_path_;
    QString target_name_;
    std::vector<std::int8_t> last_input_;  // classified again after a reconnect
    // Mlp refers to both the model and the driver: it is destroyed first.
    std::unique_ptr<tpu_host::digits::Model> model_;
    std::unique_ptr<tpu_host::TpuDriver> tpu_;
    std::unique_ptr<tpu_host::digits::Mlp> mlp_;
};
