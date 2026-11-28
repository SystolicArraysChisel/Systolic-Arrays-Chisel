#include "demo_page.hpp"

#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLocale>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
#include <QRandomGenerator>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>
#include <chrono>
#include <exception>
#include <system_error>

#include "paths.hpp"
#include "programmer.hpp"
#include "tpu_host/config.hpp"
#include "tpu_host/fake_tpu.hpp"
#include "tpu_host/serial_link.hpp"
#include "widgets.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace tpu_host;

namespace {

// Long enough for a programmed board, short enough not to freeze the window.
constexpr auto TIMEOUT = std::chrono::milliseconds(1500);
constexpr int SCORE_STEPS = 1000;

QStringList serialPorts() {
    QStringList ports;
#ifdef _WIN32
    HKEY key;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DEVICEMAP\\SERIALCOMM", 0, KEY_READ, &key) ==
        ERROR_SUCCESS) {
        for (DWORD i = 0;; ++i) {
            wchar_t name[256], value[256];
            DWORD name_len = 256, value_bytes = sizeof(value), type = 0;
            if (RegEnumValueW(key, i, name, &name_len, nullptr, &type,
                              reinterpret_cast<BYTE*>(value), &value_bytes) != ERROR_SUCCESS) {
                break;
            }
            if (type == REG_SZ) {
                ports << QString::fromWCharArray(value);
            }
        }
        RegCloseKey(key);
    }
#else
    for (const QString& f :
         QDir("/dev").entryList({"ttyUSB*", "ttyACM*", "cu.usbserial*"}, QDir::System)) {
        ports << "/dev/" + f;
    }
#endif
    ports.sort();
    return ports;
}

}  // namespace

// ---- TargetBox --------------------------------------------------------------

TargetBox::TargetBox(QWidget* parent) : QComboBox(parent) { refresh(); }

void TargetBox::showPopup() {
    refresh();
    QComboBox::showPopup();
}

void TargetBox::refresh() {
    const QString current = currentText();
    {
        const QSignalBlocker block(this);
        clear();
        addItem(SIMULATOR);
        addItems(serialPorts());
        const int i = findText(current);
        setCurrentIndex(i >= 0 ? i : 0);
    }
    if (!current.isEmpty() && currentText() != current) {
        emit currentTextChanged(currentText());  // the port went away
    }
}

// ---- DemoPage ---------------------------------------------------------------

DemoPage::DemoPage(QWidget* parent) : QWidget(parent), programmer_(new FpgaProgrammer(this)) {
    // Header: back to the model list, model name.
    auto* back = new QToolButton;
    back->setIcon(style()->standardIcon(QStyle::SP_ArrowBack));
    back->setText(tr("Models"));
    back->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    back->setAutoRaise(true);
    title_ = new QLabel;
    QFont tf = title_->font();
    tf.setPointSizeF(tf.pointSizeF() * 1.6);
    title_->setFont(tf);
    auto* header = new QHBoxLayout;
    header->addWidget(back);
    header->addSpacing(8);
    header->addWidget(title_, 1);

    // Drawing.
    canvas_ = new Canvas;
    auto* clear = new QPushButton(tr("Clear"));
    auto* example = new QPushButton(tr("Example"));
    example->setToolTip(tr("Show a digit from the test set"));
    auto* draw_buttons = new QHBoxLayout;
    draw_buttons->addWidget(clear);
    draw_buttons->addWidget(example);
    draw_buttons->addStretch();
    auto* draw_box = new QGroupBox(tr("Drawing"));
    auto* draw_layout = new QVBoxLayout(draw_box);
    draw_layout->addWidget(canvas_);
    draw_layout->addLayout(draw_buttons);

    // Result.
    digit_ = new QLabel;
    QFont df = digit_->font();
    df.setPointSize(54);
    digit_->setFont(df);
    digit_->setAlignment(Qt::AlignCenter);
    digit_->setMinimumWidth(110);
    preview_ = new InputPreview;
    auto* answer = new QVBoxLayout;
    answer->addWidget(new QLabel(tr("Prediction")), 0, Qt::AlignHCenter);
    answer->addWidget(digit_, 1);
    answer->addWidget(new QLabel(tr("TPU input")), 0, Qt::AlignHCenter);
    answer->addWidget(preview_, 0, Qt::AlignHCenter);
    bars_ = new QGridLayout;
    bars_->setVerticalSpacing(6);
    bars_->setColumnStretch(1, 1);
    auto* result_box = new QGroupBox(tr("Result"));
    auto* result_layout = new QHBoxLayout(result_box);
    result_layout->setSpacing(16);
    result_layout->addLayout(answer);
    result_layout->addLayout(bars_, 1);

    auto* body = new QHBoxLayout;
    body->addWidget(draw_box);
    body->addWidget(result_box, 1);

    // Accelerator.
    target_box_ = new TargetBox;
    target_box_->setMinimumContentsLength(10);
    array_box_ = new QComboBox;
    for (int n : {2, 4, 8}) {
        array_box_->addItem(QString("%1x%1").arg(n), n);
    }
    array_box_->setCurrentIndex(1);
    auto* program = new QPushButton(tr("Program FPGA..."));
    status_icon_ = new QLabel;
    status_icon_->setPixmap(style()->standardIcon(QStyle::SP_MessageBoxWarning).pixmap(16, 16));
    status_ = new QLabel;
    status_->setWordWrap(true);
    status_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    auto* settings = new QHBoxLayout;
    settings->addWidget(new QLabel(tr("Run on:")));
    settings->addWidget(target_box_);
    settings->addSpacing(12);
    settings->addWidget(new QLabel(tr("Array:")));
    settings->addWidget(array_box_);
    settings->addStretch();
    settings->addWidget(program);
    auto* status = new QHBoxLayout;
    status->addWidget(status_icon_);
    status->addWidget(status_, 1);
    auto* tpu_box = new QGroupBox(tr("Accelerator"));
    auto* tpu_layout = new QVBoxLayout(tpu_box);
    tpu_layout->addLayout(settings);
    tpu_layout->addLayout(status);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(16, 12, 16, 16);
    layout->setSpacing(10);
    layout->addLayout(header);
    layout->addLayout(body, 1);
    layout->addWidget(tpu_box);

    connect(back, &QToolButton::clicked, this, &DemoPage::backRequested);
    connect(canvas_, &Canvas::strokeFinished, this, [this] {
        if (!canvas_->isEmpty()) {
            classify(digits::features(canvas_->bitmap()));
        }
    });
    connect(clear, &QPushButton::clicked, this, [this] {
        canvas_->clear();
        last_input_.clear();
        clearResult();
    });
    connect(example, &QPushButton::clicked, this, &DemoPage::showExample);
    connect(program, &QPushButton::clicked, this, &DemoPage::programFpga);
    connect(programmer_, &FpgaProgrammer::finished, this, &DemoPage::programmingFinished);
    for (QComboBox* box : {static_cast<QComboBox*>(target_box_), array_box_}) {
        connect(box, &QComboBox::currentTextChanged, this, &DemoPage::connectTarget);
    }
    clearResult();
}

DemoPage::~DemoPage() = default;

void DemoPage::setModel(const QString& path) {
    model_path_ = path;
    canvas_->clear();
    last_input_.clear();
    clearResult();
    connectTarget();
}

void DemoPage::disconnectTarget() {
    mlp_.reset();
    tpu_.reset();
}

void DemoPage::connectTarget() {
    disconnectTarget();
    if (model_path_.isEmpty()) {
        return;
    }
    const QString target = target_box_->currentText();
    const QString array = array_box_->currentText();
    target_name_ = tr("%1, %2 array").arg(target, array);

    QApplication::setOverrideCursor(Qt::WaitCursor);
    try {
        model_ = std::make_unique<digits::Model>(digits::Model::load(model_path_.toStdString()));
        title_->setText(QString::fromStdString(model_->name));
        buildScores();
        HwConfig cfg;
        cfg.rows = cfg.cols = array_box_->currentData().toInt();
        cfg.name = array.toStdString();
        std::unique_ptr<Link> link;
        if (target == TargetBox::SIMULATOR) {
            link = std::make_unique<FakeTpu>(cfg);
        } else {
            link = std::make_unique<SerialLink>(target.toStdString(), cfg.baud);
        }
        tpu_ = std::make_unique<TpuDriver>(std::move(link), cfg, TIMEOUT);
        tpu_->status();  // clears stale flags; times out if the board is not there
        mlp_ = std::make_unique<digits::Mlp>(*tpu_, *model_, 1);
        setStatus(tr("Connected: %1").arg(target_name_), true);
    } catch (const std::system_error&) {
        disconnectTarget();
        setStatus(tr("Cannot open %1. Check that the board is plugged in and that no other "
                     "program uses the port.")
                      .arg(target),
                  false);
    } catch (const TpuError&) {
        disconnectTarget();
        setStatus(tr("The board on %1 does not answer. Program it with the %2 bitstream.")
                      .arg(target, array),
                  false);
    } catch (const std::exception& e) {
        disconnectTarget();
        setStatus(QString::fromStdString(e.what()), false);
    }
    QApplication::restoreOverrideCursor();

    if (!mlp_) {
        clearResult();
    } else if (!last_input_.empty()) {
        classify(last_input_);
    }
}

void DemoPage::showExample() {
    if (!model_ || model_->x.rows() == 0) {
        return;
    }
    const auto i = static_cast<std::ptrdiff_t>(
        QRandomGenerator::global()->bounded(static_cast<int>(model_->x.rows())));
    const auto first = model_->x.data().begin() + i * digits::PIXELS;
    const std::vector<std::int8_t> x(first, first + digits::PIXELS);
    canvas_->showInput(x);
    classify(x);
}

void DemoPage::classify(const std::vector<std::int8_t>& x) {
    last_input_ = x;
    if (!mlp_) {
        return;  // the status line already says why
    }
    try {
        const digits::Mlp::Result r = mlp_->run(Matrix<std::int8_t>(1, digits::PIXELS, x));
        const std::size_t n = model_->classes.size();
        const auto best = static_cast<std::size_t>(digits::predict(r.logits, 0, n));
        const auto& logits = r.logits.data();
        const auto [lo_it, hi_it] =
            std::minmax_element(logits.begin(), logits.begin() + static_cast<std::ptrdiff_t>(n));
        const double lo = double(*lo_it), span = std::max(1.0, double(*hi_it) - lo);
        for (std::size_t c = 0; c < n; ++c) {
            scores_[c]->setValue(int(SCORE_STEPS * (double(logits[c]) - lo) / span));
            QFont f = score_labels_[c]->font();
            f.setBold(c == best);
            score_labels_[c]->setFont(f);
        }
        showPrediction(QString::fromStdString(model_->classes[best]));
        preview_->setInput(x);
        setStatus(tr("Connected: %1. Last drawing took %2 cycles.")
                      .arg(target_name_, QLocale().toString(r.layer1.cycles + r.layer2.cycles)),
                  true);
    } catch (const std::exception&) {
        disconnectTarget();
        clearResult();
        setStatus(tr("Lost the connection to %1.").arg(target_box_->currentText()), false);
    }
}

void DemoPage::showPrediction(const QString& name) {
    // A digit in large type, a word smaller so that it fits.
    QFont f = digit_->font();
    f.setPointSize(name.size() <= 2 ? 54 : 24);
    digit_->setFont(f);
    digit_->setText(name);
}

void DemoPage::buildScores() {
    QStringList names;
    for (const std::string& c : model_->classes) {
        names << QString::fromStdString(c);
    }
    QStringList shown;
    for (const QLabel* l : score_labels_) {
        shown << l->text();
    }
    if (names == shown) {
        return;
    }
    for (QWidget* w : score_labels_) {
        delete w;
    }
    for (QWidget* w : scores_) {
        delete w;
    }
    score_labels_.clear();
    scores_.clear();
    for (int c = 0; c < names.size(); ++c) {
        auto* label = new QLabel(names[c]);
        auto* bar = new QProgressBar;
        bar->setRange(0, SCORE_STEPS);
        bar->setTextVisible(false);
        bar->setMaximumHeight(14);
        bars_->addWidget(label, c, 0);
        bars_->addWidget(bar, c, 1);
        score_labels_.push_back(label);
        scores_.push_back(bar);
    }
}

void DemoPage::clearResult() {
    showPrediction(QStringLiteral("–"));
    preview_->setInput({});
    for (std::size_t c = 0; c < scores_.size(); ++c) {
        scores_[c]->setValue(0);
        QFont f = score_labels_[c]->font();
        f.setBold(false);
        score_labels_[c]->setFont(f);
    }
}

void DemoPage::programFpga() {
    // The file dialog opens on the bitstream of the selected array size.
    const QString suggested = bitstreamDir() + "/" + array_box_->currentText() + "/BoardTop.bit";
    const QString start = QFileInfo::exists(suggested) ? suggested : bitstreamDir();
    const QString bit =
        QFileDialog::getOpenFileName(this, tr("Program FPGA"), start, tr("Bitstreams (*.bit)"));
    if (bit.isEmpty()) {
        return;
    }
    // A bitstream built by fpga/build.tcl has its config.json beside it: match the array size.
    const QString config = QFileInfo(bit).absolutePath() + "/config.json";
    if (QFileInfo::exists(config)) {
        try {
            const int i = array_box_->findText(
                QString::fromStdString(HwConfig::load(config.toStdString()).name));
            if (i >= 0) {
                const QSignalBlocker block(array_box_);
                array_box_->setCurrentIndex(i);
            }
        } catch (const std::exception&) {
            // not a config we know: keep the selection
        }
    }

    disconnectTarget();
    QString error;
    if (!programmer_->start(bit, &error)) {
        QMessageBox::warning(this, tr("Program FPGA"), error);
        connectTarget();
        return;
    }
    progress_ = new QProgressDialog(tr("Programming the FPGA..."), tr("Cancel"), 0, 0, this);
    progress_->setWindowTitle(tr("Program FPGA"));
    progress_->setWindowModality(Qt::WindowModal);
    progress_->setMinimumDuration(0);
    connect(progress_, &QProgressDialog::canceled, programmer_, &FpgaProgrammer::cancel);
    progress_->show();
}

void DemoPage::programmingFinished(bool ok, const QString& message) {
    if (progress_ != nullptr) {
        progress_->disconnect(programmer_);  // closing it must not cancel anything
        progress_->deleteLater();
        progress_ = nullptr;
    }
    if (!ok) {
        QMessageBox::warning(this, tr("Program FPGA"), message);
    }
    connectTarget();
    if (ok && mlp_) {
        setStatus(tr("FPGA programmed. ") + status_->text(), true);
    } else if (ok) {
        setStatus(tr("FPGA programmed. Select its serial port in \"Run on\"."), true);
    }
}

void DemoPage::setStatus(const QString& text, bool ok) {
    status_->setText(text);
    status_icon_->setVisible(!ok);
}
