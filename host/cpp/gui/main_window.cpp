#include "main_window.hpp"

#include <QStackedWidget>

#include "demo_page.hpp"
#include "model_page.hpp"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), pages_(new QStackedWidget), models_(new ModelPage), demo_(new DemoPage) {
    setWindowTitle(tr("TPU App"));
    pages_->addWidget(models_);
    pages_->addWidget(demo_);
    setCentralWidget(pages_);

    connect(models_, &ModelPage::modelChosen, this, [this](const QString& path) {
        demo_->setModel(path);
        pages_->setCurrentWidget(demo_);
    });
    connect(demo_, &DemoPage::backRequested, this, [this] {
        models_->reload();
        pages_->setCurrentWidget(models_);
    });
}
