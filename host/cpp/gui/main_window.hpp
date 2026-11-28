// The demo window: the model list, then the drawing screen.
#pragma once
#include <QMainWindow>

class QStackedWidget;
class ModelPage;
class DemoPage;

class MainWindow : public QMainWindow {
    Q_OBJECT
   public:
    explicit MainWindow(QWidget* parent = nullptr);

   private:
    QStackedWidget* pages_;
    ModelPage* models_;
    DemoPage* demo_;
};
