// tpu_app, the user application: choose a model, draw, and the network runs on
// the FPGA (or its simulator) and shows the class it recognises.
#include <QApplication>
#include <QIcon>

#include "main_window.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setWindowIcon(QIcon(":/tpu_app.png"));
    MainWindow w;
    w.resize(760, 560);
    w.show();
    return QApplication::exec();
}
