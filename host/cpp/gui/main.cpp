// TPU demo: choose a model, draw a digit, the network runs on the FPGA (or its
// simulator) and shows the class it recognises.
#include <QApplication>

#include "main_window.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    MainWindow w;
    w.resize(760, 560);
    w.show();
    return QApplication::exec();
}
