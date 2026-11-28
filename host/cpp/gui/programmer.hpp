// Loads a bitstream into the FPGA with Vivado (fpga/program.tcl), in the background.
#pragma once
#include <QObject>
#include <QProcess>
#include <QString>

class FpgaProgrammer : public QObject {
    Q_OBJECT
   public:
    explicit FpgaProgrammer(QObject* parent = nullptr);

    // Starts Vivado on `bitstream`. Returns false, with the reason in *error,
    // if it cannot start (Vivado or program.tcl not found).
    bool start(const QString& bitstream, QString* error);
    void cancel();

   signals:
    // message: one sentence for the user.
    void finished(bool ok, const QString& message);

   private:
    void done(int exitCode, QProcess::ExitStatus status);
    QProcess* process_;
    bool cancelled_ = false;
};
