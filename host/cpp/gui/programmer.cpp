#include "programmer.hpp"

#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include <QStandardPaths>
#include <algorithm>

#include "paths.hpp"

namespace {

// vivado on PATH, else the newest one in the default install folders
// (<root>/<version>/Vivado/bin or <root>/<version>/bin).
QString findVivado() {
    QString found = QStandardPaths::findExecutable("vivado");
    if (!found.isEmpty()) {
        return found;
    }
#ifdef _WIN32
    const QStringList roots = {"C:/AMDDesignTools", "C:/Xilinx/Vivado", "C:/Xilinx"};
    const QString exe = "vivado.bat";
#else
    const QStringList roots = {"/tools/Xilinx", "/tools/Xilinx/Vivado", "/opt/Xilinx",
                               "/opt/Xilinx/Vivado", QDir::homePath() + "/Xilinx"};
    const QString exe = "vivado";
#endif
    for (const QString& root : roots) {
        QStringList versions = QDir(root).entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        std::reverse(versions.begin(), versions.end());
        for (const QString& v : versions) {
            for (const QString& bin : {"/Vivado/bin/", "/bin/"}) {
                const QString path = root + "/" + v + bin + exe;
                if (QFileInfo::exists(path)) {
                    return path;
                }
            }
        }
    }
    return {};
}

// What went wrong, from Vivado's output, in one sentence.
QString failureMessage(const QString& out, int exitCode) {
    if (out.contains("no xc7a100t") || out.contains("hw_target") ||
        out.contains("No hardware target") || out.contains("hw_server")) {
        return QObject::tr("No FPGA found. Is the board switched on and connected by USB?");
    }
    if (out.contains(" not found")) {
        return QObject::tr("The bitstream file was not found.");
    }
    static const QRegularExpression error(R"(^ERROR: (?:\[[^\]]*\] )?(.*)$)",
                                          QRegularExpression::MultilineOption);
    const QRegularExpressionMatch m = error.match(out);
    if (m.hasMatch()) {
        return m.captured(1).trimmed();
    }
    return QObject::tr("Vivado stopped with exit code %1.").arg(exitCode);
}

}  // namespace

FpgaProgrammer::FpgaProgrammer(QObject* parent) : QObject(parent), process_(new QProcess(this)) {
    process_->setProcessChannelMode(QProcess::MergedChannels);
    process_->setWorkingDirectory(QDir::tempPath());  // Vivado leaves files where it runs
    connect(process_, &QProcess::finished, this, &FpgaProgrammer::done);
}

bool FpgaProgrammer::start(const QString& bitstream, QString* error) {
    const QString vivado = findVivado();
    if (vivado.isEmpty()) {
        *error = tr("Vivado was not found. Add its bin folder to PATH.");
        return false;
    }
    const QString script = programScript();
    if (!QFileInfo::exists(script)) {
        *error = tr("fpga/program.tcl was not found.");
        return false;
    }
    QStringList args = {"-mode",   "batch", "-nojournal", "-nolog", "-notrace",
                        "-source", script,  "-tclargs",   bitstream};
    cancelled_ = false;
    if (vivado.endsWith(".bat", Qt::CaseInsensitive)) {
        args.prepend(QDir::toNativeSeparators(vivado));
        args.prepend("/c");
        process_->start("cmd.exe", args);
    } else {
        process_->start(vivado, args);
    }
    if (!process_->waitForStarted(5000)) {
        *error = tr("Vivado could not be started.");
        return false;
    }
    return true;
}

void FpgaProgrammer::cancel() {
    if (process_->state() != QProcess::NotRunning) {
        cancelled_ = true;
        process_->kill();
    }
}

void FpgaProgrammer::done(int exitCode, QProcess::ExitStatus status) {
    const QString out = QString::fromLocal8Bit(process_->readAll());
    if (cancelled_) {
        emit finished(false, tr("Programming cancelled."));
    } else if (status == QProcess::NormalExit && exitCode == 0 && out.contains("programmed ")) {
        emit finished(true, tr("FPGA programmed."));
    } else {
        emit finished(false, failureMessage(out, exitCode));
    }
}
