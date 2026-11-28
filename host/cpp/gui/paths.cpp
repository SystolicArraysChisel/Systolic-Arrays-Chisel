#include "paths.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>

namespace {

// Where the app's own files are: inside the AppImage on Linux.
QString appDir() { return QCoreApplication::applicationDirPath(); }

// The folder the user sees the app in: for an AppImage, the folder of the
// .AppImage file (its executable runs from a temporary mount).
QString userDir() {
    const QString appimage = qEnvironmentVariable("APPIMAGE");
    return appimage.isEmpty() ? appDir() : QFileInfo(appimage).absolutePath();
}

}  // namespace

QString demoDir() {
    const QString shipped = appDir() + "/demo";
    return QDir(shipped).exists() ? shipped : QStringLiteral(DEMO_DIR);
}

QString programScript() {
    const QString shipped = appDir() + "/fpga/program.tcl";
    return QFileInfo::exists(shipped) ? shipped : QStringLiteral(REPO_DIR "/fpga/program.tcl");
}

QString bitstreamDir() {
    const QString shipped = userDir() + "/bitstreams";
    return QDir(shipped).exists() ? shipped : QStringLiteral(REPO_DIR "/build");
}
