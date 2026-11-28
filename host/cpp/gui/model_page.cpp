#include "model_page.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QStyle>
#include <QVBoxLayout>
#include <exception>

#include "paths.hpp"
#include "tpu_host/digits.hpp"

namespace digits = tpu_host::digits;

ModelPage::ModelPage(QWidget* parent) : QWidget(parent) {
    auto* title = new QLabel(tr("Choose a model"));
    QFont f = title->font();
    f.setPointSizeF(f.pointSizeF() * 1.6);
    title->setFont(f);

    list_ = new QListWidget;
    list_->setIconSize(QSize(48, 48));
    list_->setSpacing(4);
    QFont lf = list_->font();
    lf.setPointSizeF(lf.pointSizeF() * 1.2);
    list_->setFont(lf);

    open_ = new QPushButton(tr("Open"));
    open_->setDefault(true);
    auto* buttons = new QHBoxLayout;
    buttons->addStretch();
    buttons->addWidget(open_);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(24, 20, 24, 20);
    layout->setSpacing(12);
    layout->addWidget(title);
    layout->addWidget(list_, 1);
    layout->addLayout(buttons);

    connect(open_, &QPushButton::clicked, this, &ModelPage::openSelected);
    connect(list_, &QListWidget::itemActivated, this, &ModelPage::openSelected);
    connect(list_, &QListWidget::currentItemChanged, this, [this](QListWidgetItem* item) {
        open_->setEnabled(item != nullptr && (item->flags() & Qt::ItemIsEnabled));
    });
    reload();
}

void ModelPage::reload() {
    list_->clear();
    const QIcon generic = style()->standardIcon(QStyle::SP_FileIcon);
    for (const QFileInfo& f : QDir(demoDir()).entryInfoList({"*.json"}, QDir::Files, QDir::Name)) {
        // The logo is the PNG with the model's name: demo/digits_model.png.
        const QString logo = f.absolutePath() + "/" + f.completeBaseName() + ".png";
        auto* item =
            new QListWidgetItem(QFileInfo::exists(logo) ? QIcon(logo) : generic, {}, list_);
        item->setData(Qt::UserRole, f.absoluteFilePath());
        try {
            const digits::Model m = digits::Model::load(f.absoluteFilePath().toStdString());
            item->setText(QString::fromStdString(m.name));
        } catch (const std::exception&) {
            item->setText(tr("%1 (cannot be read)").arg(f.fileName()));
            item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
        }
    }
    if (list_->count() == 0) {
        auto* item = new QListWidgetItem(
            tr("No model found in %1").arg(QDir::toNativeSeparators(demoDir())), list_);
        item->setFlags(Qt::NoItemFlags);
    }
    list_->setCurrentRow(0);
    open_->setEnabled(list_->currentItem() != nullptr &&
                      (list_->currentItem()->flags() & Qt::ItemIsEnabled));
}

void ModelPage::openSelected() {
    const QListWidgetItem* item = list_->currentItem();
    if (item != nullptr && (item->flags() & Qt::ItemIsEnabled)) {
        emit modelChosen(item->data(Qt::UserRole).toString());
    }
}
