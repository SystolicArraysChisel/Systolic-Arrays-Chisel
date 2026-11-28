// First screen: the models found in demo/, one is chosen to open the demo.
#pragma once
#include <QListWidget>
#include <QPushButton>
#include <QString>
#include <QWidget>

class ModelPage : public QWidget {
    Q_OBJECT
   public:
    explicit ModelPage(QWidget* parent = nullptr);
    void reload();  // lists the models again

   signals:
    void modelChosen(const QString& path);

   private:
    void openSelected();
    QListWidget* list_;
    QPushButton* open_;
};
