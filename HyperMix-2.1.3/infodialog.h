#ifndef INFODIALOG_H
#define INFODIALOG_H

#include <QDialog>
#include <QDir>

#define MAXLINE 200

namespace Ui {
class InfoDialog;
}

class InfoDialog : public QDialog
{
    Q_OBJECT

    QString operatorName;

public:
    explicit InfoDialog(QString _operatorName, QWidget *parent = 0);
    ~InfoDialog();
    void loadInfo();

private:
    Ui::InfoDialog *ui;
};

#endif // INFODIALOG_H
