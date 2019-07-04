#ifndef CONSOLELOGDIALOG_H
#define CONSOLELOGDIALOG_H

#include <QDialog>
#include "infodialog.h"

namespace Ui {
class ConsoleLogDialog;
}

class ConsoleLogDialog : public QDialog
{
    Q_OBJECT


public:
    explicit ConsoleLogDialog(QWidget *parent = 0);
    ~ConsoleLogDialog();
    void loadLog();
    void write(QString str);

private:
    Ui::ConsoleLogDialog *ui;
};

#endif // CONSOLELOGDIALOG_H
