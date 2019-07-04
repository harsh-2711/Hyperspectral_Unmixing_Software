#include "consolelogdialog.h"
#include "ui_consolelogdialog.h"

#define DEBUG_MODE 1

ConsoleLogDialog::ConsoleLogDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConsoleLogDialog)
{
    ui->setupUi(this);


    QPalette p = ui->textEdit->palette();
    p.setColor(QPalette::Base, Qt::black);
    ui->textEdit->setPalette(p);
    ui->textEdit->setTextColor(Qt::green);

    this->setWindowTitle(tr("Console"));

}

void ConsoleLogDialog::loadLog()
{
    ui->textEdit->clear();

    QString address;
    if(DEBUG_MODE == 0) address = "/usr/share/hypermix-2.0/info";
    else address = "info";
     char line[MAXLINE] = "";

    FILE* fp;
    if((fp = fopen(address.append("/console.log").toStdString().c_str(), "rt")) != NULL)
    {
        fseek(fp, 0L, SEEK_SET);
        while(fgets(line, MAXLINE, fp) != '\0')
            ui->textEdit->insertPlainText(tr(line));

        fclose(fp);
    }
}

void ConsoleLogDialog::write(QString str)
{
    ui->textEdit->insertPlainText(str);
}

ConsoleLogDialog::~ConsoleLogDialog()
{
    delete ui;
}
