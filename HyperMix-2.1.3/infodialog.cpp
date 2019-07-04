#include "infodialog.h"
#include "ui_infodialog.h"

#define DEBUG_MODE 1

InfoDialog::InfoDialog(QString _operatorName,  QWidget *parent) :
    QDialog(parent),
    ui(new Ui::InfoDialog)
{
    ui->setupUi(this);

    operatorName = _operatorName;
    QPalette p = ui->textEdit->palette();
    p.setColor(QPalette::Base, Qt::white);
    ui->textEdit->setPalette(p);
    ui->textEdit->setTextColor(Qt::blue);
}

void InfoDialog::loadInfo()
{
        operatorName.remove(".exe");
        operatorName.append(".info");

        this->setWindowTitle(operatorName);

        QString address;

        if(DEBUG_MODE == 0) address = "/usr/share/hypermix-2.0/info";
        else address = "info";
        QDir dir(address);
        QStringList list = dir.entryList();
        char line[MAXLINE] = "";

        foreach(QString file, list)
        {
            if(QString::compare(file,operatorName) == 0)
            {
                ui->textEdit->clear();
                FILE* fp;
                if((fp = fopen(address.append("/").append(operatorName).toStdString().c_str(), "rt")) != NULL)
                {
                    fseek(fp, 0L, SEEK_SET);
                    while(fgets(line, MAXLINE, fp) != '\0')
                        ui->textEdit->insertPlainText(tr(line));

                    fclose(fp);
                }
            }
        }
}

InfoDialog::~InfoDialog()
{
    delete ui;
}
