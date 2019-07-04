#include "helpdialog.h"
#include "ui_helpdialog.h"

#define DEBUG_MODE 1

HelpDialog::HelpDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::HelpDialog)
{
    ui->setupUi(this);
}

void HelpDialog::start()
{
    this->show();

    ui->textEdit->clear();
    ui->textEdit->setFont(QFont("OldEnglish", 12));
    ui->textEdit->setTextColor(Qt::blue);

    if(DEBUG_MODE == 0) ui->imageLabel->setPixmap(QPixmap("/usr/share/hypermix-2.0/images/logogrande.jpg"));
    else ui->imageLabel->setPixmap(QPixmap("images/logogrande.jpg"));

    FILE* fp;
    char line[MAXLINE];

    QString address;
    if(DEBUG_MODE == 0) address = "/usr/share/hypermix-2.0/info/Hypermix.info";
    else address = "info/Hypermix.info";

    if((fp = fopen(address.toStdString().c_str(), "rt")) != NULL)
    {
        fseek(fp, 0L, SEEK_SET);
        while(fgets(line, MAXLINE, fp) != '\0')
            ui->textEdit->insertPlainText(tr(line));
        fclose(fp);
    }
}

HelpDialog::~HelpDialog()
{
    delete ui;
}
