#include "parameterdialog.h"
#include "ui_parameterdialog.h"

#define MAXPARAM 25

parameterDialog::parameterDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::parameterDialog)
{
    ui->setupUi(this);

    QComboBox *combo;
    QToolButton *tool;

    posOcup = 0;
    ui->tableWidget->setColumnWidth(0,500);
    ui->tableWidget->setColumnWidth(1,58);
    ui->tableWidget->setRowCount(MAXPARAM);
    ui->tableWidget->setSortingEnabled(false);

    for(int i=0; i<MAXPARAM; i++)
    {
        combo = new QComboBox();
        tool = new QToolButton();

        combo->setEditable(true);
        combo->setInsertPolicy(QComboBox::InsertAlphabetically);
        tool->setText(tr("..."));

        ui->tableWidget->setCellWidget(i,0,combo);
        ui->tableWidget->setCellWidget(i,1,tool);
        ui->tableWidget->cellWidget(i,0)->setEnabled(false);
        ui->tableWidget->cellWidget(i,1)->setEnabled(false);

        QObject::connect(tool, SIGNAL(clicked()), this, SLOT(selectFile()));

        parameterList.append(combo);
        fileButtonList.append(tool);
    }

    QObject::connect(ui->addButton, SIGNAL(clicked()), this, SLOT(addRow()));
    QObject::connect(ui->subButton, SIGNAL(clicked()), this, SLOT(subRow()));

}

parameterDialog::~parameterDialog()
{
    delete ui;
}

void parameterDialog::addRow()
{
    if(posOcup < MAXPARAM)
    {
        ui->tableWidget->cellWidget(posOcup,0)->setEnabled(true);
        ui->tableWidget->cellWidget(posOcup,1)->setEnabled(true);
        updateOptions();
        posOcup++;
    }
}

void parameterDialog::subRow()
{
    if(posOcup > 0)
    {
        posOcup--;
        ui->tableWidget->cellWidget(posOcup,0)->setEnabled(false);
        ui->tableWidget->cellWidget(posOcup,1)->setEnabled(false);
    }
}

void parameterDialog::setItem(EditorItem *_item)
{
    item = _item;
}

void parameterDialog::updateOptions()
{
    options.clear();
    options.append("");

    foreach(EditorArrow* arrow, item->getArrowsEndsInItemWithImages())
    {
        if(arrow->getStartImage() != 0)
            options.append(arrow->getStartImage()->getOperatorFilename());
        else if(arrow->getStartItem() != 0)
           foreach(QString str, arrow->getStartItem()->getArguments())
                options.append(str);
    }

    foreach(QComboBox* combo, parameterList)
    {
        while(combo->count() > 0) combo->removeItem(0);
        foreach(QString str, options)
            combo->addItem(str);
    }
}

void parameterDialog::updateOptions(int pos)
{
    options.clear();
    options.append("");

    foreach(EditorArrow* arrow, item->getArrowsEndsInItemWithImages())
    {
        if(arrow->getStartImage() != 0)
            options.append(arrow->getStartImage()->getOperatorFilename());
        else if(arrow->getStartItem() != 0)
            foreach(QString str, arrow->getStartItem()->getArguments())
                options.append(str);
    }

    while(parameterList[pos]->count() > 0) parameterList[pos]->removeItem(0);
    foreach(QString str, options)
            parameterList[pos]->addItem(str);

}


void parameterDialog::selectFile()
{
    QString filename = QFileDialog::getSaveFileName(this, "Select File", workspace, "*");

    if(sender() != NULL)
    {
        int it = 0;
        while(sender() != fileButtonList[it])
            it++;
        updateOptions(it);
        parameterList[it]->addItem(filename);
        options.append(filename);
        parameterList[it]->setCurrentIndex(parameterList[it]->count()-1);
    }
}


void parameterDialog::setWorkspace(QString ws)
{
    workspace = ws;
}

QStringList parameterDialog::getParameters()
{
    QStringList list;

    for(int i=0; i<posOcup; i++)
        list.append(parameterList[i]->itemText(parameterList[i]->currentIndex()));

    return list;
}

bool parameterDialog::isOutputChecked()
{
    return ui->checkBox->isChecked();
}
