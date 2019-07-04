#ifndef PARAMETERDIALOG_H
#define PARAMETERDIALOG_H

#include <QDialog>
#include <QComboBox>
#include <QToolButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include "editoritem.h"

namespace Ui {
class parameterDialog;
}
class EditorItem;

class parameterDialog : public QDialog
{
    Q_OBJECT

    QList< QComboBox* > parameterList;
    QList< QToolButton *> fileButtonList;
    int posOcup;
    EditorItem *item;
    QString workspace;
    QStringList options;

public:
    explicit parameterDialog(QWidget *parent=0);
    void setItem(EditorItem *_item);
    void setWorkspace(QString ws);
    void updateOptions();
    void updateOptions(int pos);
    bool isOutputChecked();
    QStringList getParameters();
    ~parameterDialog();

public slots:
    void addRow();
    void subRow();
    void selectFile();
    
private:
    Ui::parameterDialog *ui;
};

#endif // PARAMETERDIALOG_H
