#ifndef SCHEME_H
#define SCHEME_H

#include <QObject>
#include <QList>
#include <QString>
#include <QTextEdit>
#include <QProcess>
#include <QTime>
 #include <QMessageBox>
#include <QProgressBar>

#include "editoritem.h"
#include "consolelogdialog.h"

#define MAXCAD 120
#define NUMBERTYPESIT 5

class Scheme : public QObject
{
    Q_OBJECT

    QList<EditorImageItem*> activeImages;
    QList<EditorItem*> activeItems;
    QString workspace;
    QList< QProcess* > processList;
    int logNumber;
    QProgressBar *progress;

public:
    explicit Scheme( QProgressBar* _progress, QObject *parent = 0);
    void addItem(EditorItem* it);
    void addItem(EditorImageItem* it);
    void removeItem(EditorItem* it);
    void removeImage(EditorImageItem* it);
    int compileScheme(ConsoleLogDialog* consoleD);
    int runScheme(int numberExecutions,ConsoleLogDialog* consoleD);
    void setWorkspace(QString ws);
    void cleanScheme();
    QStringList getListActiveImages();

signals:
    void newImageResult(QString);
    void newEndmemberResult(QString);
    void newAbundanceResult(QString);


private:
    int checkLogicalOrder(ConsoleLogDialog *consoleD);
    
};

#endif // SCHEME_H
