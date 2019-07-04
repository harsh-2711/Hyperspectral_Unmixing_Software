#ifndef EDITORIMAGEITEM_H
#define EDITORIMAGEITEM_H

#include <QGraphicsPixmapItem>
#include "Editor/editorarrow.h"

class EditorArrow;

class EditorImageItem : public QGraphicsPixmapItem
{

    QMenu *myContextMenu;
    QList<EditorArrow*> arrows;
    bool validate;
    int numValidates;
    QString operatorName;
    QString filename;
    unsigned int operatorNumber;

public:
    EditorImageItem(QMenu *contextMenu, QGraphicsItem *parent = 0, QGraphicsScene *scene = 0);
    void removeArrow(EditorArrow *arrow);
    void removeArrows();
    void addArrow(EditorArrow *arrow);
    QList<EditorArrow*> getArrows();
    QList<EditorArrow*> getArrowsStartsInItem();
    QList<EditorArrow*> getArrowsEndsInItem();
    QList<EditorArrow*> getArrowsStartsInItemNotChecked();
    QList<EditorArrow*> getArrowsEndsInItemNotChecked();
    void setValidateAllArrows(bool val);
    bool isValidate();
    void setValidate(bool val);
    void setOperatorName(QString name);
    QString getOperatorName();
    void setOperatorFilename(QString file);
    QString getOperatorFilename();
    void setOperatorNumber(unsigned int n);
    unsigned int getOperatorNumber();
};

#endif // EDITORIMAGEITEM_H
