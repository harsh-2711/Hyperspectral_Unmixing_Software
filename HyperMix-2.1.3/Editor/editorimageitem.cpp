#include "editorimageitem.h"

EditorImageItem::EditorImageItem(QMenu *contextMenu, QGraphicsItem *parent, QGraphicsScene *scene)
    : QGraphicsPixmapItem(parent, scene)
{

    myContextMenu = contextMenu;

    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    validate = false;
    numValidates = 0;

}

void EditorImageItem::removeArrow(EditorArrow *arrow)
{
    int index = arrows.indexOf(arrow);

    if (index != -1)
            arrows.removeAt(index);
}

void EditorImageItem::removeArrows()
{
    foreach (EditorArrow *arrow, arrows)
    {
        arrow->getStartImage()->removeArrow(arrow);
        arrow->getEndItem()->removeArrow(arrow);
        scene()->removeItem(arrow);
        delete arrow;
    }
}

void EditorImageItem::addArrow(EditorArrow *arrow)
{
    arrows.append(arrow);
}

QList<EditorArrow*> EditorImageItem::getArrows()
{
    return arrows;
}

QList<EditorArrow*> EditorImageItem::getArrowsEndsInItem()
{

    QList<EditorArrow*> aux;
    //An image never can be an end item
    return aux;
}

QList<EditorArrow*> EditorImageItem::getArrowsStartsInItem()
{

    QList<EditorArrow*> aux;
    foreach(EditorArrow* arrow, arrows)
        if(this == arrow->getStartImage())
            aux.append(arrow);
    return aux;
}

QList<EditorArrow*> EditorImageItem::getArrowsEndsInItemNotChecked()
{

    QList<EditorArrow*> aux;
    //An image never can be an end item
    return aux;
}

QList<EditorArrow*> EditorImageItem::getArrowsStartsInItemNotChecked()
{

    QList<EditorArrow*> aux;
    foreach(EditorArrow* arrow, arrows)
         if(this == arrow->getStartImage() && !arrow->isValidate())
            aux.append(arrow);
    return aux;
}

void EditorImageItem::setValidateAllArrows(bool val)
{
    foreach(EditorArrow* arrow, arrows)
        arrow->setValidate(val);
}

bool EditorImageItem::isValidate()
{
    return validate;
}

void EditorImageItem::setValidate(bool val)
{
    validate = val;
    if(validate) numValidates++;
}

void EditorImageItem::setOperatorName(QString name)
{
    operatorName = name;
}

QString EditorImageItem::getOperatorName()
{
    return operatorName;
}

void EditorImageItem::setOperatorFilename(QString file)
{
    filename = file;
}

QString EditorImageItem::getOperatorFilename()
{
    return filename;
}

void EditorImageItem::setOperatorNumber(unsigned int n)
{
    operatorNumber = n;
}

unsigned int EditorImageItem::getOperatorNumber()
{
    return operatorNumber;
}
