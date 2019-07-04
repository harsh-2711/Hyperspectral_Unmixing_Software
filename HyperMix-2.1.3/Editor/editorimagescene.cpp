#include "editorimagescene.h"

EditorImageScene::EditorImageScene(QObject *parent) :
    QGraphicsScene(parent)
{
}

void EditorImageScene::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if(mouseEvent->button() == Qt::LeftButton)
    {
        QPointF mousePoint = mouseEvent->scenePos();
        if(mousePoint.x() >= 0 && mousePoint.y() >= 0 && mousePoint.x() < width && mousePoint.y() < height)
            emit zPoint(mousePoint);
    }
}

void EditorImageScene::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    QPointF mousePoint = mouseEvent->scenePos();
    if(mousePoint.x() >= 0 && mousePoint.y() >= 0 && mousePoint.x() < width && mousePoint.y() < height)
        emit zPoint(mousePoint);
}

void EditorImageScene::setWidth(int w)
{
    width = w;
}

void EditorImageScene::setHeight(int h)
{
    height = h;
}
