#include "editorabundancescene.h"

EditorAbundanceScene::EditorAbundanceScene(QObject *parent) :
    QGraphicsScene(parent)
{
}


void EditorAbundanceScene::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if(mouseEvent->button() == Qt::LeftButton)
    {

        QPointF mousePoint = mouseEvent->scenePos();
        if(mousePoint.x() >= 0 && mousePoint.y() >= 0 && mousePoint.x() < width && mousePoint.y() < height)
            emit percentAbun(mousePoint);
    }
}

void EditorAbundanceScene::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    QPointF mousePoint = mouseEvent->scenePos();
    if(mousePoint.x() >= 0 && mousePoint.y() >= 0 && mousePoint.x() < width && mousePoint.y() < height)
        emit percentAbun(mousePoint);
}


void EditorAbundanceScene::setWidth(int w)
{
    width = w;
}

void EditorAbundanceScene::setHeigt(int h)
{
    height = h;
}
