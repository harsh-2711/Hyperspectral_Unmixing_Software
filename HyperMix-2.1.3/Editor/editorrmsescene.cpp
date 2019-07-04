#include "editorrmsescene.h"

EditorRMSEScene::EditorRMSEScene(QObject *parent) :
    QGraphicsScene(parent)
{
}


void EditorRMSEScene::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if(mouseEvent->button() == Qt::LeftButton)
    {

        QPointF mousePoint = mouseEvent->scenePos();
        if(mousePoint.x() >= 0 && mousePoint.y() >= 0 && mousePoint.x() < width && mousePoint.y() < height)
            emit percentRMSE(mousePoint);
    }
}

void EditorRMSEScene::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    QPointF mousePoint = mouseEvent->scenePos();
    if(mousePoint.x() >= 0 && mousePoint.y() >= 0 && mousePoint.x() < width && mousePoint.y() < height)
        emit percentRMSE(mousePoint);
}


void EditorRMSEScene::setWidth(int w)
{
    width = w;
}

void EditorRMSEScene::setHeigt(int h)
{
    height = h;
}
