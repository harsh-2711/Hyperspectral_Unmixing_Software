#ifndef EDITORRMSESCENE_H
#define EDITORRMSESCENE_H

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QPointF>
#include <QDebug>

class EditorRMSEScene : public QGraphicsScene
{
    Q_OBJECT

    int width;
    int height;
public:
    explicit EditorRMSEScene(QObject *parent = 0);
    void setWidth(int w);
    void setHeigt(int h);
signals:
    void percentRMSE(QPointF pos);

public slots:

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent);

};

#endif // EDITORRMSESCENE_H
