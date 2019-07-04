#ifndef EDITORIMAGESCENE_H
#define EDITORIMAGESCENE_H

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QPointF>

class EditorImageScene : public QGraphicsScene
{
    Q_OBJECT

    int width;
    int height;

public:
    explicit EditorImageScene(QObject *parent = 0);
    void setWidth(int w);
    void setHeight(int h);
signals:
    void zPoint(QPointF pos);

public slots:

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent);

};

#endif // EDITORIMAGESCENE_H
