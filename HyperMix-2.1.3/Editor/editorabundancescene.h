#ifndef EDITORABUNDANCESCENE_H
#define EDITORABUNDANCESCENE_H

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QPointF>
#include <QDebug>

class EditorAbundanceScene : public QGraphicsScene
{
    Q_OBJECT

    int width;
    int height;
public:
    explicit EditorAbundanceScene(QObject *parent = 0);
    void setWidth(int w);
    void setHeigt(int h);
signals:
    void percentAbun(QPointF pos);

public slots:

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent);

};

#endif // EDITORABUNDANCESCENE_H
