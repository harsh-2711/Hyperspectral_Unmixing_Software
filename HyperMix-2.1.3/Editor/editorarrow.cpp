/****************************************************************************
**
** Copyright (C) 2012 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
**     the names of its contributors may be used to endorse or promote
**     products derived from this software without specific prior written
**     permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
** $QT_END_LICENSE$
**
****************************************************************************/

#include "editorarrow.h"
#include <math.h>

const qreal Pi = 3.14;

EditorArrow::EditorArrow(QMenu *contextMenu, EditorItem *_startItem, EditorItem *_endItem, QGraphicsItem *parent, QGraphicsScene *scene)
                        : QGraphicsLineItem(parent, scene)
{
        myContextMenu = contextMenu;
        startImage = NULL;
        startItem = _startItem;
        endItem = _endItem;
        color = Qt::black;
        setFlag(QGraphicsItem::ItemIsSelectable, true);
        setPen(QPen(color, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        typeItem = -1;
        validate = false;

}

EditorArrow::EditorArrow(EditorImageItem *_startItem, EditorItem *_endItem, QGraphicsItem *parent, QGraphicsScene *scene)
                        : QGraphicsLineItem(parent, scene)
{


        startImage = _startItem;
        startItem = NULL;
        endItem = _endItem;
        color = Qt::black;
        setFlag(QGraphicsItem::ItemIsSelectable, true);
        setPen(QPen(color, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        typeItem = -1;
        validate = false;

}

QRectF EditorArrow::boundingRect() const
{
    qreal extra = (pen().width() + 20) / 2.0;

    return QRectF(line().p1(), QSizeF(line().p2().x() - line().p1().x(),
                                      line().p2().y() - line().p1().y()))
        .normalized()
        .adjusted(-extra, -extra, extra, extra);
}

QPainterPath EditorArrow::shape() const
{
    QPainterPath path = QGraphicsLineItem::shape();
    path.addPolygon(arrowHead);
    return path;
}

void EditorArrow::updatePosition()
{
    QLineF line(mapFromItem(startItem, 0, 0), mapFromItem(endItem, 0, 0));
    setLine(line);
}

int EditorArrow::getType()
{
    return typeItem;
}

bool EditorArrow::isValidate()
{
    return validate;
}

void EditorArrow::setValidate(bool val)
{

    validate = val;
}

void EditorArrow::paint(QPainter *painter, const QStyleOptionGraphicsItem *,
          QWidget *)
{
    if(startItem != NULL)
    {
        if (startItem->collidesWithItem(endItem))
            return;

        QPen myPen = pen();
        myPen.setColor(color);
        qreal arrowSize = 20;
        painter->setPen(myPen);
        painter->setBrush(color);

        QLineF centerLine(startItem->pos(), endItem->pos());
        QPolygonF endPolygon = endItem->polygon();
        QPointF p1 = endPolygon.first() + endItem->pos();
        QPointF p2;
        QPointF intersectPoint;
        QLineF polyLine;
        for (int i = 1; i < endPolygon.count(); ++i)
        {
            p2 = endPolygon.at(i) + endItem->pos();
            polyLine = QLineF(p1, p2);
            QLineF::IntersectType intersectType = polyLine.intersect(centerLine, &intersectPoint);
            if (intersectType == QLineF::BoundedIntersection)
                break;
            p1 = p2;
        }

        setLine(QLineF(intersectPoint, startItem->pos()));

        double angle = ::acos(line().dx() / line().length());
        if (line().dy() >= 0)
            angle = (Pi * 2) - angle;

        QPointF arrowP1 = line().p1() + QPointF(sin(angle + Pi / 3) * arrowSize,
                                            cos(angle + Pi / 3) * arrowSize);
        QPointF arrowP2 = line().p1() + QPointF(sin(angle + Pi - Pi / 3) * arrowSize,
                                            cos(angle + Pi - Pi / 3) * arrowSize);

        arrowHead.clear();
        arrowHead << line().p1() << arrowP1 << arrowP2;

        painter->drawLine(line());
        painter->drawPolygon(arrowHead);
        if(isSelected())
        {
            painter->setPen(QPen(color, 1, Qt::DashLine));
            QLineF myLine = line();
            myLine.translate(0, 4.0);
            painter->drawLine(myLine);
            myLine.translate(0,-8.0);
            painter->drawLine(myLine);
        }
    }
    else
    {
        if (startImage->collidesWithItem(endItem))
            return;

        QPen myPen = pen();
        myPen.setColor(color);
        qreal arrowSize = 20;
        painter->setPen(myPen);
        painter->setBrush(color);

        QLineF centerLine(startImage->pos(), endItem->pos());
        QPolygonF endPolygon = endItem->polygon();
        QPointF p1 = endPolygon.first() + endItem->pos();
        QPointF p2;
        QPointF intersectPoint;
        QLineF polyLine;
        for (int i = 1; i < endPolygon.count(); ++i)
        {
            p2 = endPolygon.at(i) + endItem->pos();
            polyLine = QLineF(p1, p2);
            QLineF::IntersectType intersectType = polyLine.intersect(centerLine, &intersectPoint);
            if (intersectType == QLineF::BoundedIntersection)
                break;
            p1 = p2;
        }

        setLine(QLineF(intersectPoint, QPointF(startImage->pos().x() +startImage->pixmap().width()/2 , startImage->pos().y()+startImage->pixmap().height()/2 )));

        double angle = ::acos(line().dx() / line().length());
        if (line().dy() >= 0)
            angle = (Pi * 2) - angle;

        QPointF arrowP1 = line().p1() + QPointF(sin(angle + Pi / 3) * arrowSize,
                                            cos(angle + Pi / 3) * arrowSize);
        QPointF arrowP2 = line().p1() + QPointF(sin(angle + Pi - Pi / 3) * arrowSize,
                                            cos(angle + Pi - Pi / 3) * arrowSize);

        arrowHead.clear();
        arrowHead << line().p1() << arrowP1 << arrowP2;

        painter->drawLine(line());
        painter->drawPolygon(arrowHead);
        if(isSelected())
        {
            painter->setPen(QPen(color, 1, Qt::DashLine));
            QLineF myLine = line();
            myLine.translate(0, 4.0);
            painter->drawLine(myLine);
            myLine.translate(0,-8.0);
            painter->drawLine(myLine);
        }
    }
}

void EditorArrow::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    scene()->clearSelection();
    setSelected(true);
    myContextMenu->exec(event->screenPos());
}
