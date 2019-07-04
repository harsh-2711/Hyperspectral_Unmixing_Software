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

#ifndef EDITORARROW_H
#define EDITORARROW_H

#include <QGraphicsLineItem>
#include <QGraphicsSceneContextMenuEvent>
#include <QPen>
#include <QPainter>

#include "Editor/editoritem.h"
#include "Editor/editorimageitem.h"

class EditorItem;
class EditorImageItem;

class EditorArrow : public QGraphicsLineItem
{
    QMenu *myContextMenu;
    EditorImageItem *startImage;
    EditorItem *startItem;
    EditorItem *endItem;
    QColor color;
    QPolygonF arrowHead;
    int typeItem;
    bool validate;

public:


    EditorArrow(QMenu *contextMenu, EditorItem *_startItem, EditorItem *_endItem,
                QGraphicsItem *parent = 0, QGraphicsScene *scene = 0);
    EditorArrow(EditorImageItem *_startItem, EditorItem *_endItem,
                QGraphicsItem *parent = 0, QGraphicsScene *scene = 0);
    EditorItem* getStartItem() const {return startItem;}
    EditorImageItem* getStartImage() const {return startImage;}
    EditorItem* getEndItem() const {return endItem;}
    bool isValidate();
    void setValidate(bool val);
    void openArrowDialog();
    void addParam(QString type, QString value);
    void deleteParams();

    QRectF boundingRect() const;
    QPainterPath shape() const;
    void setColor(const QColor &_color)
        { color = _color; }

    void updatePosition();
    int getType();

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget = 0);
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

};

#endif // EDITORARROW_H
