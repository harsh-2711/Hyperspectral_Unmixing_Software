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

#include "editorscene.h"

#define CHARLENGTH 2.5

EditorScene::EditorScene(QObject *parent, QMenu* itemMenu, QMenu *arrowMenu) :
    QGraphicsScene(parent)
{
    myItemMenu = itemMenu;
    myArrowMenu = arrowMenu;
    myMode = Nothing;
    line = 0;
    contElements = 0;
    contImages = 0;
}


void EditorScene::setMode(Mode mode)
{
    myMode = mode;
}

void EditorScene::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{

    itemSel = itemAt(event->scenePos());
    if(itemSel != NULL)
    {
        if(qgraphicsitem_cast<EditorItem *>(itemSel) != 0)
        {
            item = elements.at(itemSel->data(0).toInt());
            label = item->getLabel();
            emit itemClicked(item);
        }
    }

    QGraphicsScene::mouseDoubleClickEvent(event);
}

void EditorScene::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{

    if(mouseEvent->button() == Qt::LeftButton)
    {

        switch (myMode)
        {
            case InsertItem:
                image = NULL;
                item = new EditorItem(myItemMenu);

                item->setWorkspace(workspace);
                itemSel = item;
                //item->setBrush(QColor(238,221,130));
                addItem(item);

                label = new QGraphicsTextItem();
                item->setPos(mouseEvent->scenePos());
                item->setLabel(label);
                addItem(label);

                item->setData(0, QVariant(contElements));
                elementsLabels.append(label);
                elements.append(item);
                contElements++;

                emit itemInserted(item);
                break;

            case InsertImage:
                imageD = new NewImageDialog();
                imageD->setWorkspace(workspace);
                imageD->exec();

                if(imageD->isInserted())
                {
                    item = NULL;
                    image = new EditorImageItem(NULL);
                    itemSel = image;
                    image->setPos(mouseEvent->scenePos());
                    image->setPixmap(QPixmap::fromImage(imageD->getBand(imageD->getBandSel())).scaled(200, 200));
                    addItem(image);

                    imageList.append(image);
                    contImages++;

                    emit imageInserted(imageD, image);
                }
                else emit imageInserted(imageD, NULL);

                break;

            case InsertLine:
                line = new QGraphicsLineItem(QLineF(mouseEvent->scenePos(),
                                            mouseEvent->scenePos()));
                line->setPen(QPen(Qt::black, 2));
                addItem(line);

                break;

            case MoveItem:
                itemSel = itemAt(mouseEvent->scenePos());
                if(itemSel != NULL)
                {
                    if(qgraphicsitem_cast<EditorItem *>(itemSel) != 0)
                    {
                        item = elements.at(itemSel->data(0).toInt());
                        label = item->getLabel();
                    }
                    if(qgraphicsitem_cast<EditorImageItem *>(itemSel) != 0)
                    {
                        image = imageList.at(itemSel->data(0).toInt());
                    }
                }
                break;

            default: ;
        }
    }
    else if (mouseEvent->button() == Qt::RightButton)
    {
        if(itemAt(mouseEvent->scenePos()) != NULL)
        {
            itemSel = itemAt(mouseEvent->scenePos());
            if(itemSel != NULL)
            {
                if(qgraphicsitem_cast<EditorItem *>(itemSel) != 0)
                {
                    item = elements.at(itemSel->data(0).toInt());
                    label = item->getLabel();
                }
                if(qgraphicsitem_cast<EditorImageItem *>(itemSel) != 0)
                {
                    image = imageList.at(itemSel->data(0).toInt());
                }
            }
        }
    }

    QGraphicsScene::mousePressEvent(mouseEvent);

}

void EditorScene::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{

    if (myMode == InsertLine && line != 0)
    {
        QLineF newLine(line->line().p1(), mouseEvent->scenePos());
        line->setLine(newLine);

    }
    else if (myMode == MoveItem)
    {
        if(itemSel != NULL)
        {
            if(item!=NULL && qgraphicsitem_cast<EditorImageItem *>(itemSel) == 0)
            {
                item->getLabel()->setPos(item->scenePos()-QPointF(3*label->toPlainText().length()*CHARLENGTH,10));
            }
        }
        QGraphicsScene::mouseMoveEvent(mouseEvent);
    }
}

void EditorScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if (line != 0 && myMode == InsertLine)
    {
        QList<QGraphicsItem *> startItems = items(line->line().p1());
        if (startItems.count() && startItems.first() == line)
            startItems.removeFirst();
        QList<QGraphicsItem *> endItems = items(line->line().p2());
        if (endItems.count() && endItems.first() == line)
            endItems.removeFirst();

        removeItem(line);
        delete line;
        if (startItems.count() > 0 && endItems.count() > 0 &&
            startItems.first() != endItems.first())
        {
            EditorItem *_startItem;
            EditorItem *_endItem;
            EditorImageItem *_startImage;
            if(qgraphicsitem_cast<EditorItem *>(startItems.first()) != 0)
            {
                _startItem = qgraphicsitem_cast<EditorItem *>(startItems.first());

                if(qgraphicsitem_cast<EditorItem *>(endItems.first()) != 0)
                {
                    _endItem = qgraphicsitem_cast<EditorItem *>(endItems.first());
                    EditorArrow *arrow = new EditorArrow(myArrowMenu, _startItem, _endItem);
                    arrow->setColor(Qt::black);
                    _startItem->addArrow(arrow);
                    _endItem->addArrow(arrow);
                    arrow->setZValue(-1000.0);
                    addItem(arrow);
                    arrow->updatePosition();
                }
            }
            if(qgraphicsitem_cast<EditorImageItem *>(startItems.first()) != 0)
            {
                _startImage = qgraphicsitem_cast<EditorImageItem *>(startItems.first());

                if(qgraphicsitem_cast<EditorItem *>(endItems.first()) != 0)
                {
                    _endItem = qgraphicsitem_cast<EditorItem *>(endItems.first());
                    EditorArrow *arrow = new EditorArrow(_startImage, _endItem);
                    arrow->setColor(Qt::black);
                    _startImage->addArrow(arrow);
                    _endItem->addArrow(arrow);
                    arrow->setZValue(-1000.0);
                    addItem(arrow);
                    arrow->updatePosition();

                }
            }
        }

    }
    line = 0;
    emit itemInserted(item);
    QGraphicsScene::mouseReleaseEvent(mouseEvent);
}

void EditorScene::deleteItemSelected()
{
    if(itemSel != NULL)
    {
        if(qgraphicsitem_cast<EditorItem *>(itemSel) != 0)
        {
            item = qgraphicsitem_cast<EditorItem *>(itemSel);
            item->removeArrows();
            item->removeLabel();
            label = NULL;
            this->removeItem(itemSel);
            item = NULL;
            itemSel= NULL;
        }
        else if(qgraphicsitem_cast<EditorImageItem *>(itemSel) != 0)
        {
            image = qgraphicsitem_cast<EditorImageItem *>(itemSel);
            image->removeArrows();
            this->removeItem(itemSel);
            image = NULL;
            itemSel= NULL;
        }
        else if(qgraphicsitem_cast<EditorArrow *>(itemSel) != 0)
        {
            arrow = qgraphicsitem_cast<EditorArrow *>(itemSel);
            if(arrow->getStartItem() != NULL) arrow->getStartItem()->removeArrow(arrow);
            else arrow->getStartImage()->removeArrow(arrow);
            arrow->getEndItem()->removeArrow(arrow);
            arrow = NULL;
            label = NULL;
            this->removeItem(itemSel);
            item = NULL;
            itemSel= NULL;
        }
    }

}

void EditorScene::openParameterDialog()
{
    if(qgraphicsitem_cast<EditorItem*>(itemSel) != 0)
    {
        item = qgraphicsitem_cast<EditorItem*>(itemSel);
        item->openParametersDialog();
    }
}

void EditorScene::setWorkspace(QString ws)
{
    workspace = ws;
}

void EditorScene::setImageDialog(NewImageDialog *imD)
{
    imageD = imD;
}

void EditorScene::setSelectedAll()
{
    foreach(EditorImageItem* im, imageList)
        im->setSelected(false);
    foreach(EditorItem* it, elements)
        it->setSelected(false);

}

void EditorScene::setBackgroundcolor(QString color)
{
    backgroundcolor = color;
}

