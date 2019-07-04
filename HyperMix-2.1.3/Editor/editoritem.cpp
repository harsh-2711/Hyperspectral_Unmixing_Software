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


#include "editoritem.h"

EditorItem::EditorItem(QMenu *contextMenu, QGraphicsItem *parent, QGraphicsScene *scene)
                        : QGraphicsPolygonItem(parent, scene)
{

    myContextMenu = contextMenu;

    square << QPointF(-100, -100) << QPointF(100, -100)
           << QPointF(100, 100) << QPointF(-100, 100) << QPointF(-100, -100);


    typeItem = 1;
    setPolygon(square);
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    validate = false;
    numValidates = 0;

    paramD = new parameterDialog();
    paramD->setItem(this);

}

void EditorItem::openParametersDialog()
{
    paramD->setWorkspace(workspace);
    paramD->exec();
}

QStringList EditorItem::getArguments()
{
    return paramD->getParameters();
}

void EditorItem::removeArrow(EditorArrow *arrow)
{
    int index = arrows.indexOf(arrow);

    if (index != -1)
            arrows.removeAt(index);
}

void EditorItem::removeArrows()
{
    foreach (EditorArrow *arrow, arrows)
    {
        if(arrow->getStartItem() != NULL)
            arrow->getStartItem()->removeArrow(arrow);
        else
            arrow->getStartImage()->removeArrow(arrow);
        arrow->getEndItem()->removeArrow(arrow);
        scene()->removeItem(arrow);
        delete arrow;
    }
}

void EditorItem::addArrow(EditorArrow *arrow)
{
    arrows.append(arrow);
}

QList<EditorArrow*> EditorItem::getArrows()
{
    return arrows;
}

QList<EditorArrow*> EditorItem::getArrowsEndsInItemWithImages()
{

    QList<EditorArrow*> aux;
    foreach(EditorArrow* arrow, arrows)
        if(this == arrow->getEndItem())
            aux.append(arrow);
    return aux;
}

QList<EditorArrow*> EditorItem::getArrowsEndsInItem()
{

    QList<EditorArrow*> aux;
    foreach(EditorArrow* arrow, arrows)
        if(this == arrow->getEndItem() && arrow->getStartItem() != NULL)
            aux.append(arrow);
    return aux;
}

QList<EditorArrow*> EditorItem::getArrowsStartsInItem()
{

    QList<EditorArrow*> aux;
    foreach(EditorArrow* arrow, arrows)
        if(this == arrow->getStartItem()  && arrow->getStartItem() != NULL)
            aux.append(arrow);
    return aux;
}

QList<EditorArrow*> EditorItem::getArrowsEndsInItemNotChecked()
{

    QList<EditorArrow*> aux;
    foreach(EditorArrow* arrow, arrows)
         if(this == arrow->getEndItem() && !arrow->isValidate() && arrow->getStartItem() != NULL)
            aux.append(arrow);
    return aux;
}

QList<EditorArrow*> EditorItem::getArrowsStartsInItemNotChecked()
{

    QList<EditorArrow*> aux;
    foreach(EditorArrow* arrow, arrows)
         if(this == arrow->getStartItem() && !arrow->isValidate()  && arrow->getStartItem() != NULL)
            aux.append(arrow);
    return aux;
}

void EditorItem::setValidateAllArrows(bool val)
{
    foreach(EditorArrow* arrow, arrows)
        arrow->setValidate(val);
}

bool EditorItem::isValidate()
{
    return validate;
}

void EditorItem::setValidate(bool val)
{
    validate = val;
    if(validate) numValidates++;
}

int EditorItem::getNumValidates()
{
    return numValidates;
}

void EditorItem::setLabel(QGraphicsTextItem *_label)
{
    label = _label;
}

void EditorItem::removeLabel()
{
    delete label;
}

QGraphicsTextItem* EditorItem::getLabel()
{
    return label;
}

int EditorItem::getType()
{
    return typeItem;
}

void EditorItem::setType(int type)
{
    typeItem = type;
}

void EditorItem::setOperatorName(QString name)
{
    operatorName = name;
}

QString EditorItem::getOperatorName()
{
    return operatorName;
}

void EditorItem::setOperatorNumber(unsigned int n)
{
    operatorNumber = n;
}

unsigned int EditorItem::getOperatorNumber()
{
    return operatorNumber;
}

QVariant EditorItem::itemChange(GraphicsItemChange change,
                     const QVariant &value)
{
    if (change == QGraphicsItem::ItemPositionChange)
        foreach (EditorArrow *arrow, arrows)
            arrow->updatePosition();


    return value;
}

void EditorItem::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    scene()->clearSelection();
    setSelected(true);
    myContextMenu->exec(event->screenPos());
}

void EditorItem::setWorkspace(QString ws)
{
    workspace = ws;
}

QMenu* EditorItem::getContextMenu()
{
    return myContextMenu;
}

QString EditorItem::getWorkspace()
{
    return workspace;
}

EditorItem* EditorItem::copyItem()
{
    EditorItem* newItem = new EditorItem(myContextMenu);

    newItem->setOperatorName(operatorName);
    newItem->setOperatorNumber(operatorNumber);
    newItem->setWorkspace(workspace);

    foreach(EditorArrow* arrow, arrows)
    {
        if(arrow->getStartImage() != NULL) newItem->addArrow(new EditorArrow(arrow->getStartImage(), newItem));
        else
        {
            if(arrow->getStartItem() == this) newItem->addArrow(new EditorArrow(myContextMenu, newItem, arrow->getEndItem()));
            else newItem->addArrow(new EditorArrow(myContextMenu, arrow->getStartItem(), newItem));
        }
    }

    return newItem;

}

EditorItem* EditorItem::copyItem(EditorArrow* a, bool start)
{
    EditorItem* newItem = new EditorItem(myContextMenu);

    newItem->setOperatorName(operatorName);
    newItem->setOperatorNumber(operatorNumber);
    newItem->setWorkspace(workspace);

    if(start)
    {
        foreach(EditorArrow* arrow, this->getArrowsStartsInItem())
        {
            if(arrow == a)
            {
                if(arrow->getStartItem() != NULL) newItem->addArrow(new EditorArrow(myContextMenu, newItem, arrow->getEndItem()));
                else newItem->addArrow(new EditorArrow(arrow->getStartImage(),arrow->getEndItem()));

                this->removeArrow(a);
            }
        }

        foreach(EditorArrow* arrow, this->getArrowsEndsInItem())
        {
            if(arrow->getStartItem() != NULL) newItem->addArrow(new EditorArrow(myContextMenu,arrow->getStartItem(),newItem));
            else newItem->addArrow(new EditorArrow(arrow->getStartImage(), newItem));
        }
    }
    else
    {
        foreach(EditorArrow* arrow, this->getArrowsEndsInItem())
        {
            if(arrow == a)
            {
                if(arrow->getStartItem() != NULL) newItem->addArrow(new EditorArrow(myContextMenu,arrow->getStartItem(),newItem));
                else newItem->addArrow(new EditorArrow(arrow->getStartImage(), newItem));

                this->removeArrow(a);
            }
        }

        foreach(EditorArrow* arrow, this->getArrowsStartsInItem())
        {
            if(arrow->getStartItem() != NULL) newItem->addArrow(new EditorArrow(myContextMenu, newItem, arrow->getEndItem()));
            else newItem->addArrow(new EditorArrow(arrow->getStartImage(),arrow->getEndItem()));
        }
    }

    return newItem;

}

bool EditorItem::isOutputChecked()
{
    return paramD->isOutputChecked();
}
