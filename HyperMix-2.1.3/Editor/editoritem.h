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

#ifndef EDITORITEM_H
#define EDITORITEM_H

#include <QGraphicsPolygonItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsTextItem>
#include <QGraphicsScene>
#include <QList>
#include <QMenu>
#include <QPainter>
#include <stdio.h>
#include "parameterdialog.h"
#include "editorarrow.h"

class EditorArrow;
class parameterDialog;

class EditorItem : public QGraphicsPolygonItem
{

    QPolygonF square;
    QList<EditorArrow*> arrows;
    QMenu *myContextMenu;
    QGraphicsTextItem *label;
    QString operatorName;
    unsigned int operatorNumber;
    int typeItem;
    bool validate;
    int numValidates;
    QString workspace;
    parameterDialog *paramD;

public:

    EditorItem(QMenu *contextMenu, QGraphicsItem *parent = 0, QGraphicsScene *scene = 0);
    void removeArrow(EditorArrow *arrow);
    void removeArrows();
    void addArrow(EditorArrow *arrow);
    QList<EditorArrow*> getArrows();
    QList<EditorArrow*> getArrowsStartsInItem();
    QList<EditorArrow*> getArrowsEndsInItem();
    QList<EditorArrow*> getArrowsEndsInItemWithImages();
    QList<EditorArrow*> getArrowsStartsInItemNotChecked();
    QList<EditorArrow*> getArrowsEndsInItemNotChecked();
    void setValidateAllArrows(bool val);
    bool isValidate();
    void setValidate(bool val);
    int getNumValidates();
    void setLabel(QGraphicsTextItem *_label);
    void removeLabel();
    QGraphicsTextItem* getLabel();
    int getType();
    void setType(int type);
    void setOperatorName(QString name);
    QString getOperatorName();
    void setOperatorNumber(unsigned int n);
    unsigned int getOperatorNumber();
    void openParametersDialog();
    int checkParameters();
    QStringList getArguments();
    QStringList getArgumentsShow();
    void setWorkspace(QString ws);
    void addParamsIn(QStringList params);
    QStringList getParamsOut();
    QStringList getParamsOutTypes();
    QMenu* getContextMenu();
    QString getWorkspace();
    EditorItem* copyItem();
    EditorItem* copyItem(EditorArrow* a, bool start);
    bool isOutputChecked();


private:
    void createItemMenu();

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value);
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

};

#endif // EDITORITEM_H
