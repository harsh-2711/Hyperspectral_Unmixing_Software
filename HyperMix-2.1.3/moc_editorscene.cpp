/****************************************************************************
** Meta object code from reading C++ file 'editorscene.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "Editor/editorscene.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'editorscene.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_EditorScene[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: signature, parameters, type, tag, flags
      18,   13,   12,   12, 0x05,
      44,   13,   12,   12, 0x05,
      82,   69,   12,   12, 0x05,

 // slots: signature, parameters, type, tag, flags
     135,  130,   12,   12, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_EditorScene[] = {
    "EditorScene\0\0item\0itemInserted(EditorItem*)\0"
    "itemClicked(EditorItem*)\0imageD,image\0"
    "imageInserted(NewImageDialog*,EditorImageItem*)\0"
    "mode\0setMode(Mode)\0"
};

void EditorScene::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        EditorScene *_t = static_cast<EditorScene *>(_o);
        switch (_id) {
        case 0: _t->itemInserted((*reinterpret_cast< EditorItem*(*)>(_a[1]))); break;
        case 1: _t->itemClicked((*reinterpret_cast< EditorItem*(*)>(_a[1]))); break;
        case 2: _t->imageInserted((*reinterpret_cast< NewImageDialog*(*)>(_a[1])),(*reinterpret_cast< EditorImageItem*(*)>(_a[2]))); break;
        case 3: _t->setMode((*reinterpret_cast< Mode(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData EditorScene::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject EditorScene::staticMetaObject = {
    { &QGraphicsScene::staticMetaObject, qt_meta_stringdata_EditorScene,
      qt_meta_data_EditorScene, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &EditorScene::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *EditorScene::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *EditorScene::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_EditorScene))
        return static_cast<void*>(const_cast< EditorScene*>(this));
    return QGraphicsScene::qt_metacast(_clname);
}

int EditorScene::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsScene::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void EditorScene::itemInserted(EditorItem * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void EditorScene::itemClicked(EditorItem * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void EditorScene::imageInserted(NewImageDialog * _t1, EditorImageItem * _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_END_MOC_NAMESPACE
