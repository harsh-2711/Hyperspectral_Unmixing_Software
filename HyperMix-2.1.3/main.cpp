#include <QtGui/QApplication>
#include "hypermix2.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    HyperMix2 w;
    w.show();
    
    return a.exec();
}
