#include "QtNet.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QtNet w;
    w.show();
    return a.exec();
}
