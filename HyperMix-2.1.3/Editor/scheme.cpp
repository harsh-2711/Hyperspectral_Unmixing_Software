#include "scheme.h"
#define DEBUG_MODE 1

Scheme::Scheme(QProgressBar *_progress, QObject *parent) :
    QObject(parent)
{

    progress = _progress;
    logNumber = 0;

}


void Scheme::addItem(EditorItem *it)
{

    EditorItem* aux = NULL;
    foreach(EditorItem* item, activeItems)
        if(it->getOperatorName() == item->getOperatorName())
            aux = item;

    if(aux == NULL) it->setOperatorNumber(1);
    else it->setOperatorNumber(aux->getOperatorNumber()+1);
    activeItems.append(it);
}

void Scheme::addItem(EditorImageItem *it)
{

    EditorImageItem* aux = NULL;
    foreach(EditorImageItem* item, activeImages)
        if(it->getOperatorName() == item->getOperatorName())
            aux = item;

    if(aux == NULL) it->setOperatorNumber(1);
    else it->setOperatorNumber(aux->getOperatorNumber()+1);
    activeImages.append(it);
}


void Scheme::removeItem(EditorItem *it)
{
    activeItems.removeOne(it);
}

void Scheme::removeImage(EditorImageItem *it)
{
    activeImages.removeOne(it);
}


int Scheme::compileScheme(ConsoleLogDialog *consoleD)
{

    /*
    COMPILING ERRORS
    -4: ilogical order
    -5: impossible conecction
    -6: missing parameters
    -7: empty diagram
    */
    int error;
    QString address;
    QDateTime dateTime;
    QString dateTimeString;

    char cad[MAXCAD]="";

    if(DEBUG_MODE == 0) address = "/usr/share/hypermix-2.0/info";
    else address = "info";

    FILE* fp;
    if((fp = fopen(address.append("/console.log").toStdString().c_str(), "a")) != NULL)
    {

        progress->setVisible(true);

        dateTime = QDateTime::currentDateTime();
        dateTimeString = dateTime.toString();
        fprintf(fp,"Date and Time: %s\n", dateTimeString.toStdString().c_str());
        sprintf(cad,"Date and Time: %s\n", dateTimeString.toStdString().c_str());

        consoleD->write(cad);

        fseek(fp, 0L, SEEK_END);
        fprintf(fp,"--------------------------------------------------------------------------------------------------------------\n");
        fflush(fp);
        consoleD->write(tr("--------------------------------------------------------------------------------------------------------------\n"));


        foreach(EditorImageItem* item, activeImages)
        {
            fprintf(fp,"IMAGE USED: %s\n", item->getOperatorFilename().toStdString().c_str());
            sprintf(cad,"IMAGE USED: %s\n", item->getOperatorFilename().toStdString().c_str());
            consoleD->write(cad);
        }

        fprintf(fp,"\nCompiling diagram...\n");
        fflush(fp);
        consoleD->write(tr("\nCompiling diagram...\n"));


        if(activeItems.isEmpty())
        {
            fprintf(fp,"The diagram is empty. Nothing is done.\n");
            consoleD->write(tr("The diagram is empty. Nothing is done."));
            return -7;
        }
        // Checking logical order
        error = checkLogicalOrder(consoleD);
        if(error != 0) return error;

        //All OK
        fprintf(fp,"Diagram checked ---> OK\n");
        fflush(fp);

        consoleD->write(tr("Diagram checked ---> OK\n"));

        fclose(fp);
    }

    return 0;
}

int Scheme::checkLogicalOrder(ConsoleLogDialog *consoleD)
{
    EditorItem* item;
    EditorArrow* arrow;
    QString address;
    char cad[MAXCAD] = "";

    if(DEBUG_MODE == 0) address = "/usr/share/hypermix-2.0/info";
    else address = "info";

    FILE* fp;
    if((fp = fopen(address.append("/console.log").toStdString().c_str(), "a")) != NULL)
    {
        fseek(fp, 0L, SEEK_END);

        fprintf(fp, "Checking logical order...\n");
        fflush(fp);
        consoleD->write(tr("Checking logical order...\n"));


        foreach(item, activeItems)
        {
            item->setBrush(QColor(238,221,130));

            foreach(arrow, item->getArrows())
            {
                if(arrow->getStartItem() != NULL)
                {
                    if(arrow->getStartItem() == item)
                    {
                        if(item->getType() > arrow->getEndItem()->getType())
                        {
                            fprintf(fp,"ERROR: Ilogical order of items: %s ---> %s\n", item->getOperatorName().toStdString().c_str(),
                                    arrow->getEndItem()->getOperatorName().toStdString().c_str());
                            sprintf(cad,"ERROR: Ilogical order of items: %s ---> %s\n", item->getOperatorName().toStdString().c_str(),
                                    arrow->getEndItem()->getOperatorName().toStdString().c_str());
                            consoleD->write(cad);
                            item->setBrush(QColor(255,99,71));
                            arrow->getEndItem()->setBrush(QColor(255,99,71));
                            return -4;
                        }
                        if(item->getType() == arrow->getEndItem()->getType() && item->getType() != 2)
                        {
                            fprintf(fp,"ERROR: Ilogical order of items: %s ---> %s\n", item->getOperatorName().toStdString().c_str(),
                                    arrow->getEndItem()->getOperatorName().toStdString().c_str());
                            sprintf(cad,"ERROR: Ilogical order of items: %s ---> %s\n", item->getOperatorName().toStdString().c_str(),
                                    arrow->getEndItem()->getOperatorName().toStdString().c_str());
                            consoleD->write(cad);
                            item->setBrush(QColor(255,99,71));
                            arrow->getEndItem()->setBrush(QColor(255,99,71));
                            return -4;
                        }
                    }
                    else //if it is not the start item must be the end item
                    {
                        if(item->getType() < arrow->getStartItem()->getType())
                        {
                            fprintf(fp,"ERROR: Ilogical order of items: %s ---> %s\n", arrow->getStartItem()->getOperatorName().toStdString().c_str(),
                                    item->getOperatorName().toStdString().c_str());
                            sprintf(cad,"ERROR: Ilogical order of items: %s ---> %s\n", arrow->getStartItem()->getOperatorName().toStdString().c_str(),
                                    item->getOperatorName().toStdString().c_str());
                            consoleD->write(cad);
                            item->setBrush(QColor(255,99,71));
                            arrow->getStartItem()->setBrush(QColor(255,99,71));
                            return -4;
                        }
                        if(item->getType() == arrow->getStartItem()->getType() && item->getType() != 2)
                        {
                            fprintf(fp,"ERROR: Ilogical order of items: %s ---> %s\n", arrow->getStartItem()->getOperatorName().toStdString().c_str(),
                                    item->getOperatorName().toStdString().c_str());
                            sprintf(cad,"ERROR: Ilogical order of items: %s ---> %s\n", arrow->getStartItem()->getOperatorName().toStdString().c_str(),
                                    item->getOperatorName().toStdString().c_str());
                            consoleD->write(cad);
                            item->setBrush(QColor(255,99,71));
                            arrow->getStartItem()->setBrush(QColor(255,99,71));
                            return -4;
                        }
                    }
                }
                else
                {
                    foreach(EditorArrow* aux, arrow->getEndItem()->getArrows())
                    {
                        if(aux->getStartImage() != NULL && aux->getStartImage() != arrow->getStartImage())
                        {
                            fprintf(fp,"ERROR: More than one image attached: %s\n", arrow->getEndItem()->getOperatorName().toStdString().c_str());
                            sprintf(cad,"ERROR: More than one image attached: %s\n", arrow->getEndItem()->getOperatorName().toStdString().c_str());
                            consoleD->write(cad);
                            return -5;
                        }
                    }
                }
            }
            item->setBrush(QColor(154,205,50));
        }

        fflush(fp);
        fprintf(fp,"Logical order ---> OK\n");
        fflush(fp);
        consoleD->write(tr("Logical order ---> OK\n"));

        fclose(fp);
    }


    return 0;
}

int Scheme::runScheme(int numberExecutions, ConsoleLogDialog *consoleD)
{
    int i;
    QList<EditorItem*> aux;
    EditorItem* item;
    QStringList arguments;
    QProcess *process;
    QString program;
    QString address;
    QString auxiliar;
    int cont;
    char cad[MAXCAD]="";


    if(DEBUG_MODE == 0) address = "/usr/share/hypermix-2.0/info";
    else address = "info";

    FILE* fp;
    if((fp = fopen(address.append("/console.log").toStdString().c_str(), "a")) != NULL)
    {
        fseek(fp, 0L, SEEK_END);
        fprintf(fp, "Executing chain...\n");
        fflush(fp);
        consoleD->write(tr("Executing chain...\n"));

        for(i=1; i<5; i++)
            foreach(item, activeItems)
                if(item->getType() == i)
                    aux.append(item);

        for(i=0; i<numberExecutions; i++)
        {
            fprintf(fp,"\n******** Starting execution number: %d ********\n\n", i+1);
            fflush(fp);
            sprintf(cad,"\n******** Starting execution number: %d ********\n\n", i+1);
            consoleD->write(cad);

            if(DEBUG_MODE == 0) address = "/usr/share/hypermix-2.0/bin/";
            else address = "bin/";
            cont = 0;
            foreach(item, aux)
            {
                //EXECUTING
                cont++;
                progress->setValue((int)((i*aux.count() + cont)*100/(numberExecutions*aux.count())));
                arguments.clear();
                arguments = item->getArguments();
                if(!arguments.isEmpty())
                {
                    program.clear();
                    program.append(address).append(item->getOperatorName());

                    process = new QProcess();
                    process->start(program, arguments);
                    auxiliar.clear();

                    while(process->waitForFinished(100000))
                       auxiliar.append(process->readAllStandardOutput());

                    fprintf(fp,"%s\n", auxiliar.toStdString().c_str());
                    sprintf(cad,"%s\n", auxiliar.toStdString().c_str());
                    consoleD->write(cad);

                    //SHOW RESULTS IF NEEDED
                    if(item->isOutputChecked())
                    {
                        switch(item->getType())
                        {
                        case 1:
                            break;
                        case 2: emit newImageResult(arguments.last());
                            break;
                        case 3: emit newEndmemberResult(arguments.last());
                            break;
                        case 4:emit newAbundanceResult(arguments.last());
                            break;
                        }
                    }
                }

            }
        }
        progress->setVisible(false);
        fprintf(fp, "\nEnd of execution...\n\n");
        consoleD->write(tr("\nEnd of execution...\n\n"));
        fprintf(fp,"--------------------------------------------------------------------------------------------------------------\n");
        consoleD->write(tr("--------------------------------------------------------------------------------------------------------------\n"));
        fflush(fp);
        fclose(fp);
    }

    return 0;
}


void Scheme::cleanScheme()
{
    foreach(EditorItem* item, activeItems)
        if(item->getOperatorName().contains("_CUDA")) item->setBrush(QColor(176,224,230));
        else item->setBrush(QColor(238,221,130));
}

QStringList Scheme::getListActiveImages()
{
    QStringList images;
    QString image;
    foreach (EditorImageItem* aux, activeImages)
    {
        image = aux->getOperatorFilename();
        images.append(image);
    }

    return images;
}

void Scheme::setWorkspace(QString ws)
{
    workspace = ws;

    foreach(EditorItem* item, activeItems)
        item->setWorkspace(workspace);
}




