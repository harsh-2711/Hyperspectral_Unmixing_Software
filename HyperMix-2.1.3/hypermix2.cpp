#include "hypermix2.h"
#include "ui_hypermix2.h"


#define MAXOPERATORS 30

#define DEBUG_MODE 1


HyperMix2::HyperMix2(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::HyperMix2)
{
    ui->setupUi(this);

    if(DEBUG_MODE != 1)
    {
        system("export CUDA_HOME=/usr/local/cuda");
        system("export LD_LIBRARY_PATH=${CUDA_HOME}/lib64");
        system("PATH=${CUDA_HOME}/bin:${PATH}");
        system("export PATH");
    }
    //VARIABLES*******************************************************
    cudaSupported = false;
    helpD = new HelpDialog();
    workspaceD = new WorkspaceDialog();

    endmemberResult = NULL;
    emptyEnd = true;
    endmemberRef = NULL;
    endmemberSig = NULL;
    endmemberResultRef = NULL;
    endmemberResultSig = NULL;
    imageResult = NULL;
    imageRecon = NULL;
    imageReconOri = NULL;
    abundanceResult = NULL;

    waveUnitE = NULL;
    waveUnitI = NULL;

    wavelengthI = NULL;
    wavelengthE = NULL;
    waveUnitRef = NULL;
    waveUnitSig = NULL;

    wavelengthRef = NULL;
    wavelengthSig = NULL;

    interleaveA = NULL;
    interleaveE = NULL;
    interleaveI = NULL;
    interleaveR = NULL;
    interleaveRO = NULL;
    interleaveRef = NULL;
    interleaveSig = NULL;

    imageRGB = NULL;
    imageAbundance = NULL;
    imageRMSE = NULL;



    rowRef = false;
    rowSig = false;

    maxminRMSE = 1;

    ui->RMSEButton->setEnabled(false);
    ui->sliderBands->setEnabled(false);

    //USER INTERFACE*******************************************************
    loadToolBar();
    loadButtons();
    loadOperators();

    scene = new EditorScene(this, itemMenu, arrowMenu);
    scene->setSceneRect(QRectF(0, 0, 2000, 2000));

    sceneImage = new EditorImageScene();
    ui->graphicsViewImages->setScene(sceneImage);
    sceneAbundance = new EditorAbundanceScene();
    ui->graphicsViewAbundances->setScene(sceneAbundance);
    sceneRMSE = new EditorRMSEScene();

    QMatrix oldMatrix = ui->graphicsView->matrix();
    ui->graphicsView->resetMatrix();
    ui->graphicsView->translate(oldMatrix.dx(), oldMatrix.dy());
    ui->graphicsView->scale(0.5, 0.5);
    ui->graphicsView->setScene(scene);
    ui->workspaceLineEdit->setText(QDir::homePath());

    QPalette p = ui->graphicsView->palette();
    p.setColor(QPalette::Base, Qt::white);
    ui->graphicsView->setPalette(p);


    ui->progressBar->setVisible(false);
    ui->progressBar->setValue(0);


    scheme = new Scheme(ui->progressBar);
    scheme->setWorkspace(ui->workspaceLineEdit->text());
    scene->setWorkspace(ui->workspaceLineEdit->text());

    numberExec = 1;

    ui->comboBoxImages->addItem(tr(""));
    ui->comboBoxEndmembers->addItem(tr(""));
    ui->comboBoxAbundances->addItem(tr(""));
    ui->comboBoxAbunRMSE->addItem(tr(""));
    ui->comboBoxEndRMSE->addItem(tr(""));
    ui->comboBoxImageRMSE->addItem(tr(""));
    ui->comboBoxImageComparedRMSE->addItem(tr(""));
    ui->comboBoxRefMatching->addItem(tr(""));
    ui->comboBoxSigMatching->addItem(tr(""));

    imageSet.insert(tr(""));
    endmemberSet.insert(tr(""));
    abundanceSet.insert(tr(""));
    rmseImageSet.insert(tr(""));

    ui->endmemberPlot->xAxis->setLabel("Wavelength");
    ui->endmemberPlot->yAxis->setLabel("Reflectance");

    ui->zPlot->xAxis->setLabel("Wavelength");
    ui->zPlot->yAxis->setLabel("Reflectance");

    ui->plotRef->xAxis->setLabel("Wavelength");
    ui->plotRef->yAxis->setLabel("Reflectance");

    ui->plotSig->xAxis->setLabel("Wavelength");
    ui->plotSig->yAxis->setLabel("Reflectance");

    ui->stackedWidget->setCurrentIndex(0);
    ui->tabWidget->setCurrentIndex(0);

    ui->matchingButton->setEnabled(false);
    ui->comboBoxAuto->addItem("Manual");
    ui->comboBoxAuto->addItem("Automatic");
    ui->comboBoxMatching->addItem("Average");
    ui->comboBoxMatching->addItem("Minimun");
    //ui->comboBoxMatching->addItem("Optimus");
    ui->comboBoxMatching->setEnabled(false);
    ui->helpMatchingButton->setToolTip("MINIMUN:\nChoose the most similar reference based on minimal angle for the first signature, then the second, third and so on.\n\nAVERAGE:\nChoose the most similar set of references based on the average angle between different executions of the Minimun matching initially randomly sorted.");

    consoleD = new ConsoleLogDialog();

    //**********************************************************************

    //ACTIONS***************************************************************
    QObject::connect(ui->pageWidget, SIGNAL(currentChanged(int)), this, SLOT(loadOperators()));
    QObject::connect(scene, SIGNAL(itemInserted(EditorItem*)), this, SLOT(itemInserted(EditorItem*)));
    QObject::connect(scene, SIGNAL(itemClicked(EditorItem*)), this, SLOT(itemClicked(EditorItem*)));
    QObject::connect(scene, SIGNAL(imageInserted(NewImageDialog*, EditorImageItem*)), this, SLOT(imageInserted(NewImageDialog*, EditorImageItem*)));
    QObject::connect(ui->actionAbout_HyperMix, SIGNAL(triggered()), this, SLOT(showHelp()));
    QObject::connect(ui->actionSet_Workspace, SIGNAL(triggered()), this,SLOT(openWorkspaceD()));
    QObject::connect(workspaceD, SIGNAL(workspaceChanged()), this, SLOT(changeWorkspace()));
    QObject::connect(zoomIn, SIGNAL(triggered()), this, SLOT(zoomInInc()));
    QObject::connect(zoomOut, SIGNAL(triggered()), this, SLOT(zoomOutInc()));
    QObject::connect(clearOutput, SIGNAL(triggered()), this, SLOT(clear()));
    QObject::connect(ui->actionDiagram, SIGNAL(triggered()), this, SLOT(diagramPage()));
    QObject::connect(ui->actionResults, SIGNAL(triggered()), this, SLOT(resultsPage()));
    QObject::connect(ui->actionBlog, SIGNAL(triggered()), this, SLOT(blogPage()));

    QObject::connect(scheme, SIGNAL(newImageResult(QString)), this, SLOT(newImageResult(QString)));
    QObject::connect(scheme, SIGNAL(newEndmemberResult(QString)), this, SLOT(newEndmemberResult(QString)));
    QObject::connect(scheme, SIGNAL(newAbundanceResult(QString)), this, SLOT(newAbundanceResult(QString)));

    QObject::connect(ui->comboBoxImages, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentImageResultChange(QString)));
    QObject::connect(ui->comboBoxEndmembers, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentEndmemberResultChange(QString)));
    QObject::connect(ui->comboBoxAbundances, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentAbundanceResultChange(QString)));


    QObject::connect(ui->sliderBands, SIGNAL(valueChanged(int)), this, SLOT(paintBand(int)));
    QObject::connect(ui->comboBoxEnd, SIGNAL(activated(int)), this, SLOT(paintEnd(int)));
    QObject::connect(ui->comboBoxAbun, SIGNAL(activated(int)), this, SLOT(paintAbundance(int)));

    QObject::connect(ui->toolButtonLoadEndmembers, SIGNAL(clicked()), this, SLOT(loadEndmemberFile()));
    QObject::connect(ui->toolButtonLoadAbundances, SIGNAL(clicked()), this, SLOT(loadAbundanceFile()));

    QObject::connect(ui->horizontalSliderBandsRecon, SIGNAL(valueChanged(int)), this, SLOT(paintRecon(int)));
    QObject::connect(ui->toolButtonLoadEndRMSE, SIGNAL(clicked()), this, SLOT(loadEndmemberFileRMSE()));
    QObject::connect(ui->toolButtonLoadAbunRMSE, SIGNAL(clicked()), this, SLOT(loadAbundanceFileRMSE()));
    QObject::connect(ui->toolButtonLoadImageRMSE, SIGNAL(clicked()), this, SLOT(loadImageFileRMSE()));
    QObject::connect(ui->toolButtonLoadImageComparedRMSE, SIGNAL(clicked()), this, SLOT(loadImageComFileRMSE()));
    QObject::connect(ui->RMSEButton, SIGNAL(clicked()), this, SLOT(RMSE()));

    QObject::connect(ui->comboBoxEndRMSE, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentEndmemberRMSEChange(QString)));
    QObject::connect(ui->comboBoxAbunRMSE, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentAbundanceRMSEChange(QString)));
    QObject::connect(ui->comboBoxImageRMSE, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentImageRMSEChange(QString)));
    QObject::connect(ui->comboBoxImageComparedRMSE, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentImageComRMSEChange(QString)));
    QObject::connect(ui->tabWidget_RMSE,SIGNAL(currentChanged(int)), this, SLOT(tabRMSEChanged(int)));

    QObject::connect(ui->toolButtonLoadImage, SIGNAL(clicked()), this, SLOT(loadImage()));
    QObject::connect(sceneImage, SIGNAL(zPoint(QPointF)), this, SLOT(paintZPoint(QPointF)));
    QObject::connect(sceneAbundance, SIGNAL(percentAbun(QPointF)), this, SLOT(getPercent(QPointF)));
    QObject::connect(sceneRMSE, SIGNAL(percentRMSE(QPointF)), this, SLOT(getPercentRMSE(QPointF)));

    QObject::connect(ui->toolButtonLoadRefMatching, SIGNAL(clicked()), this, SLOT(loadReferences()));
    QObject::connect(ui->toolButtonLoadSigMatching, SIGNAL(clicked()), this, SLOT(loadSignatures()));
    QObject::connect(ui->comboBoxRefMatching, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentReferencesChange(QString)));
    QObject::connect(ui->comboBoxSigMatching, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentSignaturesChange(QString)));
    QObject::connect(ui->resultsTableRef, SIGNAL(cellClicked(int,int)), this, SLOT(rowRefSelected(int,int)));
    QObject::connect(ui->resultsTableSig, SIGNAL(cellClicked(int,int)), this, SLOT(rowSigSelected(int,int)));
    QObject::connect(ui->matchingButton, SIGNAL(clicked()), this, SLOT(runSAD()));
    QObject::connect(ui->comboBoxAuto, SIGNAL(currentIndexChanged(QString)), this, SLOT(setAutoMatching(QString)));

    QObject::connect(ui->actionConsole, SIGNAL(triggered()), this, SLOT(openConsole()));

    QObject::connect(ui->saveRMSEButton, SIGNAL(clicked()), this, SLOT(saveRMSE()));
    QObject::connect(ui->saveImageButton, SIGNAL(clicked()), this, SLOT(saveImage()));
    QObject::connect(ui->saveAbundancesButton, SIGNAL(clicked()), this, SLOT(saveAbundances()));


    //**********************************************************************
}

void HyperMix2::resizeEvent(QResizeEvent* event)
{
   QMainWindow::resizeEvent(event);
   //Paint Color reference
   QImage imageR(1,255,QImage::Format_RGB888);
   int i,red, blue, green;
   QRgb value;
   for(i=254; i>=0; i--)
   {
       getColor(i, red, green, blue);
       value = qRgb(red,green,blue);
       imageR.setPixel(0,254-i,value);
   }

   QGraphicsScene *sceneR = new QGraphicsScene();
   sceneR->addPixmap(QPixmap::fromImage(imageR.scaled(ui->graphicsViewScaleBar->width(),ui->graphicsViewScaleBar->height())));

   ui->graphicsViewScaleBar->setScene(sceneR);
}

void HyperMix2::loadButtons()
{
    //Endmember Estimation***************************************
    QVBoxLayout *lEstimation = new QVBoxLayout();

    for(unsigned int i=0; i<MAXOPERATORS; i++)
    {
        buttonListEstimation.append(new QPushButton());
        buttonListEstimation.at(i)->setEnabled(false);
        buttonListEstimation.at(i)->setVisible(false);
        lEstimation->addWidget(buttonListEstimation.at(i));
        ui->pageEstimation->setLayout(lEstimation);
    }

    //Image Preprocessing***************************************
    QVBoxLayout *lPreprocessing = new QVBoxLayout();

    for(unsigned int i=0; i<MAXOPERATORS; i++)
    {
        buttonListPreprocessing.append(new QPushButton());
        buttonListPreprocessing.at(i)->setEnabled(false);
        buttonListPreprocessing.at(i)->setVisible(false);
        lPreprocessing->addWidget(buttonListPreprocessing.at(i));
        ui->pagePreprocessing->setLayout(lPreprocessing);
    }

    //Endmember Extraction***************************************
    QVBoxLayout *lExtraction = new QVBoxLayout();

    for(unsigned int i=0; i<MAXOPERATORS; i++)
    {
        buttonListExtraction.append(new QPushButton());
        buttonListExtraction.at(i)->setEnabled(false);
        buttonListExtraction.at(i)->setVisible(false);
        lExtraction->addWidget(buttonListExtraction.at(i));
        ui->pageExtraction->setLayout(lExtraction);
    }

    //Abundance Calculation***************************************
    QVBoxLayout *lAbundance = new QVBoxLayout();

    for(unsigned int i=0; i<MAXOPERATORS; i++)
    {
        buttonListAbundance.append(new QPushButton());
        buttonListAbundance.at(i)->setEnabled(false);
        buttonListAbundance.at(i)->setVisible(false);
        lAbundance->addWidget(buttonListAbundance.at(i));
        ui->pageAbundance->setLayout(lAbundance);
    }
}

void HyperMix2::loadToolBar()
{
    groupButton = new QButtonGroup(this);
    groupButton->setExclusive(false);

    QObject::connect(groupButton, SIGNAL(buttonClicked(int)), this, SLOT(updateCheckedButtons(int)));
    if(DEBUG_MODE == 0) deleteAction = new QAction(QIcon("/usr/share/hypermix-2.0/images/delete.png"),"Delete",this);
    else deleteAction = new QAction(QIcon("images/delete.png"),"Delete",this);
    deleteAction->setShortcut(tr("Delete"));
    deleteAction->setStatusTip(tr("Deletes item from diagram"));
    ui->mainToolBar->addAction(deleteAction);
    QObject::connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteItem()));

    ui->mainToolBar->addSeparator();

    pointerButton = new QToolButton;
    pointerButton->setCheckable(true);
    pointerButton->setChecked(true);
    if(DEBUG_MODE == 0) pointerButton->setIcon(QIcon("/usr/share/hypermix-2.0/images/pointer.png"));
    else  pointerButton->setIcon(QIcon("images/pointer.png"));
    pointerButton->setToolTip("Selects");
    pointerButton->setStatusTip(tr("Selects an item"));
    groupButton->addButton(pointerButton, SelectItem);

    ui->mainToolBar->addWidget(pointerButton);

    linePointerButton = new QToolButton;
    linePointerButton->setCheckable(true);
    if(DEBUG_MODE == 0) linePointerButton->setIcon(QIcon("/usr/share/hypermix-2.0/images/linepointer.png"));
    else linePointerButton->setIcon(QIcon("images/linepointer.png"));
    linePointerButton->setToolTip("Connector");
    linePointerButton->setStatusTip(tr("Connects two items"));
    groupButton->addButton(linePointerButton, InsertLine);
    ui->mainToolBar->addWidget(linePointerButton);

    ui->mainToolBar->addSeparator();

    if(DEBUG_MODE == 0) zoomIn = new QAction(QIcon("/usr/share/hypermix-2.0/images/zoomin.png"), "Zoom In", this);
    else zoomIn = new QAction(QIcon("images/zoomin.png"), "Zoom In", this);
    zoomIn->setStatusTip(tr("Makes bigger the diagram"));
    ui->mainToolBar->addAction(zoomIn);

    if(DEBUG_MODE == 0) zoomOut = new QAction(QIcon("/usr/share/hypermix-2.0/images/zoomout.png"), "Zoom out", this);
    else zoomOut = new QAction(QIcon("images/zoomout.png"), "Zoom out", this);
    zoomOut->setStatusTip(tr("Makes smaller the diagram"));
    ui->mainToolBar->addAction(zoomOut);

    ui->mainToolBar->addSeparator();


    background = new QComboBox();
    background->addItem(tr("Snow"));
    background->addItem(tr("Azure"));
    background->addItem(tr("Antique"));
    background->addItem(tr("Salmon"));
    background->addItem(tr("Granite"));
    QObject::connect(background, SIGNAL(currentIndexChanged(QString)), this, SLOT(currentBackgroundChanged(QString)));
    ui->mainToolBar->addWidget(background);

    ui->mainToolBar->addSeparator();

    if(DEBUG_MODE == 0) clearOutput = new QAction(QIcon("/usr/share/hypermix-2.0/images/clear.png"), "Clear", this);
    else  clearOutput = new QAction(QIcon("images/clear.png"), "Clear", this);
    clearOutput->setStatusTip(tr("Clean output widgets"));
    ui->mainToolBar->addAction(clearOutput);

    ui->mainToolBar->addSeparator();

    imageButton = new QToolButton;
    imageButton->setCheckable(true);
    imageButton->setChecked(false);
    if(DEBUG_MODE == 0) imageButton->setIcon(QIcon("/usr/share/hypermix-2.0/images/image.png"));
    else imageButton->setIcon(QIcon("images/image.png"));
    imageButton->setToolTip("New Image");
    imageButton->setStatusTip(tr("Inserts an image"));
    groupButton->addButton(imageButton, InsertImage);
    ui->mainToolBar->addWidget(imageButton);


    if(DEBUG_MODE == 0) compileAction = new QAction(QIcon("/usr/share/hypermix-2.0/images/play.png"),"Compile/Run",this);
    else compileAction = new QAction(QIcon("images/play.png"),"Compile/Run",this);
    compileAction->setShortcut(tr("Ctrl + r"));
    compileAction->setStatusTip(tr("Compiles and runs the diagram"));
    ui->mainToolBar->addAction(compileAction);
    QObject::connect(compileAction, SIGNAL(triggered()), this, SLOT(compileScheme()));

    numberExecutions = new QComboBox();
    numberExecutions->setEditable(true);
    for(int i=1; i<=10; i++)
        numberExecutions->addItem(tr("x").append(QString().setNum(i)));
    QIntValidator *execValidator = new QIntValidator(1,100,this);
    numberExecutions->setValidator(execValidator);
    QObject::connect(numberExecutions, SIGNAL(currentIndexChanged(QString)), this, SLOT(numberExecutionsChanged(QString)));
    ui->mainToolBar->addWidget(numberExecutions);

    //************************************************************
    itemMenu = new QMenu();

    QAction* setParamAct = new QAction(tr("Set Parameters..."), this);
    setParamAct->setStatusTip(tr("Set Operator Parameters"));
    if(DEBUG_MODE == 0) setParamAct->setIcon(QIcon("/usr/share/hypermix-2.0/images/parameter.png"));
    else setParamAct->setIcon(QIcon("images/parameter.png"));
    setParamAct->setIconVisibleInMenu(true);

    itemMenu->addAction(setParamAct);
    QObject::connect(setParamAct, SIGNAL(triggered()), this, SLOT(openParamDialog()));

    arrowMenu = new QMenu();

    QAction* transportParam = new QAction(tr("Set Parameters..."), this);
    transportParam->setStatusTip(tr("Transport Output Operator Parameters"));
    if(DEBUG_MODE == 0) transportParam->setIcon(QIcon("/usr/share/hypermix-2.0/images/parameter.png"));
    else transportParam->setIcon(QIcon("images/parameter.png"));
    transportParam->setIconVisibleInMenu(true);

    arrowMenu->addAction(transportParam);

    //************************************************************

}

void HyperMix2::openParamDialog()
{
    scene->openParameterDialog();
}


void HyperMix2::loadOperators()
{
    QString address;
    if(DEBUG_MODE == 0) address = "/usr/share/hypermix-2.0/bin";
    else address = "bin";
    QDir dir(address);
    QStringList list = dir.entryList();

    contOperators.fill(0, 4);

    groupOperators = new QButtonGroup(this);
    groupOperators->setExclusive(false);

    QObject::connect(groupOperators, SIGNAL(buttonClicked(int)), this, SLOT(updateCheckedOperators(int)));

    //CHECK CUDA DEVICE************************************************************************************
    cudaSupported = true;
    //*****************************************************************************************************
    for(int i=2; i<list.size();i++)
    {
        if(!list.at(i).contains(".dll"))
        {
            if(list.at(i).contains("1")) //Estimation Operator
            {
                buttonListEstimation.at(contOperators[0])->setVisible(true);
                if(cudaSupported || !list.at(i).contains("_CUDA")) buttonListEstimation.at(contOperators[0])->setEnabled(true);
                buttonListEstimation.at(contOperators[0])->setAccessibleName(list[i]);
                buttonListEstimation.at(contOperators[0])->setText(list[i].remove("1_"));
                if(list.at(i).contains("_CUDA"))
                {
                        buttonListEstimation.at(contOperators[0])->setText(buttonListEstimation.at(contOperators[0])->text().remove("_CUDA"));
                        if(DEBUG_MODE == 0) buttonListEstimation.at(contOperators[0])->setIcon(QIcon("/usr/share/hypermix-2.0/images/nvidiacudaicon3.png"));
                        else buttonListEstimation.at(contOperators[0])->setIcon(QIcon("images/nvidiacudaicon3.png"));
                }
                buttonListEstimation.at(contOperators[0])->setText(buttonListEstimation.at(contOperators[0])->text().replace("_", " "));
                buttonListEstimation.at(contOperators[0])->setText(buttonListEstimation.at(contOperators[0])->text().remove(".exe"));
                buttonListEstimation.at(contOperators[0])->setCheckable(true);
                groupOperators->addButton(buttonListEstimation.at(contOperators[0]),i-2);
                contOperators[0]++;
            }
            if(list.at(i).contains("2")) //Preprocessing Operator
            {
                buttonListPreprocessing.at(contOperators[1])->setVisible(true);
                if(cudaSupported || !list.at(i).contains("_CUDA")) buttonListPreprocessing.at(contOperators[1])->setEnabled(true);
                buttonListPreprocessing.at(contOperators[1])->setAccessibleName(list[i]);
                buttonListPreprocessing.at(contOperators[1])->setText(list[i].remove("2_"));
                if(list.at(i).contains("_CUDA"))
                {
                        buttonListPreprocessing.at(contOperators[1])->setText(buttonListPreprocessing.at(contOperators[1])->text().remove("_CUDA"));
                        if(DEBUG_MODE == 0) buttonListPreprocessing.at(contOperators[1])->setIcon(QIcon("/usr/share/hypermix-2.0/images/nvidiacudaicon3.png"));
                        else buttonListPreprocessing.at(contOperators[1])->setIcon(QIcon("images/nvidiacudaicon3.png"));
                }
                buttonListPreprocessing.at(contOperators[1])->setText(buttonListPreprocessing.at(contOperators[1])->text().replace("_", " "));
                buttonListPreprocessing.at(contOperators[1])->setText(buttonListPreprocessing.at(contOperators[1])->text().remove(".exe"));
                buttonListPreprocessing.at(contOperators[1])->setCheckable(true);
                groupOperators->addButton(buttonListPreprocessing.at(contOperators[1]),i-2);
                contOperators[1]++;
            }
            if(list.at(i).contains("3")) //Extraction Operator
            {
                buttonListExtraction.at(contOperators[2])->setVisible(true);
                if(cudaSupported || !list.at(i).contains("_CUDA")) buttonListExtraction.at(contOperators[2])->setEnabled(true);
                buttonListExtraction.at(contOperators[2])->setAccessibleName(list[i]);
                buttonListExtraction.at(contOperators[2])->setText(list[i].remove("3_"));
                if(list.at(i).contains("_CUDA"))
                {
                        buttonListExtraction.at(contOperators[2])->setText(buttonListExtraction.at(contOperators[2])->text().remove("_CUDA"));
                        if(DEBUG_MODE == 0) buttonListExtraction.at(contOperators[2])->setIcon(QIcon("/usr/share/hypermix-2.0/images/nvidiacudaicon3.png"));
                        else buttonListExtraction.at(contOperators[2])->setIcon(QIcon("images/nvidiacudaicon3.png"));
                }
                buttonListExtraction.at(contOperators[2])->setText(buttonListExtraction.at(contOperators[2])->text().replace("_", " "));
                buttonListExtraction.at(contOperators[2])->setText(buttonListExtraction.at(contOperators[2])->text().remove(".exe"));
                buttonListExtraction.at(contOperators[2])->setCheckable(true);
                groupOperators->addButton(buttonListExtraction.at(contOperators[2]),i-2);
                contOperators[2]++;
            }
            if(list.at(i).contains("4")) //Abundance Operator
            {
                buttonListAbundance.at(contOperators[3])->setVisible(true);
                if(cudaSupported || !list.at(i).contains("_CUDA")) buttonListAbundance.at(contOperators[3])->setEnabled(true);
                buttonListAbundance.at(contOperators[3])->setAccessibleName(list[i]);
                buttonListAbundance.at(contOperators[3])->setText(list[i].remove("4_"));
                if(list.at(i).contains("_CUDA"))
                {
                        buttonListAbundance.at(contOperators[3])->setText(buttonListAbundance.at(contOperators[3])->text().remove("_CUDA"));
                        if(DEBUG_MODE == 0) buttonListAbundance.at(contOperators[3])->setIcon(QIcon("/usr/share/hypermix-2.0/images/nvidiacudaicon3.png"));
                        else  buttonListAbundance.at(contOperators[3])->setIcon(QIcon("images/nvidiacudaicon3.png"));
                }
                buttonListAbundance.at(contOperators[3])->setText(buttonListAbundance.at(contOperators[3])->text().replace("_", " "));
                buttonListAbundance.at(contOperators[3])->setText(buttonListAbundance.at(contOperators[3])->text().remove(".exe"));
                buttonListAbundance.at(contOperators[3])->setCheckable(true);
                groupOperators->addButton(buttonListAbundance.at(contOperators[3]),i-2);
                contOperators[3]++;
            }
        }
    }
}

void HyperMix2::updateCheckedOperators(int id)
{
    QList<QAbstractButton *> buttons = groupOperators->buttons();

    foreach (QAbstractButton *button, buttons)
    {
        if (groupOperators->button(id) != button)
            button->setChecked(false);
    }

    scene->setMode(EditorScene::InsertItem);
}

void HyperMix2::updateCheckedButtons(int id)
{
    QList<QAbstractButton *> buttons = groupButton->buttons();

    foreach (QAbstractButton *button, buttons)
    {
        if (groupButton->button(id) != button)
            button->setChecked(false);
    }
    switch(id)
    {
        case InsertLine: scene->setMode(EditorScene::InsertLine); break;

        case SelectItem: scene->setMode(EditorScene::MoveItem); break;

        case InsertImage: scene->setMode(EditorScene::InsertImage); break;

        case InsertItem: scene->setMode(EditorScene::InsertItem); break;
    }
}

void HyperMix2::imageInserted(NewImageDialog *imageD, EditorImageItem* image)
{

    scene->setMode(EditorScene::MoveItem);
    scene->setSelectedAll();
    if(image != NULL)
    {
        if(groupButton->checkedButton() != NULL)
        {
            image->setOperatorName(tr("0_").append(imageD->getNameImage()));
            image->setOperatorFilename(imageD->getFile());

            scheme->addItem(image);

            groupButton->button(groupButton->checkedId())->setChecked(false);
            groupButton->button(SelectItem)->setChecked(true);

            QStringList images = scheme->getListActiveImages();
            foreach (QString image, images)
            {
                if(!rmseImageSet.contains(image))
                {
                    rmseImageSet.insert(image);
                    ui->comboBoxImageRMSE->addItem(image);
                }
            }
        }
    }
    else
    {
        groupButton->button(groupButton->checkedId())->setChecked(false);
        groupButton->button(SelectItem)->setChecked(true);
    }
}


void HyperMix2::itemInserted(EditorItem *item)
{
    QString op;
    scene->setMode(EditorScene::MoveItem);

    scene->setSelectedAll();

    if(groupButton->checkedButton() != NULL)
    {
        groupButton->button(groupButton->checkedId())->setChecked(false);
        groupButton->button(SelectItem)->setChecked(true);
    }
    if(groupOperators->checkedButton() != NULL)
    {

        item->setOperatorName(groupOperators->checkedButton()->accessibleName());

        op.clear();
        op = groupOperators->checkedButton()->text();
        item->getLabel()->setPlainText(op);

        if(item->getOperatorName().contains("1_"))
            item->setType(1);
        if(item->getOperatorName().contains("2_"))
            item->setType(2);
        if(item->getOperatorName().contains("3_"))
            item->setType(3);
        if(item->getOperatorName().contains("4_"))
            item->setType(4);

        if(item->getOperatorName().contains("_CUDA")) item->setBrush(QColor(176,224,230));
        else item->setBrush(QColor(238,221,130));

        QFont f = item->getLabel()->font();
        f.setBold(true);
        f.setPointSize(16);
        item->getLabel()->setFont(f);

        scheme->addItem(item);

        groupOperators->button(groupOperators->checkedId())->setChecked(false);
        groupButton->button(SelectItem)->setChecked(true);
    }
}

void HyperMix2::itemClicked(EditorItem *item)
{

    InfoDialog* infoD = new InfoDialog(item->getOperatorName());
    infoD->loadInfo();
    infoD->show();
}

void HyperMix2::wheelEvent(QWheelEvent *event)
{
    int numDegrees = event->delta() / 8;
    int numSteps = abs(numDegrees) / 15;
    int i;


    if (event->orientation() == Qt::Vertical)
    {
        for(i=0; i<numSteps; i++)
            if(numDegrees > 0) zoomInInc();
            else zoomOutInc();
    }
}

void HyperMix2::deleteItem()
{
    if(!scene->selectedItems().isEmpty())
    {
        if(qgraphicsitem_cast<EditorItem *>(scene->selectedItems().first()) != 0)
            scheme->removeItem(qgraphicsitem_cast<EditorItem *>(scene->selectedItems().first()));
        else if(qgraphicsitem_cast<EditorImageItem *>(scene->selectedItems().first()) != 0)
            scheme->removeImage(qgraphicsitem_cast<EditorImageItem *>(scene->selectedItems().first()));
        scene->deleteItemSelected();
    }
}

void HyperMix2::compileScheme()
{

    if(scheme->compileScheme(consoleD) == 0)
    {
        scheme->runScheme(numberExec,consoleD);
    }
}

void HyperMix2::showHelp()
{
    helpD->start();
}

void HyperMix2::openWorkspaceD()
{
    workspaceD->exec();
}

void HyperMix2::changeWorkspace()
{
    ui->workspaceLineEdit->setText(workspaceD->getWorkspace());
    scheme->setWorkspace(ui->workspaceLineEdit->text());
    scene->setWorkspace(ui->workspaceLineEdit->text());
}

void HyperMix2::openConsole()
{
    consoleD->loadLog();
    consoleD->show();
}

void HyperMix2::clear()
{
    scheme->cleanScheme();

    if(ui->comboBoxRefMatching->currentIndex() != 0)  ui->comboBoxRefMatching->setCurrentIndex(0);
    if(ui->comboBoxSigMatching->currentIndex() != 0)  ui->comboBoxSigMatching->setCurrentIndex(0);

}

void HyperMix2::zoomInInc()
{
    if(ui->stackedWidget->currentIndex() == 0) ui->graphicsView->scale(SCALEFAC, SCALEFAC);
    if(ui->stackedWidget->currentIndex() == 1)
    {
        ui->graphicsViewImages->scale(SCALEFAC,SCALEFAC);
        ui->graphicsViewAbundances->scale(SCALEFAC,SCALEFAC);
        ui->graphicsViewRMSE->scale(SCALEFAC,SCALEFAC);
        ui->graphicsViewRecon->scale(SCALEFAC,SCALEFAC);
    }
}

void HyperMix2::zoomOutInc()
{

    if(ui->stackedWidget->currentIndex() == 0) ui->graphicsView->scale(1.0/SCALEFAC, 1.0/SCALEFAC);
    if(ui->stackedWidget->currentIndex() == 1)
    {
        ui->graphicsViewImages->scale(1.0/SCALEFAC,1.0/SCALEFAC);
        ui->graphicsViewAbundances->scale(1.0/SCALEFAC,1.0/SCALEFAC);
        ui->graphicsViewRMSE->scale(1.0/SCALEFAC,1.0/SCALEFAC);
        ui->graphicsViewRecon->scale(1.0/SCALEFAC,1.0/SCALEFAC);
    }
}

void HyperMix2::currentBackgroundChanged(QString backg)
{

    QPalette p = ui->graphicsView->palette();

    if(backg == "Azure") p.setColor(QPalette::Base, QColor(240,255,255));
    if(backg == "Antique") p.setColor(QPalette::Base, QColor(250,235,215));
    if(backg == "Granite") p.setColor(QPalette::Base, QColor(190,190,190));
    if(backg == "Salmon") p.setColor(QPalette::Base, QColor(255,160,172));
    if(backg == "Snow") p.setColor(QPalette::Base, Qt::white);

    scene->setBackgroundcolor(backg);
    ui->graphicsView->setPalette(p);

}


void HyperMix2::numberExecutionsChanged(QString number)
{
    handleExecutionsChange();
}

void HyperMix2::handleExecutionsChange()
{
    numberExec = numberExecutions->currentText().remove("x").toInt();
}

void HyperMix2::diagramPage()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void HyperMix2::resultsPage()
{
    ui->stackedWidget->setCurrentIndex(1);
}

void HyperMix2::blogPage()
{
    ui->stackedWidget->setCurrentIndex(2);
    ui->webView->load(QUrl("http://hypercomphypermix.blogspot.com.br/"));
}

void HyperMix2::newImageResult(QString image)
{
    if(!imageSet.contains(image))
    {
        ui->comboBoxImages->addItem(image);
        imageSet.insert(image);
    }
}

void HyperMix2::newEndmemberResult(QString endmembers)
{
    if(!endmemberSet.contains(endmembers))
    {
        ui->comboBoxEndmembers->addItem(endmembers);
        ui->comboBoxEndRMSE->addItem(endmembers);
        ui->comboBoxRefMatching->addItem(endmembers);
        ui->comboBoxSigMatching->addItem(endmembers);
        endmemberSet.insert(endmembers);
    }
}

void HyperMix2::newAbundanceResult(QString abundance)
{
    if(!abundanceSet.contains(abundance))
    {
        ui->comboBoxAbundances->addItem(abundance);
        ui->comboBoxAbunRMSE->addItem(abundance);
        abundanceSet.insert(abundance);
    }
}

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
void HyperMix2::cleanString(char *cadena, char *out)
{
    int i,j;
    for( i = j = 0; cadena[i] != 0;++i)
    {
        if(isalnum(cadena[i])||cadena[i]=='{'||cadena[i]=='.'||cadena[i]==',')
        {
            out[j]=cadena[i];
            j++;
        }
    }
    for( i = j; out[i] != 0;++i)
        out[j]=0;
}

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
int HyperMix2::readHeader1(const char* filename, int *lines, int *samples, int *bands, int *dataType,
        char* interleave, int *byteOrder, char* waveUnit)
{
    FILE *fp;
    char line[MAXLINE]="";
    char value [MAXLINE]="";

    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
        while(fgets(line, MAXLINE, fp)!='\0')
        {
            //Samples
            if(strstr(line, "samples")!=NULL && samples !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *samples = atoi(value);
            }

            //Lines
            if(strstr(line, "lines")!=NULL && lines !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *lines = atoi(value);
            }

            //Bands
            if(strstr(line, "bands")!=NULL && bands !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *bands = atoi(value);
            }

            //Interleave
            if(strstr(line, "interleave")!=NULL && interleave !=NULL)
            {
                cleanString(strstr(line, "="),value);
                strcpy(interleave,value);
            }

            //Data Type
            if(strstr(line, "data type")!=NULL && dataType !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *dataType = atoi(value);
            }

            //Byte Order
            if(strstr(line, "byte order")!=NULL && byteOrder !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *byteOrder = atoi(value);
            }

            //Wavelength Unit
            if(strstr(line, "wavelength unit")!=NULL && waveUnit !=NULL)
            {
                cleanString(strstr(line, "="),value);
                strcpy(waveUnit,value);
            }

        }
        fclose(fp);
        return 0;
    }
    else
        return -2; //No file found
}

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
int HyperMix2::readHeader2(const char* filename, double* wavelength)
{
    FILE *fp;
    char line[MAXLINE]="";
    char value [MAXLINE]="";

    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
        while(fgets(line, MAXLINE, fp)!='\0')
        {
            //Wavelength
            if(strstr(line, "wavelength =")!=NULL && wavelength !=NULL)
            {
                char strAll[100000]=" ";
                char *pch;
                int cont = 0;
                do
                {
                    fgets(line, 200, fp);
                    cleanString(line,value);
                    strcat(strAll,value);
                } while(strstr(line, "}")==NULL);

                pch = strtok(strAll,",");

                while (pch != NULL)
                {
                    if(QString::fromUtf8(pch).toFloat() < 10) wavelength[cont]= QString::fromUtf8(pch).toFloat()*1000;
                    else wavelength[cont]= QString::fromUtf8(pch).toFloat();
                    pch = strtok (NULL, ",");
                    cont++;
                }
            }

        }
        fclose(fp);
        return 0;
    }
    else
        return -2; //No file found
}

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
int HyperMix2::loadImage(const char* filename, double* image, int lines, int samples, int bands, int dataType, char* interleave)
{

    FILE *fp;
    short int *tipo_short_int;
    float *tipo_float;
    double * tipo_double;
    unsigned int *tipo_uint;
    int i, j, k, op;
    long int lines_samples = lines*samples;


    if ((fp=fopen(filename,"rb"))!=NULL)
    {

        fseek(fp,0L,SEEK_SET);
        tipo_float = (float*)malloc(lines_samples*bands*sizeof(float));
        switch(dataType)
        {
            case 2:
                tipo_short_int = (short int*)malloc(lines_samples*bands*sizeof(short int));
                fread(tipo_short_int,1,(sizeof(short int)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                    tipo_float[i]=(float)tipo_short_int[i];
                free(tipo_short_int);
                break;

            case 4:
                fread(tipo_float,1,(sizeof(float)*lines_samples*bands),fp);
                break;

            case 5:
                tipo_double = (double*)malloc(lines_samples*bands*sizeof(double));
                fread(tipo_double,1,(sizeof(double)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                    tipo_float[i]=(float)tipo_double[i];
                free(tipo_double);
                break;

            case 12:
                tipo_uint = (unsigned int*)malloc(lines_samples*bands*sizeof(unsigned int));
                fread(tipo_uint,1,(sizeof(unsigned int)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                    tipo_float[i]=(float)tipo_uint[i];
                free(tipo_uint);
                break;

        }
        fclose(fp);

        if(interleave == NULL)
            op = 0;
        else
        {
            if(strcmp(interleave, "bsq") == 0) op = 0;
            if(strcmp(interleave, "bip") == 0) op = 1;
            if(strcmp(interleave, "bil") == 0) op = 2;
        }


        switch(op)
        {
            case 0:
                for(i=0; i<lines*samples*bands; i++)
                    image[i] = tipo_float[i];
                break;

            case 1:
                for(i=0; i<bands; i++)
                    for(j=0; j<lines*samples; j++)
                        image[i*lines*samples + j] = tipo_float[j*bands + i];
                break;

            case 2:
                for(i=0; i<lines; i++)
                    for(j=0; j<bands; j++)
                        for(k=0; k<samples; k++)
                            image[j*lines*samples + (i*samples+k)] = tipo_float[k+samples*(i*bands+j)];
                break;
        }
        free(tipo_float);
        return 0;
    }
    return -2;
}

void HyperMix2::currentEndmemberResultChange(QString file)
{
    ui->comboBoxEndRMSE->setCurrentIndex(ui->comboBoxEndmembers->currentIndex());
    clearEndmemberResultCanvas();
    if(file.compare(tr("")) != 0)
    {        
        if(endmemberResult != NULL) delete endmemberResult;
        QString header = file;
        header.append(".hdr");

        if(interleaveE != NULL) free(interleaveE);
        interleaveE = (char*)malloc(MAXSTR*sizeof(char));
        if(waveUnitE != NULL) free(waveUnitE);
        waveUnitE = (char*)malloc(MAXSTR*sizeof(char));
        int error = readHeader1(header.toStdString().c_str(), &linesE, &samplesE, &bandsE, &dataTypeE, interleaveE , &byteOrderE, waveUnitE);
        if(error != 0)
            return;

        if(wavelengthE != NULL) free(wavelengthE);
        wavelengthE = (double*)malloc(bandsE*sizeof(double));
        error = readHeader2(header.toStdString().c_str(), wavelengthE);
        if(error != 0)
            return;

        //if(!emptyEnd){ free(endmemberResult); emptyEnd = true;}
        endmemberResult = (double*)malloc(linesE*samplesE*bandsE*sizeof(double));
        //emptyEnd = false;

        error = loadImage(file.toStdString().c_str(), endmemberResult, linesE, samplesE, bandsE, dataTypeE, interleaveE);
        if(error != 0)
            return;
        paintEnd(0);
    }
}

void HyperMix2::currentEndmemberRMSEChange(QString file)
{
    ui->comboBoxEndmembers->setCurrentIndex(ui->comboBoxEndRMSE->currentIndex());
    clearEndmemberResultCanvas();
    if(file.compare(tr("")) != 0)
    {
        if(endmemberResult != NULL) delete endmemberResult;

        QString header = file;
        header.append(".hdr");

        if(interleaveE != NULL) free(interleaveE);
        interleaveE = (char*)malloc(MAXSTR*sizeof(char));
        if(waveUnitE != NULL) free(waveUnitE);
        waveUnitE = (char*)malloc(MAXSTR*sizeof(char));
        int error = readHeader1(header.toStdString().c_str(), &linesE, &samplesE, &bandsE, &dataTypeE, interleaveE, &byteOrderE, waveUnitE);
        if(error != 0)
            return;

        if(wavelengthE != NULL) free(wavelengthE);
        wavelengthE = (double*)malloc(bandsE*sizeof(double));
        error = readHeader2(header.toStdString().c_str(), wavelengthE);
        if(error != 0)
            return;

        //if(!emptyEnd){ free(endmemberResult); emptyEnd = true;}
        endmemberResult = (double*)malloc(linesE*samplesE*bandsE*sizeof(double));
        //emptyEnd = false;

        error = loadImage(file.toStdString().c_str(), endmemberResult, linesE, samplesE, bandsE, dataTypeE, interleaveE);
        if(error != 0)
            return;
        paintEnd(0);
    }
    checkRMSEValues();
}

void HyperMix2::currentImageResultChange(QString file)
{
    clearImageResultCanvas();

    if(file.compare(tr("")) != 0)
    {
        ui->sliderBands->setEnabled(true);
        QString header = file;
        header.append(".hdr");

        if(interleaveI != NULL) free(interleaveI);
        interleaveI = (char*)malloc(MAXSTR*sizeof(char));
        if(waveUnitI != NULL) free(waveUnitI);
        waveUnitI = (char*)malloc(MAXSTR*sizeof(char));
        int error = readHeader1(header.toStdString().c_str(), &linesI, &samplesI, &bandsI, &dataTypeI, interleaveI, &byteOrderI, waveUnitI);
        if(error != 0)
            return;

        if(wavelengthI != NULL) free(wavelengthI);
        wavelengthI = (double*)malloc(bandsI*sizeof(double));
        error = readHeader2(header.toStdString().c_str(), wavelengthI);
        if(error != 0)
            return;

        if(imageResult != NULL) free(imageResult);
        imageResult = (double*)malloc(linesI*samplesI*bandsI*sizeof(double));

        error = loadImage(file.toStdString().c_str(), imageResult, linesI, samplesI, bandsI, dataTypeI, interleaveI);
        if(error != 0)
            return;

        ui->sliderBands->setEnabled(true);
        ui->sliderBands->setMinimum(0);
        ui->sliderBands->setMaximum(bandsI-1);
        ui->sliderBands->setSliderPosition(0);
        ui->sliderBands->setSingleStep(1);

        paintBand(0);
    }
}

void HyperMix2::currentAbundanceResultChange(QString file)
{
    ui->comboBoxAbunRMSE->setCurrentIndex(ui->comboBoxAbundances->currentIndex());
    clearAbundanceResultCanvas();

    if(file.compare(tr("")) != 0)
    {

        QString header = file;
        header.append(".hdr");

        if(interleaveA != NULL) free(interleaveA);
        interleaveA = (char*)malloc(MAXSTR*sizeof(char));
        int error = readHeader1(header.toStdString().c_str(), &linesA, &samplesA, &bandsA, &dataTypeA, interleaveA, NULL, NULL);
        if(error != 0)
            return;

        if(abundanceResult != NULL) free(abundanceResult);
        abundanceResult = (double*)malloc(linesA*samplesA*bandsA*sizeof(double));

        error = loadImage(file.toStdString().c_str(), abundanceResult, linesA, samplesA, bandsA, dataTypeA, interleaveA);
        if(error != 0)
            return;

        paintAbundance(0);
    }
}

void HyperMix2::currentAbundanceRMSEChange(QString file)
{
    ui->comboBoxAbundances->setCurrentIndex(ui->comboBoxAbunRMSE->currentIndex());
    clearAbundanceResultCanvas();
    if(file.compare(tr("")) != 0)
    {

        QString header = file;
        header.append(".hdr");

        if(interleaveA != NULL) free(interleaveA);
        interleaveA = (char*)malloc(MAXSTR*sizeof(char));
        int error = readHeader1(header.toStdString().c_str(), &linesA, &samplesA, &bandsA, &dataTypeA, interleaveA, NULL, NULL);
        if(error != 0)
            return;

        if(abundanceResult != NULL) free(abundanceResult);
        abundanceResult = (double*)malloc(linesA*samplesA*bandsA*sizeof(double));

        error = loadImage(file.toStdString().c_str(), abundanceResult, linesA, samplesA, bandsA, dataTypeA, interleaveA);
        if(error != 0)
            return;

        paintAbundance(0);
    }
    checkRMSEValues();
}

void HyperMix2::currentImageRMSEChange(QString file)
{
    clearImageRMSEResultCanvas();
    if(file.compare(tr("")) != 0)
    {

        QString header = file;
        header.append(".hdr");

        if(interleaveRO != NULL) free(interleaveRO);
        interleaveRO = (char*)malloc(MAXSTR*sizeof(char));
        int error = readHeader1(header.toStdString().c_str(), &linesRO, &samplesRO, &bandsRO, &dataTypeRO, interleaveRO, NULL, NULL);
        if(error != 0)
            return;

        if(imageReconOri != NULL) free(imageReconOri);
        imageReconOri = (double*)malloc(linesRO*samplesRO*bandsRO*sizeof(double));

        error = loadImage(file.toStdString().c_str(), imageReconOri, linesRO, samplesRO, bandsRO, dataTypeRO, interleaveRO);
        if(error != 0)
            return;

    }
    checkRMSEValues();

}

void HyperMix2::tabRMSEChanged(int tab)
{
    checkRMSEValues();
}

void HyperMix2::currentImageComRMSEChange(QString file)
{

    if(ui->tabWidget_RMSE->currentIndex() == 1)
    {
        clearImageRMSEResultCanvas();
        if(file.compare(tr("")) != 0)
        {

            QString header = file;
            header.append(".hdr");

            if(interleaveR != NULL) free(interleaveR);
            interleaveR = (char*)malloc(MAXSTR*sizeof(char));
            int error = readHeader1(header.toStdString().c_str(), &linesR, &samplesR, &bandsR, &dataTypeR, interleaveR, NULL, NULL);
            if(error != 0)
                return;

            if(imageRecon != NULL) free(imageRecon);
            imageRecon = (double*)malloc(linesR*samplesR*bandsR*sizeof(double));

            error = loadImage(file.toStdString().c_str(), imageRecon, linesR, samplesR, bandsR, dataTypeR, interleaveR);
            if(error != 0)
                return;

        }
        checkRMSEValues();
    }
}


void HyperMix2::currentReferencesChange(QString file)
{

    int i, j;
    clearSAD();

    if(file.compare(tr("")) != 0)
    {
        if(endmemberResultRef != NULL) delete endmemberResultRef;
        QString header = file;
        header.append(".hdr");

        if(interleaveRef != NULL) free(interleaveRef);
        interleaveRef = (char*)malloc(MAXSTR*sizeof(char));
        if(waveUnitRef != NULL) free(waveUnitRef);
        waveUnitRef = (char*)malloc(MAXSTR*sizeof(char));
        int error = readHeader1(header.toStdString().c_str(), &linesRef, &samplesRef, &bandsRef, &dataTypeRef, interleaveRef , &byteOrderRef, waveUnitRef);
        if(error != 0)
            return;

        if(wavelengthRef != NULL) free(wavelengthRef);
        wavelengthRef = (double*)malloc(bandsRef*sizeof(double));
        error = readHeader2(header.toStdString().c_str(), wavelengthRef);
        if(error != 0)
            return;

        //if(endmemberResultRef != NULL) free(endmemberResultRef);
        endmemberResultRef = (double*)malloc(linesRef*samplesRef*bandsRef*sizeof(double));

        error = loadImage(file.toStdString().c_str(), endmemberResultRef, linesRef, samplesRef, bandsRef, dataTypeRef, interleaveRef);
        if(error != 0)
            return;

        //Display the results
        ui->resultsTableRef->setRowCount(linesRef);
        ui->resultsTableRef->setColumnCount(bandsRef);
        ui->resultsTableRef->verticalHeader()->setVisible(true);
        ui->resultsTableRef->horizontalHeader()->setVisible(true);

        for(i=0; i<linesRef;i ++)
        {
            for(j=0; j<bandsRef;j ++)
            {
                QTableWidgetItem *item = new QTableWidgetItem(QString::number(endmemberResultRef[j*linesRef*samplesRef + i]));
                item->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEnabled);
                ui->resultsTableRef->setItem(i,j,item);
            }
        }
    }
}


void HyperMix2::currentSignaturesChange(QString file)
{
    int i,j;
    clearSAD();

    if(file.compare(tr("")) != 0)
    {
        if(endmemberResultSig != NULL) delete endmemberResultSig;
        QString header = file;
        header.append(".hdr");

        if(interleaveSig != NULL) free(interleaveSig);
        interleaveSig = (char*)malloc(MAXSTR*sizeof(char));
        if(waveUnitSig != NULL) free(waveUnitSig);
        waveUnitSig = (char*)malloc(MAXSTR*sizeof(char));
        int error = readHeader1(header.toStdString().c_str(), &linesSig, &samplesSig, &bandsSig, &dataTypeSig, interleaveSig , &byteOrderSig, waveUnitSig);
        if(error != 0)
            return;

        if(wavelengthSig != NULL) free(wavelengthSig);
        wavelengthSig = (double*)malloc(bandsSig*sizeof(double));
        error = readHeader2(header.toStdString().c_str(), wavelengthSig);
        if(error != 0)
            return;

        //if(endmemberResultSig != NULL) free(endmemberResultSig);
        endmemberResultSig = (double*)malloc(linesSig*samplesSig*bandsSig*sizeof(double));

        error = loadImage(file.toStdString().c_str(), endmemberResultSig, linesSig, samplesSig, bandsSig, dataTypeSig, interleaveSig);
        if(error != 0)
            return;

        //Display the results
        ui->resultsTableSig->setRowCount(linesSig);
        ui->resultsTableSig->setColumnCount(bandsSig);
        ui->resultsTableSig->verticalHeader()->setVisible(true);
        ui->resultsTableSig->horizontalHeader()->setVisible(true);

        for(i=0; i<linesSig;i ++)
        {
            for(j=0; j<bandsSig;j ++)
            {
                QTableWidgetItem *item = new QTableWidgetItem(QString::number(endmemberResultSig[j*linesSig*samplesSig + i]));
                item->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEnabled);
                ui->resultsTableSig->setItem(i,j,item);
            }
        }
    }
}


void HyperMix2::paintBand(int numBand)
{

    if(numBand<bandsI)
    {
        int i,j, cont = 0;
        double min, max;
        double* bandVector = (double*)malloc(linesI*samplesI*sizeof(double));

        //Extract the band that we wat to display
        //Find maximun and minimun value of the band
        min = max = imageResult[0];
        for(j = numBand * linesI * samplesI; j < (numBand + 1) * linesI * samplesI; j++)
        {
            bandVector[cont] = imageResult[j];
            cont++;
            if(imageResult[j] < min) min = imageResult[j];
            if(imageResult[j] > max) max = imageResult[j];
        }

        QRgb value;
        int color;

        if(imageRMSE != NULL) free(imageRGB);
        imageRGB = new QImage(samplesI,linesI,QImage::Format_RGB888);

        for(int i=0; i<samplesI*linesI; i++)
        {
            color = ((bandVector[i]-min)*255)/(max-min);
            value = qRgb(color,color,color);
            imageRGB->setPixel(i%samplesI,i/samplesI,value);
        }

        sceneImage->setWidth(samplesI);
        sceneImage->setHeight(linesI);
        sceneImage->addPixmap(QPixmap::fromImage(*imageRGB));
        sceneImage->setSceneRect(0, 0, imageRGB->width(), imageRGB->height());

        //Paint RGB Composition 460 - 530 - 630

        QImage imageCom(samplesI,linesI,QImage::Format_RGB888);
        bool red= false, blue= false, green= false;
        double* redBand = (double*)malloc(linesI*samplesI*sizeof(double));
        double* greenBand = (double*)malloc(linesI*samplesI*sizeof(double));
        double* blueBand = (double*)malloc(linesI*samplesI*sizeof(double));

        int colorR, colorG, colorB;

        for(i=0; i<bandsI; i++)
        {
            if(wavelengthI[i] > 440 && wavelengthI[i] < 470 && !blue)
            {
                blue = true;
                cont = 0;
                for(j = i * linesI * samplesI; j < (i + 1) * linesI * samplesI; j++)
                {
                    blueBand[cont] = imageResult[j];
                    cont++;
                    if(imageResult[j] < min) min = imageResult[j];
                    if(imageResult[j] > max) max = imageResult[j];
                }
            }

            if(wavelengthI[i] > 500 && wavelengthI[i] < 550 && !green)
            {
                green = true;
                cont = 0;
                for(j = i * linesI * samplesI; j < (i + 1) * linesI * samplesI; j++)
                {
                    greenBand[cont] = imageResult[j];
                    cont++;
                    if(imageResult[j] < min) min = imageResult[j];
                    if(imageResult[j] > max) max = imageResult[j];
                }
            }

            if(wavelengthI[i] > 600 && wavelengthI[i] < 650 && !red)
            {
                red = true;
                cont = 0;
                for(j = i * linesI * samplesI; j < (i + 1) * linesI * samplesI; j++)
                {
                    redBand[cont] = imageResult[j];
                    cont++;
                    if(imageResult[j] < min) min = imageResult[j];
                    if(imageResult[j] > max) max = imageResult[j];
                }
            }
        }

        for(i=0; i<samplesI*linesI; i++)
        {
            colorR = ((redBand[i]-min)*255)/(max-min);
            colorG = ((greenBand[i]-min)*255)/(max-min);
            colorB = ((blueBand[i]-min)*255)/(max-min);
            value = qRgb(colorR,colorG,colorB);
            imageCom.setPixel(i%samplesI,i/samplesI,value);
        }

        QGraphicsScene *sceneRGB = new QGraphicsScene();
        sceneRGB->addPixmap(QPixmap::fromImage(imageCom));
        ui->graphicsViewRGB->setScene(sceneRGB);
        ui->graphicsViewRGB->fitInView(QRectF(0, 0, samplesI, linesI), Qt::KeepAspectRatio);

        delete bandVector;
        delete redBand;
        delete greenBand;
        delete blueBand;
    }
}

void HyperMix2::paintRecon(int numBand)
{

    if(numBand<bandsR)
    {
        int j, cont = 0;
        double min, max;
        double* bandVector = (double*)malloc(linesR*samplesR*sizeof(double));

        //Extract the band that we wat to display
        //Find maximun and minimun value of the band
        min = max = imageRecon[0];
        for(j = numBand * linesR * samplesR; j < (numBand + 1) * linesR * samplesR; j++)
        {
            bandVector[cont] = imageRecon[j];
            cont++;
            if(imageRecon[j] < min) min = imageRecon[j];
            if(imageRecon[j] > max) max = imageRecon[j];
        }

        QRgb value;
        int color;

        QImage imageRGB(samplesR,linesR,QImage::Format_RGB888);

        for(int i=0; i<samplesR*linesR; i++)
        {
            color = ((bandVector[i]-min)*255)/(max-min);
            value = qRgb(color,color,color);
            imageRGB.setPixel(i%samplesR,i/samplesR,value);
        }

        QGraphicsScene* sceneRecon = new QGraphicsScene();
        sceneRecon->addPixmap(QPixmap::fromImage(imageRGB));
        ui->graphicsViewRecon->setScene(sceneRecon);
        sceneRecon->setSceneRect(0, 0, imageRGB.width(), imageRGB.height());
        delete bandVector;
    }
}



void HyperMix2::paintAbundance(int numAbun)
{
    if(numAbun<bandsA)
    {
        int j, cont = 0;
        double min, max;
        double* bandVector = (double*)malloc(linesA*samplesA*sizeof(double));

        while(ui->comboBoxAbun->count() != 0)
            ui->comboBoxAbun->removeItem(0);

        for(j=0;j<bandsA;j++) //Each Abundance
            ui->comboBoxAbun->addItem(tr("Abundance %1").arg(j));

        //Extract the band that we wat to display
        //Find maximun and minimun value of the band
        min = max = abundanceResult[0];
        for(j = numAbun * linesA * samplesA; j < (numAbun + 1) * linesA * samplesA; j++)
        {
            bandVector[cont] = abundanceResult[j];
            cont++;
            if(abundanceResult[j] < min) min = abundanceResult[j];
            if(abundanceResult[j] > max) max = abundanceResult[j];
        }


        QRgb value;
        int color;

        if(imageAbundance != NULL) free(imageAbundance);
        imageAbundance = new QImage(samplesA,linesA,QImage::Format_RGB888);

        for(int i=0; i<samplesA*linesA; i++)
        {
            color = ((bandVector[i]-min)*255)/(max-min);
            value = qRgb(color,color,color);
            imageAbundance->setPixel(i%samplesA,i/samplesA,value);
        }


        sceneAbundance->addPixmap(QPixmap::fromImage(*imageAbundance));
        sceneAbundance->setWidth(samplesA);
        sceneAbundance->setHeigt(linesA);

        sceneAbundance->setSceneRect(0, 0, imageAbundance->width(), imageAbundance->height());
        ui->comboBoxAbun->setCurrentIndex(numAbun);

        delete bandVector;
    }
}

void HyperMix2::clearImageResultCanvas()
{
    if(ui->graphicsViewImages->scene() != NULL)
    {
        ui->graphicsViewImages->scene()->clear();
        ui->graphicsViewImages->update();
    }
    if(ui->zPlot->graphCount() > 0)
    {
        ui->zPlot->graph(0)->clearData();
        ui->zPlot->removeGraph(0);
        ui->zPlot->replot();
    }
    if(sceneImage != NULL)
    {
        sceneImage->setWidth(0);
        sceneImage->setHeight(0);
    }

}

void HyperMix2::clearAbundanceResultCanvas()
{
    if(ui->graphicsViewAbundances->scene() != NULL)
    {
        ui->graphicsViewAbundances->scene()->clear();
        ui->graphicsViewAbundances->update();
    }
}

void HyperMix2::clearEndmemberResultCanvas()
{
    ui->endmemberPlot->clearGraphs();
    ui->endmemberPlot->replot();
}

void HyperMix2::clearImageRMSEResultCanvas()
{
    if(ui->graphicsViewRecon->scene() != NULL)
    {
        ui->graphicsViewRecon->scene()->clear();
        ui->graphicsViewRecon->update();
    }
    if(ui->graphicsViewRMSE->scene() != NULL)
    {
        ui->graphicsViewRMSE->scene()->clear();
        ui->graphicsViewRMSE->update();
    }
}


void HyperMix2::clearSAD()
{
    if(ui->comboBoxRefMatching->currentIndex() == 0)
    {
        ui->resultsTableRef->clearContents();
        ui->resultsTableRef->setRowCount(0);
        ui->resultsTableRef->setColumnCount(0);

        ui->plotRef->clearGraphs();
        ui->plotRef->replot();

        rowRef = false;

    }
    if(ui->comboBoxSigMatching->currentIndex() == 0)
    {
        ui->resultsTableSig->clearContents();
        ui->resultsTableSig->setRowCount(0);
        ui->resultsTableSig->setColumnCount(0);

        ui->plotSig->clearGraphs();
        ui->plotSig->replot();

        rowSig = false;
    }
    ui->matchingButton->setEnabled(false);
}

void HyperMix2::paintEnd(int numEnd)
{

    if(numEnd == 0) paintEndmembers();
    else paintEndmembers(numEnd-1);

}

void HyperMix2::paintEndmembers()
{
    if(wavelengthE != NULL)
    {
        ui->endmemberPlot->clearGraphs();

        double *endmember = (double*)malloc(bandsE*sizeof(double));

        double min, max;

        int i, j, inc = 0;

        while(ui->comboBoxEnd->count() != 0)
            ui->comboBoxEnd->removeItem(0);

        ui->comboBoxEnd->addItem(tr("All"));

        QVector<double> wavelengthAxis(bandsE);
        QVector<double> reflectanceAxis(bandsE);

        max = min = endmemberResult[0];

        for(i=0; i<linesE; i++)
            for(j=0; j<bandsE; j++)
            {
                if(endmemberResult[j*linesE+i] > max) max = endmemberResult[j*linesE+i];
                if(endmemberResult[j*linesE+i] < min) min = endmemberResult[j*linesE+i];
            }

        for(i=0; i<linesE; i++)
        {
            ui->comboBoxEnd->addItem(tr("Endmember %1").arg(i));

            for(j=0; j<bandsE; j++)
            {
                endmember[j] = endmemberResult[j*linesE+i];
                wavelengthAxis[j] = wavelengthE[j];
            }
            for(j=0; j<bandsE; j++) reflectanceAxis[j] = (endmember[j]-min)/(max-min);

            ui->endmemberPlot->addGraph();
            ui->endmemberPlot->graph(i)->setData(wavelengthAxis, reflectanceAxis);

            inc = i%15;
            switch(inc)
            {
                case 0: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::black))); break;
                case 1: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::red))); break;
                case 2: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::darkRed))); break;
                case 3: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::green))); break;
                case 4: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::darkGreen))); break;
                case 5: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::blue))); break;
                case 6: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::darkBlue))); break;
                case 7: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::cyan))); break;
                case 8: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::darkCyan))); break;
                case 9: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::magenta))); break;
                case 10: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::darkMagenta))); break;
                case 11: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::darkYellow))); break;
                case 12: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::gray))); break;
                case 13: ui->endmemberPlot->graph(i)->setPen(QColor(Qt::GlobalColor(Qt::darkGray))); break;
            }
        }
        ui->endmemberPlot->xAxis->setRange(wavelengthAxis[0], wavelengthAxis[bandsE-1]);
        ui->endmemberPlot->yAxis->setRange(0,1);
        ui->comboBoxEnd->setCurrentIndex(0);
        ui->endmemberPlot->replot();

        free(endmember);
    }
}

void HyperMix2::paintEndmembers(int numEnd)
{
    if(wavelengthE != NULL)
    {
        ui->endmemberPlot->clearGraphs();

        double *endmember = (double*)malloc(bandsE*sizeof(double));

        double min, max;

        int i, j, inc = 0;

        while(ui->comboBoxEnd->count() != 0)
            ui->comboBoxEnd->removeItem(0);

        ui->comboBoxEnd->addItem(tr("All"));

        QVector<double> wavelengthAxis(bandsE);
        QVector<double> reflectanceAxis(bandsE);

        max = min = endmemberResult[0];
        for(i=0; i<linesE; i++)
            for(j=0; j<bandsE; j++)
            {
                if(endmemberResult[j*linesE+i] > max) max = endmemberResult[j*linesE+i];
                if(endmemberResult[j*linesE+i] < min) min = endmemberResult[j*linesE+i];
            }

        for(i=0;i<linesE;i++) //Each endmember
        {
            ui->comboBoxEnd->addItem(tr("Endmember %1").arg(i));

            if(i == numEnd)
            {
                for(j=0; j<bandsE; j++)
                {
                    endmember[j] = endmemberResult[j*linesE+i];
                    wavelengthAxis[j] = wavelengthE[j];
                }
                for(j=0; j<bandsE; j++) reflectanceAxis[j] = (endmember[j]-min)/(max-min);

                ui->endmemberPlot->addGraph();
                ui->endmemberPlot->graph(0)->setData(wavelengthAxis, reflectanceAxis);

                inc = i%15;
                switch(inc)
                {
                    case 0: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::black))); break;
                    case 1: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::red))); break;
                    case 2: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::darkRed))); break;
                    case 3: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::green))); break;
                    case 4: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::darkGreen))); break;
                    case 5: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::blue))); break;
                    case 6: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::darkBlue))); break;
                    case 7: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::cyan))); break;
                    case 8: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::darkCyan))); break;
                    case 9: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::magenta))); break;
                    case 10: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::darkMagenta))); break;
                    case 11: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::darkYellow))); break;
                    case 12: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::gray))); break;
                    case 13: ui->endmemberPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::darkGray))); break;
                }
            }
        }
        ui->endmemberPlot->xAxis->setRange(wavelengthAxis[0], wavelengthAxis[bandsE-1]);
        ui->endmemberPlot->yAxis->setRange(0,1);
        ui->comboBoxEnd->setCurrentIndex(numEnd+1);
        ui->endmemberPlot->replot();

        free(endmember);
    }
}

void HyperMix2::paintZPoint(QPointF pos)
{
    if(ui->zPlot->graphCount() == 0) ui->zPlot->addGraph();
    ui->zPlot->graph(0)->clearData();

    QVector<double> wavelengthAxis(bandsI);
    QVector<double> reflectanceAxis(bandsI);

    double min, max;
    min = max = imageResult[0];

    for(int i=0; i<bandsI; i++)
    {
        if(imageResult[(int)((floor(pos.y())*samplesI+floor(pos.x()))+ i*linesI*samplesI)] < min) min = imageResult[(int)((floor(pos.y())*samplesI+floor(pos.x()))+ i*linesI*samplesI)];
        if(imageResult[(int)((floor(pos.y())*samplesI+floor(pos.x()))+ i*linesI*samplesI)] > max) max = imageResult[(int)((floor(pos.y())*samplesI+floor(pos.x()))+ i*linesI*samplesI)];

    }

    for(int i=0; i<bandsI; i++)
    {
        reflectanceAxis[i] = (imageResult[(int)((floor(pos.y())*samplesI+floor(pos.x()))+ i*linesI*samplesI)] - min) / (max -min);
        wavelengthAxis[i] = wavelengthI[i];

    }

    ui->zPlot->graph(0)->setData(wavelengthAxis, reflectanceAxis);
    ui->zPlot->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::red)));
    ui->zPlot->xAxis->setRange(wavelengthAxis[0], wavelengthAxis[bandsI-1]);
    ui->zPlot->yAxis->setRange(0,1);
    ui->zPlot->replot();
}

void HyperMix2::getPercent(QPointF pos)
{
    ui->lineEditPercentAbun->clear();
    int currentMap = ui->comboBoxAbun->currentIndex();
    int percent = (int)(abundanceResult[(int)((floor(pos.y())*samplesA+floor(pos.x()))+ currentMap*linesA*samplesA)]*100);
    ui->lineEditPercentAbun->setText(tr("%1 %").arg(percent));

}

void HyperMix2::getPercentRMSE(QPointF pos)
{
    ui->lineEditPercentError->clear();
    double percent = rmse[(int)(floor(pos.y()) * samplesR + (int)floor(pos.x()))];
    double max, min, value;
    max = min = imageRecon[((int)(floor(pos.y()) * samplesR + (int)floor(pos.x())))];
    for(int i=0; i<bandsR; i++)
    {
        value = imageRecon[i*linesR+samplesR +((int)(floor(pos.y()) * samplesR + (int)floor(pos.x())))];
        if(value > max) max = value;
        if(value < min) min = value;
    }
    ui->lineEditPercentError->setText(tr("%1").arg(percent*(max-min)));
    ui->lineEditNRMSE->setText(tr("%1").arg(percent));

}

void HyperMix2::loadEndmemberFile()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Endmembers File", ui->workspaceLineEdit->text(), tr("BSQ, BIP, BIL (*)"));

    if(filename != NULL)
    {
        if(!endmemberSet.contains(filename))
        {
            endmemberSet.insert(filename);
            ui->comboBoxEndmembers->addItem(filename);
            ui->comboBoxEndmembers->setCurrentIndex(ui->comboBoxEndRMSE->count()-1);
            ui->comboBoxEndRMSE->addItem(filename);
            ui->comboBoxEndRMSE->setCurrentIndex(ui->comboBoxEndRMSE->count()-1);
            ui->comboBoxRefMatching->addItem(filename);
            ui->comboBoxRefMatching->setCurrentIndex(ui->comboBoxRefMatching->count()-1);
            ui->comboBoxSigMatching->addItem(filename);
            ui->comboBoxSigMatching->setCurrentIndex(ui->comboBoxSigMatching->count()-1);

            ui->stackedWidget->setCurrentIndex(1);
            checkRMSEValues();
        }
    }
}

void HyperMix2::loadAbundanceFile()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Abundance File", ui->workspaceLineEdit->text(), tr("BSQ, BIP, BIL (*)"));

    if(filename != NULL)
    {
        if(!abundanceSet.contains(filename))
        {
            abundanceSet.insert(filename);
            ui->comboBoxAbundances->addItem(filename);
            ui->comboBoxAbundances->setCurrentIndex(ui->comboBoxAbunRMSE->count()-1);
            ui->comboBoxAbunRMSE->addItem(filename);
            ui->comboBoxAbunRMSE->setCurrentIndex(ui->comboBoxAbunRMSE->count()-1);

            ui->stackedWidget->setCurrentIndex(1);
            checkRMSEValues();
        }
    }
}

void HyperMix2::loadEndmemberFileRMSE()
{
    loadEndmemberFile();
}

void HyperMix2::loadAbundanceFileRMSE()
{
    loadAbundanceFile();

}

void HyperMix2::loadImageFileRMSE()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Image File", ui->workspaceLineEdit->text(), tr("BSQ, BIP, BIL (*)"));

    if(filename != NULL)
    {
        if(!rmseImageSet.contains(filename))
        {
            rmseImageSet.insert(filename);
            ui->comboBoxImageRMSE->addItem(filename);
            ui->comboBoxImageComparedRMSE->addItem(filename);
            ui->comboBoxImageRMSE->setCurrentIndex(ui->comboBoxImageRMSE->count()-1);
        }
    }
}

void HyperMix2::loadImageComFileRMSE()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Image File", ui->workspaceLineEdit->text(), tr("BSQ, BIP, BIL (*)"));

    if(filename != NULL)
    {
        if(!rmseImageSet.contains(filename))
        {
            rmseImageSet.insert(filename);
            ui->comboBoxImageRMSE->addItem(filename);
            ui->comboBoxImageComparedRMSE->addItem(filename);
            ui->comboBoxImageComparedRMSE->setCurrentIndex(ui->comboBoxImageComparedRMSE->count()-1);
        }
    }
}

void HyperMix2::loadImage()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Image File", ui->workspaceLineEdit->text(), tr("BSQ, BIP, BIL (*)"));

    if(filename != NULL)
    {
        if(!imageSet.contains(filename))
        {
            imageSet.insert(filename);
            ui->comboBoxImages->addItem(filename);
            ui->comboBoxImages->setCurrentIndex(ui->comboBoxImages->count()-1);
        }
    }
}


void HyperMix2::loadReferences()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Reference Signatures", ui->workspaceLineEdit->text(), tr("BSQ, BIP, BIL (*)"));

    if(filename != NULL)
    {
        if(!endmemberSet.contains(filename))
        {
            endmemberSet.insert(filename);
            ui->comboBoxEndmembers->addItem(filename);
            ui->comboBoxEndmembers->setCurrentIndex(ui->comboBoxEndRMSE->count()-1);
            ui->comboBoxEndRMSE->addItem(filename);
            ui->comboBoxEndRMSE->setCurrentIndex(ui->comboBoxEndRMSE->count()-1);
            ui->comboBoxRefMatching->addItem(filename);
            ui->comboBoxRefMatching->setCurrentIndex(ui->comboBoxRefMatching->count()-1);
            ui->comboBoxSigMatching->addItem(filename);
            ui->comboBoxSigMatching->setCurrentIndex(ui->comboBoxSigMatching->count()-1);
        }
    }
}

void HyperMix2::loadSignatures()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Signatures File", ui->workspaceLineEdit->text(), tr("BSQ, BIP, BIL (*)"));

    if(filename != NULL)
    {
        if(!endmemberSet.contains(filename))
        {
            endmemberSet.insert(filename);
            ui->comboBoxEndmembers->addItem(filename);
            ui->comboBoxEndmembers->setCurrentIndex(ui->comboBoxEndRMSE->count()-1);
            ui->comboBoxEndRMSE->addItem(filename);
            ui->comboBoxEndRMSE->setCurrentIndex(ui->comboBoxEndRMSE->count()-1);
            ui->comboBoxRefMatching->addItem(filename);
            ui->comboBoxRefMatching->setCurrentIndex(ui->comboBoxRefMatching->count()-1);
            ui->comboBoxSigMatching->addItem(filename);
            ui->comboBoxSigMatching->setCurrentIndex(ui->comboBoxSigMatching->count()-1);
        }
    }
}



void HyperMix2::checkRMSEValues()
{
    if(ui->tabWidget_RMSE->currentIndex() == 0)
    {
        if(ui->comboBoxEndRMSE->currentIndex() != 0 && ui->comboBoxAbunRMSE->currentIndex() != 0)
        {
            if(linesE == bandsA) // Both files have same dimension
            {
                if(ui->comboBoxImageRMSE->currentIndex() != 0) ui->RMSEButton->setEnabled(true);
            }
            else
                ui->RMSEButton->setEnabled(false);
        }
        else
        {
            ui->lineEditPercentError->clear();
            ui->RMSEButton->setEnabled(false);
        }
    }
    else
    {
        if(ui->comboBoxImageComparedRMSE ->currentIndex() != 0)
        {
            if(bandsR == bandsRO)
            {
                if(ui->comboBoxImageRMSE->currentIndex() != 0) ui->RMSEButton->setEnabled(true);
            }
            else
                ui->RMSEButton->setEnabled(false);
        }
        else
        {
            ui->lineEditPercentError->clear();
            ui->RMSEButton->setEnabled(false);
        }
    }
}


void HyperMix2::RMSE()
{
    ui->progressBar->setVisible(true);
    ui->progressBar->setValue(0);
    int linesSamples, i, j;

    if(ui->tabWidget_RMSE->currentIndex() == 0)
    {
        if(imageRecon != NULL) free(imageRecon);
        imageRecon = (double*)malloc(linesA*samplesA*bandsE*sizeof(double));
        linesR = linesA;
        samplesR = samplesA;
        bandsR = bandsE;

        linesSamples = linesA*samplesA;

        dgemm_('N', 'N', linesSamples, bandsE, bandsA, 1.0, abundanceResult, linesSamples, endmemberResult, bandsA, 0.0, imageRecon, linesSamples);
    }

    ui->progressBar->setValue(15);
    paintRecon(0);
    ui->progressBar->setValue(30);
    ui->horizontalSliderBandsRecon->setEnabled(true);

    if(linesR != linesRO || samplesR != samplesRO)
         ui->lineEditPercentError->setText(tr("Files don't match"));
    else
    {
        ui->progressBar->setValue(40);
        double difference, acumulate;
        rmse = (double*)malloc(linesR*samplesR*sizeof(double));
        for(i=0; i<linesR*samplesR; i++)
        {
            for(j=0; j<bandsR; j++)
            {
                difference = imageReconOri[j*linesRO*samplesRO + i] - imageRecon[j*linesR*samplesRO + i];
                difference = difference*difference;
                acumulate = acumulate + difference;
            }
            rmse[i] = sqrt(acumulate)/bandsR;
            acumulate = 0;
        }
        ui->progressBar->setValue(60);
        double min, max, mean;
        acumulate = 0;
        //Find maximun and minimun value of the band
        min = rmse[0];
        max = rmse[0];
        for(j = 0; j < linesR * samplesR; j++)
        {
            if(rmse[j] < min) min = rmse[j];
            if(rmse[j] > max) max = rmse[j];

            acumulate = acumulate + rmse[j];
        }
        mean = acumulate/(linesR*samplesR);

        ui->progressBar->setValue(80);
        ui->lineEditMeanError->setText(QString::number(mean));

        ui->lineEditMaxError->setText(QString::number(max));
        ui->lineEditMinError->setText(QString::number(min));

        QRgb value;
        int color, red, green, blue;

        if(imageRMSE != NULL) free(imageRMSE);

        imageRMSE = new QImage(samplesR,linesR,QImage::Format_RGB888);

        acumulate = 0;
        for(i=0; i<samplesR*linesR; i++)
        {
            min = max = 0;
            for(j=0; j<bandsR; j++)
            {
                if(imageReconOri[j*linesR*samplesRO + i] > max) max = imageReconOri[j*linesR*samplesRO + i];
                if(imageReconOri[j*linesR*samplesRO + i] < min) min = imageReconOri[j*linesR*samplesRO + i];
            }
            rmse[i] /= (max-min);
            acumulate = acumulate + rmse[i];

            color = rmse[i] * 255;
            getColor(color, red, green, blue);
            value = qRgb(red,green,blue);
            imageRMSE->setPixel(i%samplesR,i/samplesR,value);
        }
        ui->lineEditNRMSEMean->setText(QString::number(acumulate/(linesR*samplesR)));

        ui->progressBar->setValue(90);
        sceneRMSE->setWidth(samplesR);
        sceneRMSE->setHeigt(linesR);
        sceneRMSE->addPixmap(QPixmap::fromImage(*imageRMSE));
        sceneRMSE->setSceneRect(0,0,imageRMSE->width(), imageRMSE->height());
        ui->graphicsViewRMSE->setScene(sceneRMSE);



    }
    ui->progressBar->setValue(100);
    ui->progressBar->setVisible(false);

}

void HyperMix2::runSAD()
{
    int i,j , R, G, B;
    QTableWidgetItem *item;
    double *ref, *sig, mean = 0;
    char cad[MAXCAD]="";
    QMessageBox msgBox;

    if(bandsRef == bandsSig)
    {
        if(ui->comboBoxAuto->currentText().compare("Manual") == 0)
            ui->lineEditSAD->setText(QString::number(angle(endmemberRef, endmemberSig, bandsRef)));
        else
        {
            for(j=0; j<bandsRef; j++)
            {
                for(i=0; i<linesRef; i++)
                    ui->resultsTableRef->item(i,j)->setBackgroundColor(QColor(Qt::white));
                for(i=0; i<linesSig; i++)
                    ui->resultsTableSig->item(i,j)->setBackgroundColor(QColor(Qt::white));
            }

            table = (int*)calloc(linesRef,sizeof(int));
            order = (int*)calloc(linesRef,sizeof(int));
            for(i=0; i<linesRef; i++) order[i] = i;
            createAnglesMatrix();
            if(ui->comboBoxMatching->currentText().compare("Minimun") == 0) minimunSAD();
            if(ui->comboBoxMatching->currentText().compare("Average") == 0) averageSAD();
            if(ui->comboBoxMatching->currentText().compare("Optimus") == 0) optimusSAD();

            ref = (double*)calloc(bandsRef,sizeof(double));
            sig = (double*)calloc(bandsRef,sizeof(double));

            if(linesRef <=linesSig)
            {
                for(i=0; i<linesRef; i++)
                {
                    randColor(&R,&G,&B);
                    for(j=0; j<bandsRef; j++)
                    {
                        item = ui->resultsTableRef->item(order[i],j);
                        ref[j] = item->text().toDouble();
                        item->setBackgroundColor(QColor(R,G,B));
                        item = ui->resultsTableSig->item(table[order[i]],j);
                        sig[j] = item->text().toDouble();
                        item->setBackgroundColor(QColor(R,G,B));

                        if(j==bandsRef-1) mean += angle(sig,ref,bandsRef);
                    }
                }
            }
            else
            {
                msgBox.setInformativeText("The number of references must be less or equal to the signatures.");
                msgBox.setStandardButtons(QMessageBox::Ok);
                msgBox.exec();
            }


            mean /= linesRef;
            ui->lineEditSAD->setText(QString::number(mean));
            free(table);
        }
    }
    else
    {

        msgBox.setInformativeText("The number of bands must be the same between references and signatures.");
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.exec();
    }

}

double HyperMix2::angle(double *a, double *b, int size)
{

    double tita= 0;
    double cos= 0;
    double abstita= 0;

    //sqrt of sum of squares of values of elements
    int i;
    float acumulateA = 0, acumulateB = 0, dotProduct = 0, modA, modB;
    for(i = 0; i< size; i++)
    {
        acumulateA = acumulateA + (a[i]*a[i]);
        acumulateB = acumulateB + (b[i]*b[i]);
        dotProduct = dotProduct + (a[i]*b[i]);
    }
    modA = sqrt(acumulateA);
    modB = sqrt(acumulateB);

    cos = dotProduct / (modA*modB);  // calculamos el coseno del angulo entre ambos
    if (cos >= 1) return 0;
    tita = ::acos( cos );   // calculamos el angulo como el arcoseno
    if (tita < 0 ) abstita = tita * -1;  // lo devolvemos en valor absoluto (positivo)
    else abstita = tita;

    return abstita;

}

void HyperMix2::createAnglesMatrix()
{

    double* vecA = (double*)calloc(bandsRef,sizeof(double));
    double* vecB = (double*)calloc(bandsSig,sizeof(double));
    QTableWidgetItem *item;
    int i,j,k,l;

    angles = (double*)calloc(bandsRef*bandsSig,sizeof(double)); //angles between endmembers and real signatures
    //Build the angles matrix
    if(linesRef <= linesSig)
    {
        for(i=0; i<linesRef; i++)
        {
            for(j=0; j<bandsRef; j++)
            {
                item = ui->resultsTableRef->item(i,j);
                vecB[j] = item->text().toDouble();
            }

            for(k=0; k<linesSig;k++)
            {
                for(l=0; l<bandsSig; l++)
                {
                    item = ui->resultsTableSig->item(k,l);
                    vecA[l] = item->text().toDouble();
                }

                angles[i*linesSig +k] = angle(vecA,vecB,bandsRef);
            }
        }
    }
    free(vecA);
    free(vecB);

}

#define MAXRAD 6.28318

void HyperMix2::minimunSAD()
{
    int i, j,k;

    double min = MAXRAD;
    bool enc;

    if(linesRef <=linesSig)
    {
        for(i=0; i<linesRef; i++)
        {
            table[order[i]] = -1;
            min = MAXRAD;
            for(j=0; j<linesSig; j++)
            {
                if(angles[order[i]*linesSig+j] < min)
                {
                    enc = false;
                    for(k=0; k<i; k++)
                        if(table[order[k]] == j) enc = true;
                    if(!enc)
                    {
                        min = angles[order[i]*linesSig+j];
                        table[order[i]] = j;
                    }
                }
            }
        }

    }
}

#define MAXITERATIONS 500

void HyperMix2::averageSAD()
{
    int i,j,k, aux;
    double min = MAXRAD;
    QTableWidgetItem *item;
    double *ref, *sig, mean = 0;
    int *copytable = (int*)calloc(linesRef,sizeof(int));

    ref = (double*)calloc(bandsRef,sizeof(double));
    sig = (double*)calloc(bandsRef,sizeof(double));

    for(k=0; k<MAXITERATIONS; k++)
    {

        for(i=0; i<MAXITERATIONS; i++)
        {
            j = order[qrand()%linesRef];
            if(j != i%linesRef)
            {
                aux = order[j];
                order[j] = order[i%linesRef];
                order[i%linesRef] = aux;
            }
        }

        minimunSAD();

        for(i=0; i<linesRef; i++)
        {
            for(j=0; j<bandsRef; j++)
            {
                item = ui->resultsTableRef->item(i,j);
                ref[j] = item->text().toDouble();
                item = ui->resultsTableSig->item(table[i],j);
                sig[j] = item->text().toDouble();
                if(j==bandsRef-1) mean += angle(sig,ref,bandsRef);
            }
        }
        mean /= linesRef;

        if(mean < min)
        {
            mean = min;
            for(i=0; i<linesRef; i++)
                copytable[i] = table[i];
        }
    }

    for(i=0; i<linesRef; i++)
        table[i] = copytable[i];
}

void HyperMix2::optimusSAD()
{
    int* sol = (int*)malloc(linesSig*sizeof(int));
    int i;
    double min;

    for(i=0; i<linesSig; i++)
    {
        sol[i] =i;
        table[i] = i;
    }

    MatchingOptRec(0,min,sol,table);

    free(sol);

}

void HyperMix2::MatchingOptRec(int k, double &min, int* sol,int* solmin)
{

    int j, ind;
    double oneNorm = 0;
    bool valido;
    min = MAXRAD;

    if (k == linesSig)
    {
        for (j=0; j<linesRef; j++)
            oneNorm = oneNorm + angles[sol[k]*linesRef + j];

        if (oneNorm < min)
        {
            min = oneNorm;
            for(j=0; j<linesSig;j++) solmin[j] = sol[j];
        }
    }
    else
    {
        for (j = 0; j < linesRef;j++)
        {
            valido = true;
            ind =0;		 // mira si j no está ya en la solucion
            while (valido && ind < k)
            {
                if (sol[ind] == j) valido = false;
                ind++;
            }
            if (valido)
            {
                sol[k] = j;
                MatchingOptRec(k+1,min,sol,solmin);
            }
        }
    }
}

void HyperMix2::rowRefSelected(int row, int col)
{
    if(ui->plotRef->graphCount() == 0) ui->plotRef->addGraph();
    ui->plotRef->graph(0)->clearData();

    QVector<double> wavelengthAxis(bandsRef);
    QVector<double> reflectanceAxis(bandsRef);

    endmemberRef = (double*)malloc(bandsRef*sizeof(double));
    double min, max;
    min = max = endmemberResultRef[0];
    for(int i=0; i<bandsRef; i++)
    {
        if(endmemberResultRef[i*linesRef + row] < min) min = endmemberResultRef[i*linesRef + row];
        if(endmemberResultRef[i*linesRef + row] > max) max = endmemberResultRef[i*linesRef + row];
        endmemberRef[i] = endmemberResultRef[i*linesRef + row];
    }

    for(int i=0; i<bandsRef; i++)
    {
        reflectanceAxis[i] = (endmemberResultRef[i*linesRef + row] - min) / (max -min);
        wavelengthAxis[i] = wavelengthRef[i];
    }

    ui->plotRef->graph(0)->setData(wavelengthAxis, reflectanceAxis);
    ui->plotRef->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::blue)));
    ui->plotRef->xAxis->setRange(wavelengthAxis[0], wavelengthAxis[bandsRef-1]);
    ui->plotRef->yAxis->setRange(0,1);
    ui->plotRef->replot();

    rowRef = true;
    ui->matchingButton->setEnabled(rowRef && rowSig);

}

void HyperMix2::rowSigSelected(int row, int col)
{

    if(ui->plotSig->graphCount() == 0) ui->plotSig->addGraph();
    ui->plotSig->graph(0)->clearData();

    QVector<double> wavelengthAxis(bandsSig);
    QVector<double> reflectanceAxis(bandsSig);

    endmemberSig = (double*)malloc(bandsSig*sizeof(double));
    double min, max;
    min = max = endmemberResultSig[0];
    for(int i=0; i<bandsSig; i++)
    {
        if(endmemberResultSig[i*linesSig + row] < min) min = endmemberResultSig[i*linesSig + row];
        if(endmemberResultSig[i*linesSig + row] > max) max = endmemberResultSig[i*linesSig + row];
        endmemberSig[i] = endmemberResultSig[i*linesSig + row];
    }

    for(int i=0; i<bandsSig; i++)
    {
        reflectanceAxis[i] = (endmemberResultSig[i*linesSig + row] - min) / (max -min);
        wavelengthAxis[i] = wavelengthSig[i];
    }

    ui->plotSig->graph(0)->setData(wavelengthAxis, reflectanceAxis);
    ui->plotSig->graph(0)->setPen(QColor(Qt::GlobalColor(Qt::blue)));
    ui->plotSig->xAxis->setRange(wavelengthAxis[0], wavelengthAxis[bandsSig-1]);
    ui->plotSig->yAxis->setRange(0,1);
    ui->plotSig->replot();

    rowSig = true;
    ui->matchingButton->setEnabled(rowRef && rowSig);

}



void HyperMix2::setAutoMatching(QString str)
{

    if(str.compare("Automatic") == 0)
    {
        ui->comboBoxMatching->setEnabled(true);
    }
    else
    {
        ui->comboBoxMatching->setEnabled(false);
    }
}

void HyperMix2::randColor(int *R, int *G, int *B)
{
    *R = qrand() % 256;
    *G = qrand() % 256;
    *B = qrand() % 256;
}


void HyperMix2::savePNG(QPixmap *pixmap)
{

    QString filename = QFileDialog::getSaveFileName(this, "Save as...", ui->workspaceLineEdit->text(), tr("Image Files (*.png)"));

    if(!filename.contains(".png")) filename.append(".png");

    if(filename != NULL)
    {
        QFile file(filename.toStdString().c_str());
        file.open(QIODevice::WriteOnly);
        pixmap->save(&file, "PNG");
    }
}


void HyperMix2::saveRMSE()
{

    if(imageRMSE != NULL)
    {
        QPixmap pixmap = QPixmap::fromImage(*imageRMSE);
        savePNG(&pixmap);
    }

}


void HyperMix2::saveImage()
{
    if(imageRGB != NULL)
    {
        QPixmap pixmap = QPixmap::fromImage(*imageRGB);
        savePNG(&pixmap);
    }
}

void HyperMix2::saveAbundances()
{
    if(imageAbundance != NULL)
    {
        QPixmap pixmap = QPixmap::fromImage(*imageAbundance);
        savePNG(&pixmap);
    }
}

int HyperMix2::dgemm_(char transa, char transb, int m, int n,
    int k, double alpha, double *a, int lda, double *b, int ldb,
    double beta, double *c, int ldc){


    /* System generated locals */
        int a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
        i__3;


    /* Local variables */
        int info;
        bool nota;
        bool notb;
        double temp;
        int i, j, l, ncola;
        int nrowa, nrowb;



#define A(I,J) a[(I)-1 + ((J)-1)* ( lda)]
#define B(I,J) b[(I)-1 + ((J)-1)* ( ldb)]
#define C(I,J) c[(I)-1 + ((J)-1)* ( ldc)]

    if(transa =='N'){
        nota=true;
    }
    else{
        nota=false;
    }
    if(transb =='N'){
        notb=true;
    }
    else{
        notb=false;
    }
    //nota = lsame_(transa, "N");
    //notb = lsame_(transb, "N");
    if (nota) {
    nrowa = m;
    ncola = k;
    } else {
    nrowa = k;
    ncola = m;
    }
    if (notb) {
    nrowb = k;
    } else {
    nrowb = n;
    }


/*     Quick return if possible. */

    if (m == 0 || n == 0 || (alpha == 0 || k == 0) && beta == 1) {
    return 0;
    }

/*     And if  alpha.eq.zero. */

    if (alpha == 0) {
    if (beta == 0) {
        i__1 = n;
        for (j = 1; j <= n; ++j) {
        i__2 = m;
        for (i = 1; i <= m; ++i) {
            C(i,j) = 0;
/* L10: */
        }
/* L20: */
        }
    } else {
        i__1 = n;
        for (j = 1; j <= n; ++j) {
        i__2 = m;
        for (i = 1; i <= m; ++i) {
            C(i,j) = beta * C(i,j);
/* L30: */
        }
/* L40: */
        }
    }
    return 0;
    }

/*     Start the operations. */

    if (notb) {
    if (nota) {

/*           Form  C := alpha*A*B + beta*C. */

        i__1 = n;
        for (j = 1; j <= n; ++j) {
        if (beta == 0) {
            i__2 = m;
            for (i = 1; i <= m; ++i) {
            C(i,j) = 0;
/* L50: */
            }
        } else if (beta != 1) {
            i__2 = m;
            for (i = 1; i <= m; ++i) {
            C(i,j) = beta * C(i,j);
/* L60: */
            }
        }
        i__2 = k;
        for (l = 1; l <= k; ++l) {
            if (B(l,j) != 0) {
            temp = alpha * B(l,j);
            i__3 = m;
            for (i = 1; i <= m; ++i) {
                C(i,j) += temp * A(i,l);
/* L70: */
            }
            }
/* L80: */
        }
/* L90: */
        }
    } else {

/*           Form  C := alpha*A'*B + beta*C */

        i__1 = n;
        for (j = 1; j <= n; ++j) {
        i__2 = m;
        for (i = 1; i <= m; ++i) {
            temp = 0;
            i__3 = k;
            for (l = 1; l <= k; ++l) {
            temp += A(l,i) * B(l,j);
/* L100: */
            }
            if (beta == 0) {
            C(i,j) = alpha * temp;
            } else {
            C(i,j) = alpha * temp + beta * C(i,j);
            }
/* L110: */
        }
/* L120: */
        }
    }
    } else {
    if (nota) {

/*           Form  C := alpha*A*B' + beta*C */

        i__1 = n;
        for (j = 1; j <= n; ++j) {
        if (beta == 0) {
            i__2 = m;
            for (i = 1; i <= m; ++i) {
            C(i,j) = 0;
/* L130: */
            }
        } else if (beta != 1) {
            i__2 = m;
            for (i = 1; i <= m; ++i) {
            C(i,j) = beta * C(i,j);
/* L140: */
            }
        }
        i__2 = k;
        for (l = 1; l <= k; ++l) {
            if (B(j,l) != 0) {
            temp = alpha * B(j,l);
            i__3 = m;
            for (i = 1; i <= m; ++i) {
                C(i,j) += temp * A(i,l);
/* L150: */
            }
            }
/* L160: */
        }
/* L170: */
        }
    } else {

/*           Form  C := alpha*A'*B' + beta*C */

        i__1 = n;
        for (j = 1; j <= n; ++j) {
        i__2 = m;
        for (i = 1; i <= m; ++i) {
            temp = 0;
            i__3 = k;
            for (l = 1; l <= k; ++l) {
            temp += A(l,i) * B(j,l);
/* L180: */
            }
            if (beta == 0) {
            C(i,j) = alpha * temp;
            } else {
            C(i,j) = alpha * temp + beta * C(i,j);
            }
/* L190: */
        }
/* L200: */
        }
    }
    }

    return 0;

/*     End of DGEMM . */

} /* dgemm_ */


void HyperMix2::getColor(int color, int &red,int &green,int &blue)
{
    if(color>=0 && color<32)
    {
        red = 0;
        green = 0;
        blue = 131 + 4*color;
    }
    if(color>=32 && color<=96)
    {
        red = 0;
        if(color<65) green = 4*(color-32);
        else green = 4*(color-32)-1;
        blue = 255;
    }
    if(color>=97 && color<=160)
    {
        if(color<130) red = 4*(color-97);
        else red = 4*(color-97) -1;
        green = 255;
        if(color<129) blue = 4*(64-(color-97))-1;
        else blue = 4*(64-(color-97));
    }
    if(color>=161 && color<=224)
    {
        red = 255;
        if(color <= 193) green = 4*(64 -(color-161))-1;
        else green = 4*(64 -(color-161));
        blue = 0;
    }
    if(color>=225 && color<=255)
    {
        red = 131 + 4*(32 -(color-225)-1);
        green = 0;
        blue = 0;
    }
    if(color > 255)
    {
        red = 255;
        green = 0;
        blue = 0;
    }
}

HyperMix2::~HyperMix2()
{
    delete workspaceD;
    delete helpD;
    delete scene;
    delete ui;
}
