#ifndef HYPERMIX2_H
#define HYPERMIX2_H

#include<fstream>
#include <QMainWindow>
#include <QDir>
#include <QMouseEvent>
#include <QPushButton>
#include <QToolButton>
#include <QTextStream>
#include <QMessageBox>
#include <QFontComboBox>
#include <QComboBox>
#include <QSet>
#include <QStringList>
#include <QWebFrame>

#include "Editor/editorscene.h"
#include "Editor/editorimagescene.h"
#include "Editor/editorabundancescene.h"
#include "Editor/editorrmsescene.h"
#include "Editor/scheme.h"
#include "workspacedialog.h"
#include "helpdialog.h"
#include "qcustomplot.h"
#include "infodialog.h"
#include "consolelogdialog.h"

#define SCALEFAC 1.15
#define DBL_MAX 1E+37

namespace Ui {
class HyperMix2;
}

class HyperMix2 : public QMainWindow
{
    Q_OBJECT
    
public:
    enum{SelectItem, InsertLine, InsertImage, InsertItem};

    explicit HyperMix2(QWidget *parent = 0);
    ~HyperMix2();

private:
    Ui::HyperMix2 *ui;

    //Attributes**************************
    EditorScene *scene;
    EditorImageScene *sceneImage;
    EditorAbundanceScene *sceneAbundance;
    EditorRMSEScene *sceneRMSE;

    QMenu *itemMenu;
    QMenu *arrowMenu;
    bool cudaSupported;
    double zoom;
    QComboBox *numberExecutions;
    int numberExec;
    QComboBox *background;

    QSet<QString> imageSet;
    QSet<QString> endmemberSet;
    QSet<QString> abundanceSet;
    QSet<QString> rmseImageSet
;
    int linesA;
    int samplesA;
    int bandsA;
    int dataTypeA;
    char *interleaveA;
    int byteOrderA;
    double* abundanceResult;

    int linesI;
    int samplesI;
    int bandsI;
    int dataTypeI;
    char *interleaveI;
    int byteOrderI;
    double* imageResult;

    int linesR;
    int samplesR;
    int bandsR;
    int dataTypeR;
    char *interleaveR;
    int byteOrderR;
    double* imageRecon;

    int linesRO;
    int samplesRO;
    int bandsRO;
    int dataTypeRO;
    char *interleaveRO;
    int byteOrderRO;
    double* imageReconOri;

    double* rmse;
    bool allow;

    int linesE;
    int samplesE;
    int bandsE;
    int dataTypeE;
    char *interleaveE;
    int byteOrderE;
    double* endmemberResult;
    bool emptyEnd;

    char *waveUnitI;
    char *waveUnitE;
    double* wavelengthI;
    double* wavelengthE;    


    int linesRef;
    int samplesRef;
    int bandsRef;
    int dataTypeRef;
    char *interleaveRef;
    int byteOrderRef;
    double* endmemberResultRef;


    int linesSig;
    int samplesSig;
    int bandsSig;
    int dataTypeSig;
    char *interleaveSig;
    int byteOrderSig;
    double* endmemberResultSig;

    char *waveUnitRef;
    char *waveUnitSig;
    double* wavelengthRef;
    double* wavelengthSig;

    bool rowRef;
    bool rowSig;
    double* endmemberRef;
    double* endmemberSig;

    double* angles;
    int* table;
    int* order;

    double maxminRMSE;

    QImage* imageRGB;
    QImage* imageRMSE;
    QImage* imageSB;
    QImage* imageAbundance;
    QGraphicsScene *SB;

    //************************************
    //Actions*****************************
    QAction *deleteAction;
    QAction *compileAction;
    QAction *zoomIn;
    QAction *zoomOut;
    QAction *clearOutput;
    //************************************
    //Buttons*****************************
    QList<QPushButton*> buttonListEstimation;
    QList<QPushButton*> buttonListPreprocessing;
    QList<QPushButton*> buttonListExtraction;
    QList<QPushButton*> buttonListAbundance;

    QVector<int> contOperators;

    QButtonGroup *groupButton;
    QButtonGroup *groupOperators;
    QToolButton *pointerButton;
    QToolButton *linePointerButton;
    QToolButton *imageButton;
    //************************************
    //Scheme******************************
    Scheme *scheme;
    //************************************
    //Dilogs******************************
    HelpDialog *helpD;
    WorkspaceDialog *workspaceD;
    ConsoleLogDialog *consoleD;
    //************************************

protected:
    void loadToolBar();
    void loadButtons();
    void handleExecutionsChange();

    void resizeEvent(QResizeEvent* event);

public slots:
    void loadOperators();
    void updateCheckedOperators(int id);
    void updateCheckedButtons(int id);
    void itemInserted(EditorItem *item);
    void itemClicked(EditorItem *item);
    void imageInserted(NewImageDialog *imageD, EditorImageItem *image);
    void deleteItem();
    void compileScheme();
    void showHelp();
    void changeWorkspace();
    void openWorkspaceD();
    void zoomInInc();
    void zoomOutInc();
    void clear();
    void openParamDialog();
    void numberExecutionsChanged(QString number);
    void diagramPage();
    void resultsPage();
    void blogPage();
    //Results
    void paintEnd(int numEnd);
    void paintEndmembers();
    void paintEndmembers(int numEnd);
    void newImageResult(QString image);
    void newEndmemberResult(QString endmembers);
    void newAbundanceResult(QString abundance);
    void currentImageResultChange(QString file);
    void currentEndmemberResultChange(QString file);
    void currentAbundanceResultChange(QString file);
    void currentEndmemberRMSEChange(QString file);
    void currentAbundanceRMSEChange(QString file);
    void currentImageRMSEChange(QString file);
    void currentImageComRMSEChange(QString file);
    void currentReferencesChange(QString file);
    void currentSignaturesChange(QString file);
    void tabRMSEChanged(int tab);
    void paintBand(int numBand);
    void paintRecon(int numBand);
    void paintAbundance(int numAbun);
    void loadEndmemberFile();
    void loadAbundanceFile();
    void loadEndmemberFileRMSE();
    void loadAbundanceFileRMSE();
    void loadImageFileRMSE();
    void loadImageComFileRMSE();
    void loadImage();
    void loadReferences();
    void loadSignatures();
    void paintZPoint(QPointF pos);
    void getPercent(QPointF pos);
    void getPercentRMSE(QPointF pos);
    void runSAD();
    void rowRefSelected(int row, int col);
    void rowSigSelected(int row, int col);
    void RMSE();
    void setAutoMatching(QString str);
    //Palettes
    void currentBackgroundChanged(QString backg);
    //Console
    void openConsole();
    void saveRMSE();
    void saveImage();
    void saveAbundances();


private:
    void cleanString(char *cadena, char *out);
    int readHeader1(const char* filename, int *lines, int *samples, int *bands, int *dataType,
            char* interleave, int *byteOrder, char* waveUnit);
    int readHeader2(const char* filename, double* wavelength);
    int loadImage(const char* filename, double* image, int lines, int samples, int bands, int dataType, char *interleave);
    void clearImageResultCanvas();
    void clearEndmemberResultCanvas();
    void clearAbundanceResultCanvas();
    void clearImageRMSEResultCanvas();
    void clearSAD();
    void checkRMSEValues();
    void wheelEvent(QWheelEvent* event);
    void createAnglesMatrix();
    void minimunSAD();
    void averageSAD();
    void optimusSAD();
    void MatchingOptRec(int k, double &min, int* sol, int* solmin);
    void randColor(int *R, int *G, int *B);

    void savePNG(QPixmap* pixmap);

    double angle(double *a, double *b, int size);
    int dgemm_(char transa, char transb, int m, int n,
        int k, double alpha, double *a, int lda, double *b, int ldb,
        double beta, double *c, int ldc);
    void getColor(int color, int &red,int &green,int &blue);

};

#endif // HYPERMIX2_H
