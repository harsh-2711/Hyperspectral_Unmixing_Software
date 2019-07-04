#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define MAXLINE 200
#define MAXCAD 200
#define FPS 5

extern "C" int dgemm_(char *transa, char *transb, int *m, int *
		n, int *k, double *alpha, double *a, int *lda,
		double *b, int *ldb, double *beta, double *c, int
		*ldc);


extern "C" int dgesvd_(char *jobu, char *jobvt, int *m, int *n,
	double *a, int *lda, double *s, double *u, int *
	ldu, double *vt, int *ldvt, double *work, int *lwork,
	int *info);

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
void cleanString(char *cadena, char *out)
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
int readHeader1(char* filename, int *lines, int *samples, int *bands, int *dataType,
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
int readHeader2(char* filename, double* wavelength)
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
                    wavelength[cont]= atof(pch);
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
int loadImage(char* filename, double* image, int lines, int samples, int bands, int dataType, char* interleave)
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

/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
int writeValueResults(char* filename, int result)
{
    FILE *fp;
    int out = result;
    if ((fp=fopen(filename,"wb"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
        fwrite(&out,1,sizeof(int),fp);
        fclose(fp);
    }
    else return -1;

    return 0;
}

/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char* argv[])
{

	/*
	 * PARAMETERS:
	 *
	 * argv[1]: Image filename
	 * argv[2]: Approximation value
	 * argv[3]: Output Result File
	 *
	 * */

	if(argc != 4)
	{
		printf("EXECUTION ERROR VD Iterative: Parameters are not correct.");
		printf("./VD [Image Filename] [Approximation] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}


	int i, j, N;
    float mean;

    double sigmaSquareTest;
    double sigmaTest;
    double TaoTest;

    double *meanSpect;
    double *Cov;
    double *Corr;
    double *CovEigVal;
    double *CorrEigVal;
    double *U;
    double *VT;

	//READ IMAGE
	char cad[MAXCAD];
	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)malloc(MAXCAD*sizeof(char));
	waveUnit = (char*)malloc(MAXCAD*sizeof(char));

	strcpy(cad,argv[1]); // Second parameter: Header file:
	strcat(cad,".hdr");
	int error = readHeader1(cad, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("EXECUTION ERROR VD Iterative: Error 1 reading header file: %s.", cad);
		fflush(stdout);
		exit(-1);
	}

	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(cad,argv[1]);
	strcat(cad,".hdr");
	error = readHeader2(cad, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR VD Iterative: Error 2 reading header file: %s.", cad);
		fflush(stdout);
		exit(-1);
	}
	double *image = (double*)malloc(lines*samples*bands*sizeof(double));
	error = loadImage(argv[1], image, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR: Error reading image file: %s.", argv[1]);
		fflush(stdout);
		exit(-1);
	}


    N=lines*samples;
	//COVARIANCE
    meanSpect		= (double*) malloc(bands * sizeof(double));
    Cov			= (double*) malloc(bands * bands * sizeof(double));
    Corr			= (double*) malloc(bands * bands * sizeof(double));
    CovEigVal		= (double*) malloc(bands * sizeof(double));
    CorrEigVal	= (double*) malloc(bands * sizeof(double));
    U		= (double*) malloc(bands * bands * sizeof(double));
    VT	= (double*) malloc(bands * bands * sizeof(double));

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************

    for(i=0; i<bands; i++)
    {
		mean=0;
        for(j=0; j<N; j++)
			mean+=(image[(i*N)+j]);

		mean/=N;
        meanSpect[i]=mean;

        for(j=0; j<N; j++)
			image[(i*N)+j]=image[(i*N)+j]-mean;

	}

    double alpha = (double)1/N, beta = 0;
    dgemm_("T", "N", &bands, &bands, &N, &alpha, image, &N, image, &N, &beta, Cov, &bands);

	//CORRELATION
    for(j=0; j<bands; j++)
        for(i=0; i<bands; i++)
        	Corr[(i*bands)+j] = Cov[(i*bands)+j]+(meanSpect[i] * meanSpect[j]);

	//SVD
    int lwork = MAX(1,MAX(3*MIN(bands, bands)+MAX(bands,bands),5*MIN(bands,bands)));
    int info;
    double *work = (double*)malloc(lwork*sizeof(double));
    dgesvd_("N", "N", &bands, &bands, Cov, &bands, CovEigVal, U, &bands, VT, &bands, work, &lwork, &info);
    dgesvd_("N", "N", &bands, &bands, Corr, &bands, CorrEigVal, U, &bands, VT, &bands, work, &lwork, &info);

    //ESTIMATION
    int* count = (int*)malloc(FPS * sizeof(int));
    double e;
    for(i=0; i<FPS; i++) count[i] = 0;

    for(i=0; i<bands; i++)
    {
    	sigmaSquareTest = (CovEigVal[i]*CovEigVal[i]+CorrEigVal[i]*CorrEigVal[i])*2/samples/lines;
    	sigmaTest = sqrt(sigmaSquareTest);

    	for(j=1;j<=FPS;j++)
        {
        	switch(j)
        	{
				case 1: e = 0.906193802436823;
				break;
				case 2: e = 1.644976357133188;
				break;
				case 3: e = 2.185124219133003;
				break;
				case 4: e = 2.629741776210312;
				break;
				case 5: e = 3.015733201402701;
				break;
        	}
            TaoTest = sqrt(2) * sigmaTest * e;

            if((CorrEigVal[i]-CovEigVal[i]) > TaoTest)
                count[j-1]++;
        }
    }
    int res = count[atoi(argv[2])-1];

    //END CLOCK*****************************************
	end = clock();
	printf("Iterative VD: %f segundos, Result : %d", (double)(end - start) / CLOCKS_PER_SEC, res);
	fflush(stdout);
	//**************************************************

	error = writeValueResults(argv[3], res);
	if(error != 0)
	{
		printf("EXECUTION ERROR VD Iterative: Error writing results file: %s.", argv[3]);
		fflush(stdout);
		exit(-1);
	}

	//FREE MEMORY
	free(meanSpect);
    free(count);
    free(Cov);
    free(Corr);
    free(CovEigVal);
    free(CorrEigVal);
    free(work);
    free(U);
    free(VT);
	free(image);
	free(interleave);
	free(waveUnit);
	free(wavelength);

	return 0;
}
