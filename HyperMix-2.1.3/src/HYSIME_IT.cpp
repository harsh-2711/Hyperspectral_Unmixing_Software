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

#define SMALL 0.000001

extern "C" int dgemm_(char *transa, char *transb, int *m, int *
		n, int *k, double *alpha, double *a, int *lda,
		double *b, int *ldb, double *beta, double *c, int
		*ldc);

extern "C" int dgetrf_(int *m, int *n, double *a, int *
	lda, int *ipiv, int *info);

extern "C" int dgetri_(int *n, double *a, int *lda, int
	*ipiv, double *work, int *lwork, int *info);

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
    char line[MAXLINE] ="";
    char value [MAXLINE] ="";

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
    char line[MAXLINE] ="";
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
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */
void avg_X(double *X, int lines_samples, int num_bands, double *meanSpect)
{

	int i,j;
    float mean;

    for(i=0; i<num_bands; i++){
		mean=0;
        for(j=0; j<lines_samples; j++){
			mean=mean+(X[(i*lines_samples)+j]);
        }
		mean=mean/lines_samples;
        meanSpect[i]=mean;

        for(j=0; j<lines_samples; j++){
			X[(i*lines_samples)+j]=X[(i*lines_samples)+j]-mean;
        }
	}

}

/*
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */
void Covariance(double *M, double *Cov, int N, int p, double *meanSpect)
{
	double alphaN = 1/N, beta = 0;
    avg_X(M, N, p, meanSpect);
    dgemm_("T", "N", &p, &p, &N, &alphaN, M, &N, M, &N, &beta, Cov, &p);
}

/*
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */
void Correlation(double *Cov, double *Corr, int p, double *meanSpect)
{
    int j, k;
    //double meanMatrix;
    //meanMatrix = (double*) malloc(p * p * sizeof(double));
    for(j=0; j<p; j++){//col
        for(k=0; k<p; k++){//fil
            Corr[p*j+k]=Cov[p*j+k]+(meanSpect[k] * meanSpect[j]);
        }
    }
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
	 * argv[2]: Output Result File
	 *
	 * */

	if(argc != 3)
	{
		printf("EXECUTION ERROR HYSIME Iterative: Parameters are not correct.");
		printf("./HYSIME [Image Filename] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}

	//READ IMAGE
	char cad[MAXCAD];
	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char interleave[MAXCAD], waveUnit[MAXCAD];

	strcpy(cad,argv[1]); // Second parameter: Header file:
	strcat(cad,".hdr");
	int error = readHeader1(cad, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("EXECUTION ERROR HYSIME Iterative: Error 1 reading header file: %s.", cad);
		fflush(stdout);
		exit(-1);
	}

	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(cad,argv[1]);
	strcat(cad,".hdr");
	error = readHeader2(cad, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR HYSIME Iterative: Error 2 reading header file: %s.", cad);
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

	int lines_samples = lines*samples, one =1, res = 0, i, j;
    double alpha = 1, beta = 0;


    double *Z = (double*)calloc(lines_samples*bands, sizeof(double));
    double *R = (double*)calloc(bands*bands, sizeof(double));
    double *R_INV = (double*)calloc(bands*bands, sizeof(double));
    double *R_INV_AA = (double*)calloc(bands*bands, sizeof(double));
    double *R_iA = (double*)calloc(bands, sizeof(double));
    double *R_INV_iA = (double*)calloc(bands, sizeof(double));
    double R_INV_ii;
    double *beta_i = (double*)calloc(bands, sizeof(double));
    double *Z_i = (double*)calloc(lines_samples, sizeof(double));
    double *E_i = (double*)calloc(lines_samples, sizeof(double));
    double *E = (double*)calloc(lines_samples*bands, sizeof(double));
    double *meanSpect =  (double*)calloc(bands, sizeof(double));
    double *Cov =  (double*)calloc(bands*bands, sizeof(double));
    double* RN = (double*)calloc(bands*bands, sizeof(double));
    double* RX = (double*)calloc(bands*bands, sizeof(double));
    double *eigen_RX = (double*)calloc(bands, sizeof(double));
    double *delta = (double*)calloc(bands, sizeof(double));
    double *micro = (double*)calloc(bands, sizeof(double));
    double *sigma = (double*)calloc(bands, sizeof(double));

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************

	for(i=0; i<bands; i++)
		for(j=0; j<lines_samples; j++)
			Z[j*bands+i] = image[i*lines_samples + j];

	dgemm_("T", "N", &bands, &bands, &lines_samples, &alpha, Z, &lines_samples, Z, &lines_samples, &beta, R, &bands);

	for(i=0; i<bands*bands; i++) R_INV[i] = R[i];

	int *ipiv = (int*)malloc(bands*sizeof(int));
	int info;
	int lwork = bands;
	double *work = (double*)malloc(lwork*sizeof(double));
	dgetrf_(&bands,&bands,R_INV,&bands,ipiv, &info);
	dgetri_(&bands, R_INV, &bands, ipiv,work, &lwork, &info);


	for(i=0; i<bands; i++)
	{
		R_INV_ii = R_INV[i*bands+i];

		for(j=0; j<bands; j++)
		{
			if(j != i) R_INV_iA[j] = R_INV[i*bands+j];
			else R_INV_iA[j] = 0;
			if(j != i) R_iA[j] = R[i*bands+j];
			else R_iA[j] = 0;
		}

		for(j=0; j<lines_samples; j++)
		{
			Z_i[j] = Z[j*bands+i];
			Z[j*bands+i] = 0;
		}

		dgemm_("T", "N", &bands, &bands, &one, &alpha, R_INV_iA, &one, R_INV_iA, &one, &beta, R_INV_AA, &bands);

		for(j=0; j<bands*bands; j++)
			if(j/bands != i && j%bands != i) R_INV_AA[j] = R_INV[j] - R_INV_AA[j]/R_INV_ii;
			else R_INV_AA[j] = - R_INV_AA[j]/R_INV_ii;

		dgemm_("N","T",&bands, &one, &bands, &alpha, R_INV_AA, &bands, R_iA, &one, &beta, beta_i, &bands);

		dgemm_("N","N",&lines_samples, &one, &bands, &alpha, Z, &lines_samples, beta_i, &bands, &beta, E_i, &lines_samples);

		for(j=0; j<lines_samples; j++)
		{
			E[j*bands+i] = Z_i[j] - E_i[j];
			Z[j*bands+i] = Z_i[j];
		}
	}

	Covariance(E, Cov, lines_samples, bands, meanSpect);
	Correlation(Cov, RN, bands, meanSpect);

	for(i=0; i<lines_samples*bands; i++)
	{
		E[i] = image[i] - E[i];
		if(i<bands*bands) R[i] /= lines_samples;
	}

	Covariance(E, Cov, lines_samples, bands, meanSpect);
	Correlation(Cov, RX, bands, meanSpect);

    double* U = (double*)calloc(bands*bands, sizeof(double));
    double* VT = (double*)calloc(bands*bands, sizeof(double));
    lwork = MAX(1,MAX(3*MIN(bands, bands)+MAX(bands,bands),5*MIN(bands,bands)));
    free(work);
    work = (double*)malloc(lwork*sizeof(double));
 	dgesvd_("A", "A", &bands, &bands, RX, &bands, eigen_RX, U, &bands, VT, &bands, work, &lwork, &info);

    for(i=0; i<bands; i++)
    {
    	for(j=0; j<bands; j++)
    		eigen_RX[j] = VT[i*bands+j];

    	dgemm_("T","N",&one, &bands, &bands, &alpha, eigen_RX, &bands, R, &bands, &beta, meanSpect, &one);
    	dgemm_("T","N",&one, &bands, &bands, &alpha, eigen_RX, &bands, RN, &bands, &beta, R_iA, &one);

    	for(j=0; j<bands; j++)
    	{
    		micro[i] += meanSpect[j] * eigen_RX[j];
    		sigma[i] += R_iA[j] * eigen_RX[j];
    	}

    	delta[i] = -micro[i] + 2*sigma[i]*sigma[i];
    	if(delta[i] < 0) res++;
    }

	//END CLOCK*****************************************
	end = clock();
	printf("Iterative HYSIME: %f segundos res: %d ", (double)(end - start) / CLOCKS_PER_SEC, res);
	fflush(stdout);
	//**************************************************


	error = writeValueResults(argv[2], res);
	if(error != 0)
	{
		printf("EXECUTION ERROR HYSIME Iterative: Error writing results file: %s.", argv[2]);
		fflush(stdout);
		exit(-1);
	}

	free(image);
	free(Cov);
	free(work);
	free(U);
	free(VT);
	free(ipiv);
	free(wavelength);
	free(delta);
	free(micro);
	free(sigma);
	free(eigen_RX);
	free(meanSpect);
	free(RN);
	free(RX);
	free(E);
	free(E_i);
	free(Z_i);
	free(beta_i);
	free(R_iA);
	free(R_INV_iA);
	free(R_INV_AA);
	free(R_INV);
	free(R);
	free(Z);


	return 0;
}
