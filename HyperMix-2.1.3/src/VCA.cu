#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <time.h>
#include <limits>
#include <cublas.h>

using namespace std;

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define MAXLINE 200
#define MAXCAD 200
#define EPSILON 1.11e-16

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
    char line[MAXLINE];
    char value [MAXLINE];

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
    char line[MAXLINE];
    char value [MAXLINE];

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
int readValueResults(char* filename)
{
	FILE *fp;
	int in, error;
	int value;
	if ((fp=fopen(filename,"rb"))!=NULL)
	{
		fseek(fp,0L,SEEK_SET);
		error = fread(&value,1,sizeof(int),fp);
		if(error != sizeof(int)) in = -1;
		else in = (int)value;
		fclose(fp);
	}
	else return -1;

	return in;
}

/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
int writeResult(double *image, const char* filename, int lines, int samples, int bands, int dataType, char* interleave)
{

	short int *imageSI;
	float *imageF;
	double *imageD;

	int i,j,k,op;

	if(interleave == NULL)
		op = 0;
	else
	{
		if(strcmp(interleave, "bsq") == 0) op = 0;
		if(strcmp(interleave, "bip") == 0) op = 1;
		if(strcmp(interleave, "bil") == 0) op = 2;
	}

	if(dataType == 2)
	{
		imageSI = (short int*)malloc(lines*samples*bands*sizeof(short int));

        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					imageSI[i] = (short int)image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageSI[j*bands + i] = (short int)image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageSI[i*bands*samples + (j*samples + k)] = (short int)image[j*lines*samples + (i*samples + k)];
				break;
        }

	}

	if(dataType == 4)
	{
		imageF = (float*)malloc(lines*samples*bands*sizeof(float));
        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					imageF[i] = (float)image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageF[j*bands + i] = (float)image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageF[i*bands*samples + (j*samples + k)] = (float)image[j*lines*samples + (i*samples + k)];
				break;
        }
	}

	if(dataType == 5)
	{
		imageD = (double*)malloc(lines*samples*bands*sizeof(double));
        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					imageD[i] = image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageD[j*bands + i] = image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageD[i*bands*samples + (j*samples + k)] = image[j*lines*samples + (i*samples + k)];
				break;
        }
	}

    FILE *fp;
    if ((fp=fopen(filename,"wb"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
	    switch(dataType)
	    {
	    case 2: fwrite(imageSI,1,(lines*samples*bands * sizeof(short int)),fp); free(imageSI); break;
	    case 4: fwrite(imageF,1,(lines*samples*bands * sizeof(float)),fp); free(imageF); break;
	    case 5: fwrite(imageD,1,(lines*samples*bands * sizeof(double)),fp); free(imageD); break;
	    }
	    fclose(fp);


	    return 0;
    }

    return -3;
}

/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
int writeHeader(char* filename, int lines, int samples, int bands, int dataType,
		char* interleave, int byteOrder, char* waveUnit, double* wavelength)
{
    FILE *fp;
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
		fseek(fp,0L,SEEK_SET);
		fprintf(fp,"ENVI\ndescription = {\nExported from MATLAB}\n");
		if(samples != 0) fprintf(fp,"samples = %d", samples);
		if(lines != 0) fprintf(fp,"\nlines   = %d", lines);
		if(bands != 0) fprintf(fp,"\nbands   = %d", bands);
		if(dataType != 0) fprintf(fp,"\ndata type = %d", dataType);
		if(interleave != NULL) fprintf(fp,"\ninterleave = %s", interleave);
		if(byteOrder != 0) fprintf(fp,"\nbyte order = %d", byteOrder);
		if(waveUnit != NULL) fprintf(fp,"\nwavelength units = %s", waveUnit);
		if(waveUnit != NULL)
		{
			fprintf(fp,"\nwavelength = {\n");
			for(int i=0; i<bands; i++)
			{
				if(i==0) fprintf(fp, "%f", wavelength[i]);
				else
					if(i%3 == 0) fprintf(fp, ", %f\n", wavelength[i]);
					else fprintf(fp, ", %f", wavelength[i]);
			}
			fprintf(fp,"}");
		}
		fclose(fp);
		return 0;
    }
    return -3;
}

/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
void mean(double* matrix, int rows, int cols, int dim, double* out)
{
	int i,j;

	if(dim == 1)
	{
		for(i=0; i<cols; i++) out[i] = 0;

		for(i=0; i<cols; i++)
			for(j=0; j<rows; j++)
				out[i] += matrix[j*cols + i];

		for(i=0; i<cols; i++) out[i] = out[i]/cols;
	}
	else
	{
		for(i=0; i<rows; i++) out[i] = 0;

		for(i=0; i<rows; i++)
			for(j=0; j<cols; j++)
				out[i] += matrix[i*cols + j];

		for(i=0; i<rows; i++) out[i] = out[i]/rows;

	}
}

/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
void pinv(double * A, int n, int m)
{
	int dimS = min(n,m);
	int LDU = n;
	int LDVT = m;
	int i,j;
	double alpha = 1, beta = 0;
	int lwork  = max(1,max(3*min(m, n)+max(m,n),5*min(m,n))) , info;
	double *work  = (double*)malloc(lwork*sizeof(double));

	double *S = (double*) malloc(dimS * sizeof(double));//eigenvalues
	double *U = (double*) malloc(LDU * n * sizeof(double));//eigenvectors
	double *VT = (double*) malloc(LDVT * m * sizeof(double));//eigenvectors

	dgesvd_("S", "S", &n, &m, A, &n, S, U, &LDU, VT, &LDVT, work, &lwork, &info);

	double maxi = S[0];
	for(i=1; i<dimS; i++)
		if(maxi<S[i]) maxi = S[i];

	double tolerance = EPSILON*max(n,m)*maxi;

	int rank = 0;
	for (i=0; i<dimS; i++)
	{
		if (S[i] > tolerance)
		{
			rank ++;
			S[i] = 1.0 / S[i];
		}
	}

	/*Compute the pseudo inverse */
	/* Costly version with full DGEMM*/
	double * Utranstmp = (double*)malloc(n * m * sizeof(double));
	for (i=0; i<dimS; i++)
		for (j=0;  j<n; j++) Utranstmp[i + j * m] = S[i] * U[j + i * n];

	for (i=dimS;  i<m; i++)
		for (j=0;  j<n; j++) Utranstmp[i + j * m] = 0.0;

	dgemm_("T", "N", &m, &n, &m, &alpha, VT, &m, Utranstmp, &m, &beta, A, &m);


	free(U);
	free(VT);
	free(Utranstmp);
	free(S);
}
/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char *argv[])//(double *image, int lines, int samples, int bands, int targets, double SNR, double *endmembers)
{
	/*
	 * PARAMETERS
	 * argv[1]: Input image file
	 * argv[2]: number of endmembers to be extracted
	 * argv[3]: Signal noise ratio (SNR)
	 * argv[4]: estimated endmember signatures obtained
	 * */

	if(argc != 5)
	{
		printf("EXECUTION ERROR VCA Parallel: Parameters are not correct.");
		printf(" ./VCA [Image Filename] [Number of endmembers] [SNR] [Output endmembers]");
		fflush(stdout);
		exit(-1);
	}
	// Input parameters:
	char image_filename[MAXCAD];
	char header_filename[MAXCAD];

	strcpy(image_filename,argv[1]);
	strcpy(header_filename,argv[1]);
	strcat(header_filename,".hdr");



	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)malloc(MAXCAD*sizeof(char));
	waveUnit = (char*)malloc(MAXCAD*sizeof(char));

	// Load image
	int error = readHeader1(header_filename, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("\nEXECUTION ERROR VCA Parallel: Error 1 reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double* wavelength = (double*)malloc(bands*sizeof(double));

	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("\nEXECUTION ERROR VCA Parallel: Error 2 reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double *image_vector = (double *) malloc (sizeof(double)*(lines*samples*bands));

	error = loadImage(argv[1],image_vector, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("\nEXECUTION ERROR VCA Parallel: Error reading image file: %s.", argv[1]);
		fflush(stdout);
		exit(-1);
	}

	//***TARGETS VALUE OR LOAD VALUE****
	int targets;
	if(strstr(argv[2], "/") == NULL)
		targets = atoi(argv[2]);
	else
	{
		targets = readValueResults(argv[2]);
		fflush(stdout);
		if(targets == -1)
		{
			printf("EXECUTION ERROR IEA Iterative: Targets was not set correctly file: %s.", argv[2]);
			fflush(stdout);
			exit(-1);
		}
	}
	//**********************************

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	int i, j, lines_samples = lines*samples;
	double alpha =1, beta = 0;

	double *Ud = (double*) calloc(bands * targets , sizeof(double));
	double *x_p = (double*) calloc(lines_samples * targets , sizeof(double));
	double *y = (double*) calloc(lines_samples * targets , sizeof(double));
	double *R_o = (double*)calloc(bands*lines_samples,sizeof(double));
	double *r_m = (double*)calloc(bands,sizeof(double));
	double *svdMat = (double*)calloc(bands*bands,sizeof(double));
	double *D = (double*) calloc(bands , sizeof(double));//eigenvalues
	double *U = (double*) calloc(bands * bands , sizeof(double));//eigenvectors
	double *VT = (double*) calloc(bands * bands , sizeof(double));//eigenvectors
	double *endmembers = (double*) calloc(targets * bands , sizeof(double));
	double *Rp = (double*)calloc(bands*lines_samples,sizeof(double));
	double *u = (double*)calloc(targets,sizeof(double));
	double *sumxu = (double*)calloc(lines_samples,sizeof(double));
	int* index = (int*)calloc(targets,sizeof(int));
	double *w = (double*)calloc(targets,sizeof(double));
	double *A = (double*)calloc(targets*targets,sizeof(double));
	double *A2 = (double*)calloc(targets*targets,sizeof(double));
	double *aux = (double*)calloc(targets*targets,sizeof(double));
	double *f = (double*)calloc(targets,sizeof(double));

	double* R_oD;
	double* svdMatD;
	double* UdD;
	double *x_pD;
	double *RpD;
	double *imageD;

	int lwork  = MAX(1,MAX(3*MIN(bands, bands)+MAX(bands,bands),5*MIN(bands,bands))) , info;
	double *work  = (double*)malloc(lwork*sizeof(double));

	double SNR = atof(argv[3]), SNR_es;
	double sum1, sum2, powery, powerx, mult = 0;

	for(i=0; i<bands; i++)
	{
		for(j=0; j<lines_samples; j++)
			r_m[i] += image_vector[i*lines_samples+j];

		r_m[i] /= lines_samples;

		for(j=0; j<lines_samples; j++)
			R_o[i*lines_samples+j] = image_vector[i*lines_samples+j] - r_m[i];
	}

	cudaMalloc((void**)&R_oD, bands*lines_samples*sizeof(double));
	cudaMemcpy((void*)R_oD, R_o, bands*lines_samples*sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&svdMatD, bands*bands*sizeof(double));
	cublasDgemm('T', 'N', bands, bands, lines_samples, alpha, R_oD, lines_samples, R_oD, lines_samples, beta, svdMatD, bands);
	cudaMemcpy((void*)svdMat, svdMatD, bands*bands*sizeof(double), cudaMemcpyDeviceToHost);

	for(i=0; i<bands*bands; i++) svdMat[i] /= lines_samples;

	dgesvd_("S", "S", &bands, &bands, svdMat, &bands, D, U, &bands, VT, &bands, work, &lwork, &info);

	for(i=0; i<bands; i++)
		for(j=0; j<targets; j++)
			Ud[i*targets +j] = VT[i*bands +j];

	cudaMalloc((void**)&UdD, bands*targets*sizeof(double));
	cudaMemcpy((void*)UdD, Ud, bands*targets*sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&x_pD, targets*lines_samples*sizeof(double));
	cublasDgemm('N', 'T', targets, lines_samples, bands, alpha, UdD, targets, R_oD, lines_samples, beta, x_pD, targets);
	cudaMemcpy((void*)x_p, x_pD, targets*lines_samples*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(R_oD);


	sum1 =0;
	sum2 = 0;
	mult = 0;

	for(i=0; i<lines_samples*bands; i++)
	{
		sum1 += pow(image_vector[i],2);
		if(i < lines_samples*targets) sum2 += pow(x_p[i],2);
		if(i < bands) mult += pow(r_m[i],2);
	}

	powery = sum1 / lines_samples;
	powerx = sum2 / lines_samples + mult;

	SNR_es = 10 * log10((powerx - targets / bands * powery) / (powery - powerx));


	if(SNR == 0) SNR = SNR_es;
	double SNR_th = 15 + 10*log10(targets), c;

	cudaMalloc((void**)&RpD, bands*lines_samples*sizeof(double));

	if(SNR < SNR_th)
	{
		for(i=0; i<bands; i++)
			for(j=0; j<targets; j++)
				if(j<targets-1) Ud[i*targets +j] = VT[i*bands +j];
				else  Ud[i*targets +j] = 0;

		sum1 = 0;
		for(i=0; i<targets; i++)
		{
			for(j=0; j<lines_samples; j++)
			{
				if(i == (targets-1)) x_p[i*lines_samples+j] = 0;
				u[i] += pow(x_p[i*lines_samples+j], 2);
			}

			if(sum1 < u[i]) sum1 = u[i];
		}

		c = sqrt(sum1);


		cudaMemcpy((void*)UdD, Ud, bands*targets*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy((void*)x_pD, x_p, targets*lines_samples*sizeof(double), cudaMemcpyHostToDevice);
		cublasDgemm('T', 'N', bands, lines_samples, targets, alpha, UdD, targets, x_pD, targets, beta, RpD, bands);
		cudaMemcpy((void*)Rp, RpD, bands*lines_samples*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(RpD);


		for(i=0; i<bands; i++)
			for(j=0; j<lines_samples; j++)
				Rp[i*lines_samples+j] += r_m[i];

		for(i=0; i<targets; i++)
			for(j=0; j<lines_samples; j++)
				if(i<targets-1) y[i*lines_samples+j] = x_p[i*lines_samples+j];
				else y[i*lines_samples+j] = c;
	}
	else
	{
		cudaMalloc((void**)&imageD, bands*lines_samples*sizeof(double));
		cudaMemcpy((void*)imageD, image_vector, bands*lines_samples*sizeof(double), cudaMemcpyHostToDevice);
		cublasDgemm('T', 'N', bands, bands, lines_samples, alpha, imageD, lines_samples, imageD, lines_samples, beta, svdMatD, bands);
		cudaMemcpy((void*)svdMat, svdMatD, bands*bands*sizeof(double), cudaMemcpyDeviceToHost);

		for(i=0; i<bands*bands; i++) svdMat[i] /= lines_samples;

		dgesvd_("S", "S", &bands, &bands, svdMat, &bands, D, U, &bands, VT, &bands, work, &lwork, &info);

		for(i=0; i<bands; i++)
			for(j=0; j<targets; j++)
				Ud[i*targets +j] = VT[i*bands +j];

		cudaMemcpy((void*)UdD, Ud, bands*targets*sizeof(double), cudaMemcpyHostToDevice);
		cublasDgemm('N', 'T', targets, lines_samples, bands, alpha, UdD, targets, imageD, lines_samples, beta, x_pD, targets);
		cublasDgemm('T', 'N', bands, lines_samples, targets, alpha, UdD, targets, x_pD, targets, beta, RpD, bands);
		cudaMemcpy((void*)Rp, RpD, bands*lines_samples*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(svdMatD);
		cudaFree(UdD);
		cudaFree(x_pD);
		cudaFree(imageD);

		for(i=0; i<targets; i++)
		{
			for(j=0; j<lines_samples; j++)
				u[i] += x_p[i*lines_samples+j];

			for(j=0; j<lines_samples; j++)
				y[i*lines_samples+j] = x_p[i*lines_samples+j] * u[i];
		}

		for(i=0; i<lines_samples; i++)
			for(j=0; j<targets; j++)
				sumxu[i] += y[j*lines_samples+i];


		for(i=0; i<targets; i++)
			for(j=0; j<lines_samples; j++)
				y[i*lines_samples+j] /= sumxu[j];
	}

	srand(time(NULL));

	int lmax = std::numeric_limits<int>::max(), one = 1;
	A[(targets-1)*targets] = 1;

	for(i=0; i<targets; i++)
	{
		for(j=0; j<targets; j++)
		{
			w[j] = rand() % lmax;
			w[j] /= lmax;
		}

		for(j=0; j<targets*targets; j++) A2[j] = A[j];

		pinv(A2, targets, targets);

		dgemm_("N", "N", &targets, &targets, &targets, &alpha, A2, &targets, A, &targets, &beta, aux, &targets);
		dgemm_("N", "N", &targets, &one, &targets, &alpha, aux, &targets, w, &targets, &beta, f, &targets);

	    sum1 = 0;
	    for(j=0; j<targets; j++)
	    {

	    	f[j] = w[j] - f[j];
	    	sum1 += pow(f[j],2);
	    }

	    for(j=0; j<targets; j++) f[j] /= sqrt(sum1);

	    dgemm_("N", "T", &one, &lines_samples, &targets, &alpha, f, &one, y, &lines_samples, &beta, sumxu, &one);

	    sum2 = 0;

	    for(j=0; j<lines_samples; j++)
	    {
	    	if(sumxu[j] < 0) sumxu[j] *= -1;
	    	if(sum2 < sumxu[j])
	    	{
	    		sum2 = sumxu[j];
	    		index[i] = j;
	    	}
	    }

	    for(j=0; j<targets; j++)
	    	A[j*targets + i] = y[j*lines_samples+index[i]];

	    for(j=0; j<bands; j++)
	    	endmembers[j*targets+ i] = Rp[j+bands * index[i]];
	}


	//END CLOCK*****************************************
	end = clock();
	printf("Parallel VCA: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************

	strcpy(image_filename, argv[4]);
	strcpy(header_filename, image_filename);
	strcat(header_filename, ".hdr");
	error = writeHeader(header_filename, targets, 1, bands, dataType, interleave, byteOrder, waveUnit, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR VCA Parallel: Error writing header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	error = writeResult(endmembers, argv[4], targets, 1, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR VCA Parallel: Error writing image file: %s.", argv[3]);
		fflush(stdout);
		exit(-1);
	}

    free(aux);
	free(f);
	free(A);
	free(A2);
	free(w);
	free(index);
	free(y);
	free(u);
	free(Rp);
	free(x_p);
	free(R_o);
	free(svdMat);
	free(D);
	free(U);
	free(VT);
	free(work);
	free(r_m);
	free(Ud);

	free(image_vector);
	free(wavelength);
	free(interleave);
	free(waveUnit);
	free(endmembers);



	return 0;
}
