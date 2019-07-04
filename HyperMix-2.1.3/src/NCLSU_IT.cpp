#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAXLINE 200
#define MAXCAD 200
#define MAXITERATIONS 7
#define MAXCOUNTER 3

extern "C" int dgemm_(char *transa, char *transb, int *m, int *
		n, int *k, double *alpha, double *a, int *lda,
		double *b, int *ldb, double *beta, double *c, int
		*ldc);

extern "C" int dgemv_(char *trans, int *m, int *n, double *
		alpha, double *a, int *lda, double *x, int *incx,
		double *beta, double *y, int *incy);

extern "C" int dgetrf_(int *m, int *n, double *a, int *
	lda, int *ipiv, int *info);

extern "C" int dgetri_(int *n, double *a, int *lda, int
	*ipiv, double *work, int *lwork, int *info);

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
    char line[MAXLINE] = "";
    char value [MAXLINE] = "";

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
    char line[MAXLINE] = "";
    char value [MAXLINE] = "";

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
bool anyNegative(double* vector, int size)
{
	int i;
	for(i=0; i<size; i++)
		if(vector[i] < 0.0) return true;
	return false;
}

/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
bool anyNegative2(double* vector, int* S, int size)
{
	int i;
	for(i=0; i<size; i++)
		if(vector[i] < 0.0 && S[i] == 1) return true;
	return false;
}

/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char* argv[])
{

	/*
	 * PARAMETERS
	 *
	 * argv[1]: Input image file
	 * argv[2]: Input endmembers file
	 * argv[3]: Output abundances file
	 * */
	if(argc !=  4)
	{
		printf("EXECUTION ERROR NCLSU Iterative: Parameters are not correct.");
		printf("./NCLSU [Image Filename] [Endmembers file] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}

	//READ IMAGE
	char header_filename[MAXCAD];
	strcpy(header_filename, argv[1]);
	strcat(header_filename, ".hdr");


	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)malloc(MAXCAD*sizeof(char));
	waveUnit = (char*)malloc(MAXCAD*sizeof(char));

	// Load image
	int error = readHeader1(header_filename, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("EXECUTION ERROR NCLSU Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR NCLSU Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double *image = (double*)malloc(lines*samples*bands*sizeof(double));
	error = loadImage(argv[1], image, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR NCLSU Iterative: Error reading image file: %s.", argv[1]);
		fflush(stdout);
		exit(-1);
	}

	//READ ENDMEMBERS
	int samplesE, targets, bandsEnd;
	char *interleaveE;
	interleaveE = (char*)malloc(MAXCAD*sizeof(char));

	strcpy(header_filename, argv[2]);
	strcat(header_filename, ".hdr");
	error = readHeader1(header_filename, &targets, &samplesE, &bandsEnd, &dataType, interleaveE, NULL, NULL);
	if(error != 0)
	{
		printf("EXECUTION ERROR NCLSU Iterative: Error reading endmembers header file: %s.", header_filename);
		fflush(stdout);
		return error;
	}
	double *endmembers = (double*)malloc(targets*bandsEnd*sizeof(double));
	error = loadImage(argv[2], endmembers, targets, samplesE, bandsEnd, dataType, interleaveE);
	if(error != 0)
	{
		printf("EXECUTION ERROR NCLSU Iterative: Error reading endmembers file: %s.", argv[2]);
		fflush(stdout);
		return error;
	}

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************

	double alpha = 1, beta = 0;
	double *Et_E = (double*)malloc(targets*targets*sizeof(double));
	dgemm_("N", "T", &targets, &targets, &bandsEnd, &alpha, endmembers, &targets, endmembers, &targets, &beta, Et_E, &targets);

	int *ipiv = (int*)malloc(targets*sizeof(int));
	int info;
	int lwork = targets;
	double *work = (double*)malloc(lwork*sizeof(double));
	dgetrf_(&targets,&targets,Et_E,&targets,ipiv, &info);
	dgetri_(&targets, Et_E, &targets, ipiv,work, &lwork, &info);

	double *COMPUT = (double*)malloc(targets*bandsEnd*sizeof(double));
	dgemm_("N", "N", &targets, &bandsEnd, &targets, &alpha, Et_E, &targets, endmembers, &targets, &beta, COMPUT, &targets);

	int lines_samples = lines*samples;
	double *ABUN = (double*)malloc(targets*lines_samples*sizeof(double));

	dgemm_("N","T", &lines_samples, &targets, &bands, &alpha, image, &lines_samples, COMPUT, &targets, &beta, ABUN, &lines_samples);

	//Iterative process*********************************

	int i, j, k, one = 1;

	bool terminated, all, change;
	int iteration, counter, step, pos;
	double max;

	int *P = (int*)malloc(targets*sizeof(int));
	int *R = (int*)malloc(targets*sizeof(int));
	int *S = (int*)malloc(targets*sizeof(int));

	double *NCLSABUN = (double*)malloc(targets*sizeof(double));
	double *LSABUN = (double*)malloc(targets*sizeof(double));

	double *ALFAR = (double*)malloc(targets*sizeof(double));
	double *PHI = (double*)malloc(targets*targets*sizeof(double));
	double *LANDAK = (double*)malloc(targets*sizeof(double));
	double *DELTA = (double*)malloc(targets*targets*sizeof(double));
	double *ALFAS = (double*)malloc(targets*sizeof(double));


	for(i=0; i<lines*samples; i++)
	{
		iteration = 0;
		terminated = false;
		counter = 0;
		step = 1;

		while(!terminated && iteration < MAXITERATIONS)
		{
			switch(step)
			{
			case 1:
				for(j=0; j<targets; j++)
				{
					P[j] = 1;
					R[j] = 0;
				}
				iteration = 0;
				terminated = false;
				counter = 0;
				step = 2;
				break;
			case 2:
				for(j=0; j<targets; j++)
					NCLSABUN[j] = LSABUN[j] = ABUN[i*targets+j];
				step = 3;
				break;
			case 3:
				if(!anyNegative(NCLSABUN, targets))
				{
					for(j=0; j<targets; j++)
						ABUN[i*targets+j] = NCLSABUN[j];
					terminated = true;
				}
				else step = 4;
				break;
			case 4:
				if(iteration < MAXITERATIONS)
				{
					iteration++;
					step = 5;
				}
				else
				{
					for(j=0; j<targets; j++)
						ABUN[i*targets+j] = NCLSABUN[j];
					terminated = true;
				}
				break;
			case 5:
				for(j=0; j<targets; j++)
				{
					if(NCLSABUN[j] < 0.0 && P[j] == 1)
					{
						P[j] = 0;
						R[j] = 1;
					}
					S[j] = R[j];
				}
				step = 6;
				break;
			case 6:
				//GENERATE ALFAR**************************************
				for(j=0; j<targets; j++)
					if(R[j] == 1) ALFAR[j] = LSABUN[j];
					else ALFAR[j] = 0;
				//****************************************************
				step = 7;
				break;
			case 7:
				//CREATE PHI******************************************
				for(j=0; j<targets; j++)
					if(P[j] == 0)
						for(k=0; k<targets; k++)
							if(P[k] == 0) PHI[j*targets+k] = Et_E[j*targets+k];
							else PHI[j*targets+k] = 0;
					else
						for(k=0; k<targets; k++) PHI[j*targets+k] = 0;
				//****************************************************
				step = 8;
				break;
			case 8:
				//CREATE LANDAK***************************************
				dgetrf_(&targets,&targets,PHI,&targets,ipiv, &info);
				dgetri_(&targets, PHI, &targets, ipiv,work, &lwork, &info);
				dgemm_("N", "N", &targets, &one, &targets, &alpha, PHI, &targets, ALFAR, &targets, &beta, LANDAK, &targets);
				//****************************************************
				all = true;
				for(j=0; j<targets; j++)
					if(LANDAK[j] > 0.0) all = false;

				if(all) step = 13;
				else step = 9;
				break;
			case 9:
				max = LANDAK[0];
				pos = 0;
				for(j=1; j<targets; j++)
					if(LANDAK[j] > max)
					{
						max = LANDAK[j];
						pos = j;
					}
				R[pos] = 0;
				P[pos] = 1;
				step = 10;
				break;
			case 10:
				//GENERATE ALFAR**************************************
				for(j=0; j<targets; j++)
					if(R[j] == 1) ALFAR[j] = LSABUN[j];
					else ALFAR[j] = 0;
				//****************************************************
				//CREATE PHI******************************************
				for(j=0; j<targets; j++)
					if(P[j] == 0)
						for(k=0; k<targets; k++)
							if(P[k] == 0) PHI[j*targets+k] = Et_E[j*targets+k];
							else PHI[j*targets+k] = 0;
					else
						for(k=0; k<targets; k++) PHI[j*targets+k] = 0;
				//****************************************************
				//CREATE LANDAK***************************************
				dgetrf_(&targets,&targets,PHI,&targets,ipiv, &info);
				dgetri_(&targets, PHI, &targets, ipiv,work, &lwork, &info);
				dgemm_("N", "N", &targets, &one, &targets, &alpha, PHI, &targets, ALFAR, &targets, &beta, LANDAK, &targets);
				//****************************************************
				//CREATE DELTA****************************************
				for(j=0; j<targets; j++)
					for(k=0; k<targets; k++)
							if(P[j] == 0) DELTA[j*targets+k] = Et_E[j*targets+k];
							else DELTA[j*targets+k] = 0;
				//****************************************************
				step = 11;
				break;
			case 11:
				dgemm_("N", "N", &targets, &one, &targets, &alpha, DELTA, &targets, LANDAK, &targets, &beta, ALFAS, &targets);
				for(j=0; j<targets; j++)
					ALFAS[j] = LSABUN[j] - ALFAS[j];
				step = 12;
				break;
			case 12:
				change = false;
				if(anyNegative2(ALFAS, S, targets))
				{
					for(j=0; j<targets; j++)
						if(ALFAS[j] < 0.0 && S[j] == 1)
						{
							P[j] = 0;
							R[j] = 0;
							change = true;
						}
					if(change)
					{
						//GENERATE ALFAR**************************************
						for(j=0; j<targets; j++)
							if(R[j] == 1) ALFAR[j] = LSABUN[j];
							else ALFAR[j] = 0;
						//****************************************************
						//CREATE PHI******************************************
						for(j=0; j<targets; j++)
							if(P[j] == 0)
								for(k=0; k<targets; k++)
									if(P[k] == 0) PHI[j*targets+k] = Et_E[j*targets+k];
									else PHI[j*targets+k] = 0;
							else
								for(k=0; k<targets; k++) PHI[j*targets+k] = 0;
						//****************************************************
						//CREATE LANDAK***************************************
						dgetrf_(&targets,&targets,PHI,&targets,ipiv, &info);
						dgetri_(&targets, PHI, &targets, ipiv,work, &lwork, &info);
						dgemm_("N", "N", &targets, &one, &targets, &alpha, PHI, &targets, ALFAR, &targets, &beta, LANDAK, &targets);
						//****************************************************
					}
					counter++;
					if(counter < MAXCOUNTER) step = 6;
					else
					{
						counter = 0;
						step = 13;
					}
				}
				else
				{
					counter = 0;
					step = 13;
				}
				break;
			case 13:
				//CREATE DELTA****************************************
				for(j=0; j<targets; j++)
					for(k=0; k<targets; k++)
							if(P[j] == 0) DELTA[j*targets+k] = Et_E[j*targets+k];
							else DELTA[j*targets+k] = 0;
				//****************************************************
				step = 14;
				break;
			case 14:
				//dgemv_("N",&targets, &targets, &alpha, DELTA, &targets, LANDAK, &targets, &beta, NCLSABUN, &targets);
				dgemm_("N", "N", &targets, &one, &targets, &alpha, DELTA, &targets, LANDAK, &targets, &beta, NCLSABUN, &targets);
				for(j=0; j<targets; j++)
					NCLSABUN[j] = LSABUN[j] - NCLSABUN[j];
				step = 3;
				break;
			}
		}
	}

	//END CLOCK*****************************************
	end = clock();
	printf("Iterative NCLSU: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************

	strcpy(header_filename, argv[3]);
	strcat(header_filename, ".hdr");
	error = writeHeader(header_filename, lines,samples, targets, 4, interleave, 0, NULL, NULL);
	if(error != 0)
	{
		printf("EXECUTION ERROR NCLSU Iterative: Error writing endmembers header file: %s.", header_filename);
		fflush(stdout);
		return error;
	}

	error = writeResult(ABUN,argv[3],lines,samples, targets, 4, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR NCLSU Iterative: Error writing endmembers file: %s.", argv[3]);
		fflush(stdout);
		return error;
	}

	//FREE MEMORY***************************************
	free(Et_E);
	free(wavelength);
	free(COMPUT);
	free(ABUN);
	free(P);
	free(R);
	free(S);
	free(NCLSABUN);
	free(LSABUN);
	free(ALFAR);
	free(ALFAS);
	free(DELTA);
	free(PHI);
	free(LANDAK);
	free(ipiv);
	free(work);
	free(image);
	free(endmembers);
	free(interleaveE);
	free(interleave);
	free(waveUnit);

	return 0;
}
