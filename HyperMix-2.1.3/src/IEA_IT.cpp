#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAXLINE 200
#define MAXCAD 200

extern "C" int dgemm_(char *transa, char *transb, int *m, int *
		n, int *k, double *alpha, double *a, int *lda,
		double *b, int *ldb, double *beta, double *c, int
		*ldc);

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

double* LSU(double* endmembers, double* image, int lines, int samples, int bands, int targets)
{
	// E^t * E
	double alpha = 1, beta = 0;
	double *Et_E = (double*)malloc(targets*targets*sizeof(double));
	dgemm_("T", "N", &targets, &targets, &bands, &alpha, endmembers, &bands, endmembers, &bands, &beta, Et_E, &targets);

	//inv(E^t * E)
	int *ipiv = (int*)malloc(targets*sizeof(int));
	int info;
	int lwork = targets;
	double *work = (double*)malloc(lwork*sizeof(double));
	dgetrf_(&targets,&targets,Et_E,&targets,ipiv, &info);
	dgetri_(&targets, Et_E, &targets, ipiv,work, &lwork, &info);


	//inv * E^t
	double *COMPUT = (double*)malloc(targets*bands*sizeof(double));
	dgemm_("N", "T", &targets, &bands, &targets, &alpha, Et_E, &targets, endmembers, &bands, &beta, COMPUT, &targets);

	// COMPUT * image
	int lines_samples = lines*samples;
	double *ABUN = (double*)malloc(targets*lines_samples*sizeof(double));

	dgemm_("N","T", &lines_samples, &targets, &bands, &alpha, image, &lines_samples, COMPUT, &targets, &beta, ABUN, &lines_samples);


	free(Et_E);
	free(ipiv);
	free(work);
	free(COMPUT);

	return ABUN;
}

/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
double* RMSE(double* A, double* B, int rows, int cols)
{
	double res, sum = 0;
	double* error = (double*)malloc(cols*sizeof(double));

	for(int i=0; i<cols; i++)
	{
		for(int j=0; j<rows; j++)
		{
			res = A[j*rows+i] - B[j*rows+i];
			res = res*res;
			sum += res;
		}
		sum = sqrt(sum) / rows;
		error[i] = sum;
		sum = 0;
	}
	return error;
}

/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char* argv[])
{
	// Variables
	char image_filename[MAXCAD];
	char header_filename[MAXCAD];
	int	i, j, k;

	/*
	 * PARAMETERS
	 * argv[1]: Input image file
	 * argv[2]: number of endmembers to be extracted
	 * argv[3]: Output endmembers file
	 * */
	if(argc != 4)
	{
		printf("EXECUTION ERROR IEA Iterative: Parameters are not correct.");
		printf("./IEA [Image Filename] [Number of endmembers] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}

	// Input parameters:
	strcpy(image_filename,argv[1]); // First parameter: Image file:
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");

	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)malloc(MAXCAD*sizeof(char));
	waveUnit = (char*)malloc(MAXCAD*sizeof(char));

	// Load image
	int error = readHeader1(header_filename, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("EXECUTION ERROR IEA Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR IEA Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double *image_vector = (double *) malloc (sizeof(double)*(lines*samples*bands));

	//Load_image
	error = loadImage(argv[1],image_vector, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR IEA Iterative: Error reading image file: %s.", argv[1]);
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
	double *umatrix=(double *) malloc (sizeof(double)*(bands*(targets+1)));
	double* meanVector = (double*)malloc(bands*sizeof(double));
	double* reconstructed = (double*)malloc(lines*samples*bands*sizeof(double));
	double alpha = 1, beta = 0;
	int one = 1, lines_samples = lines*samples;

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	//Initial endmember calculation
	for(i=0; i<bands; i++)
		meanVector[i] = 0;

	for(i=0; i<lines*samples; i++)
		for(j=0; j<bands; j++)
			meanVector[j] += image_vector[j*lines_samples+i];

	for(i=0; i<bands; i++)
		meanVector[i] = meanVector[i]/(lines*samples); //First false endmember

	double* abundances = LSU(meanVector, image_vector, lines, samples, bands, 1);

	dgemm_("N", "N", &bands, &lines_samples, &one, &alpha, meanVector, &bands, abundances, &one, &beta, reconstructed, &bands);

	double* errorVector = RMSE(image_vector, reconstructed, bands, lines_samples);

	double max = errorVector[0];
	int arg = 0;
	for(i=1; i<lines_samples; i++)
	{
		if(max < errorVector[i])
		{
			max = errorVector[i];
			arg = i;
		}
	}

	for(j=0; j<bands; j++)
	{
		umatrix[j*(targets+1) + 0] = meanVector[j];
		umatrix[j*(targets+1) + 1] = image_vector[j*lines_samples + arg]; //First endmember
	}
	//Iterative processs
	double* endmembers;

	for(k=2; k<=targets; k++)
	{
		endmembers = (double*)malloc(bands*k*sizeof(double));
		for(i=0; i<k; i++)
			for(j=0; j<bands; j++)
				endmembers[j*k+i] = umatrix[j*(targets+1) + i];

		free(abundances);
		abundances = LSU(endmembers, image_vector, lines, samples, bands, k);

		dgemm_("N", "N", &bands, &lines_samples, &k, &alpha, endmembers, &bands, abundances, &k, &beta, reconstructed, &bands);

		free(errorVector);
		errorVector = RMSE(image_vector, reconstructed, bands, lines_samples);

		max = errorVector[0];
		arg = 0;
		for(i=1; i<lines_samples; i++)
		{
			if(max < errorVector[i])
			{
				max = errorVector[i];
				arg = i;
			}
		}
		for(j=0; j<bands; j++)
			umatrix[j*(targets+1) + k] = image_vector[j*lines_samples + arg];

		free(endmembers);
	}


	//END CLOCK*****************************************
	end = clock();
	printf("Iterative IEA: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	double *umatrix2=(double *) malloc (sizeof(double)*(bands*targets));

	for(j=1;j<=targets; j++)
		for(i=0; i<bands; i++)
			umatrix2[i*targets+ (j-1)] = umatrix[i*(targets+1) + j];

	strcpy(image_filename, argv[3]);
	strcpy(header_filename, image_filename);
	strcat(header_filename, ".hdr");
	error = writeHeader(header_filename, targets, 1, bands, dataType, interleave, byteOrder, waveUnit, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR IEA Iterative: Error writing header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	error = writeResult(umatrix2,argv[3], targets, 1, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR IEA Iterative: Error writing image file: %s.", argv[3]);
		fflush(stdout);
		exit(-1);
	}

	free(image_vector);
	free(meanVector);
	free(abundances);
	free(errorVector);
	free(reconstructed);
	free(umatrix);
	free(umatrix2);
	free(interleave);
	free(waveUnit);
	free(wavelength);

	return 0;
}
