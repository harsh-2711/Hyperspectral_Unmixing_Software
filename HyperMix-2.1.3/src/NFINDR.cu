#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cublas.h>

#define MAXLINE 200
#define MAXCAD 200
#define MIN_DOUBLE 2.23e-308//Double mínimo
#define EPS 1.0e-10//Distancia minima entre dos números

extern "C" int dgemm_(char *transa, char *transb, int *m, int *
		n, int *k, double *alpha, double *a, int *lda,
		double *b, int *ldb, double *beta, double *c, int
		*ldc);

extern "C" int dgetrf_(int *m, int *n, double *a, int *
	lda, int *ipiv, int *info);

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
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */

int main(int argc, char *argv[])//(double *imageReduced, int *endmembersIndex, int endmembers, int num_lines, int num_samples, int lines_samples, int n_pc)
{
	// Variables
	char image_filename[MAXCAD];
	char header_filename[MAXCAD];

	/*
	 * PARAMETERS
	 * argv[1]: Input image file
	 * argv[2]: Reduced image file
	 * argv[3]: Output endmembers file
	 * */
	if(argc != 4)
	{
		printf("EXECUTION ERROR N-FINDR Iterative: Parameters are not correct.");
		printf("./NFINDR [Image Filename] [Reduced image] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}

	// Input parameters:
	strcpy(image_filename,argv[2]); // First parameter: Image file:
	strcpy(header_filename,argv[2]); // Second parameter: Header file:
	strcat(header_filename,".hdr");

	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)malloc(MAXCAD*sizeof(char));
	waveUnit = (char*)malloc(MAXCAD*sizeof(char));

	// Load image
	int error = readHeader1(header_filename, &lines, &samples, &bands, &dataType, interleave, &byteOrder, NULL);
	if(error != 0)
	{
		printf("EXECUTION ERROR NFINDR Iterative: Error reading PCA header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	int targets = bands; // Third parameter: number of targets

	double *image_vector = (double *) malloc (sizeof(double)*(lines*samples*bands));
	//Load_image
	error = loadImage(argv[2],image_vector, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR N-FINDR Iterative: Error reading PCA image file: %s.", argv[2]);
		fflush(stdout);
		return error;
	}

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	int lines_samples = lines*samples;
	int i,j,k, l;
	int* endmembersIndex = (int*)calloc(bands,sizeof(int));
	int* aleatorios = (int*)calloc(lines_samples,sizeof(int));
	double* E = (double*)calloc((targets+1)*targets, sizeof(double));
	double* copyE = (double*)calloc((targets+1)*targets, sizeof(double));

	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	int *d_pivot_array;
	error = cudaMalloc((void **)&d_pivot_array, sizeof(int));
	int *d_info_array;
	error = cudaMalloc((void **)&d_info_array, sizeof(int));

	double* copyE_d;
	error = cudaMalloc((void **)&copyE_d, (targets+1)*targets*sizeof(double));

	srand(time(NULL));
	for(i=0; i<lines_samples; i++)
		aleatorios[i] = i;

	for(i=0; i<lines_samples; i++)
	{
		j = rand()%lines_samples;
		k = aleatorios[i];
		aleatorios[i] = aleatorios[j];
		aleatorios[j] = k;
	}

	for(i=0; i<targets; i++)
	{
		E[i] = copyE[i] = 1;
		for(j=0; j<targets; j++)
			E[(j+1)*targets+i] = copyE[(j+1)*targets+i] = image_vector[j*lines_samples + aleatorios[i]];
	}

	int targets1 = targets+1;

	// Calculate volume of E****************
	cudaMemcpy((void*)copyE_d, copyE, targets*targets*sizeof(double), cudaMemcpyHostToDevice);
	cublasDgetrfBatched(handle, targets1, &copyE_d, targets1, d_pivot_array, d_info_array, 1);
	cudaMemcpy((void*)copyE, copyE_d, targets*targets*sizeof(double), cudaMemcpyDeviceToHost);

	double vold = 1, vnew, vaux, perL =1;
	int argmax;

	for(i=2; i<targets; i++) perL *= i;

	for(i=0; i<targets; i++)
		vold *= copyE[i*targets1+i];

	if(vold < 0) vold *= -1;

	vold = vold/perL;
	//***************************************

	for(i=targets; i<lines_samples; i++)
	{
		argmax = 0;
		vaux = vold;
		for(j=0; j<targets; j++)
		{
			for(k=0; k<targets; k++)
			{
				copyE[k] = 1;
				for(l=0; l<targets; l++)
					if(k != j) copyE[(l+1)*targets+k] = E[(l+1)*targets+k];
					else copyE[(l+1)*targets+k] = image_vector[l*lines_samples + aleatorios[i]];
			}

			cudaMemcpy((void*)copyE_d, copyE, targets*targets*sizeof(double), cudaMemcpyHostToDevice);
			cublasDgetrfBatched(handle, targets1, &copyE_d, targets1, d_pivot_array, d_info_array, 1);
			cudaMemcpy((void*)copyE, copyE_d, targets*targets*sizeof(double), cudaMemcpyDeviceToHost);

			vnew = 1;

			for(k=0; k<targets; k++)
				vnew *= copyE[k*targets1+k];

			if(vnew < 0) vnew *= -1;

			vnew = vnew/perL;

			if(vnew > vaux)
			{
				argmax = j;
				vaux = vnew;

			}
		}

		if(vaux > vold)
		{
			vold = vaux;
			for(l=0; l<targets; l++)
				E[(l+1)*targets+argmax] = image_vector[l*lines_samples + aleatorios[i]];

			k = aleatorios[i];
			aleatorios[i] = aleatorios[argmax];
			aleatorios[argmax] = k;
		}
	}

	for(i=0; i<targets; i++){ endmembersIndex[i] = aleatorios[i];}

	//END CLOCK*****************************************
	end = clock();
	printf("Iterative NFINDR: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************

	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");

	// Load image
	error = readHeader1(header_filename, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("EXECUTION ERROR NFINDR Iterative: Error reading image header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR NFINDR Iterative: Error reading image header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double *image_vector2 = (double *) malloc (sizeof(double)*(lines*samples*bands));
	error = loadImage(argv[1],image_vector2, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR N-FINDR Iterative: Error reading image file: %s.", argv[1]);
		fflush(stdout);
		return error;
	}

	double *h_umatrix =(double *) malloc (sizeof(double)*(bands*targets));

	for(int i=0; i<targets; i++)
	{
		for(int j=0; j<bands; j++)
			h_umatrix[j*targets+i] = image_vector2[endmembersIndex[i]+(lines_samples*j)];
	}

	strcpy(header_filename, argv[3]);
	strcat(header_filename, ".hdr");
	error = writeHeader(header_filename, targets, 1, bands, dataType, interleave, byteOrder, waveUnit, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR N-FINDR Iterative: Error writing header image file: %s.", header_filename);
		fflush(stdout);
		return error;
	}
	error = writeResult(h_umatrix,argv[3], targets, 1, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR N-FINDR Iterative: Error writing image file: %s.", argv[3]);
		fflush(stdout);
		return error;
	}

	free(image_vector2);
	free(h_umatrix);
	free(image_vector);
	free(endmembersIndex);
	free(aleatorios);
	free(E);
	free(copyE);
    free(wavelength);
	free(interleave);
	free(waveUnit);


    return 0;
}
