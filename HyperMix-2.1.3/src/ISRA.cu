#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>
#include <cublas.h>

#define MAXLINE 200
#define MAXCAD 200

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
__global__ void Ones (float *d_Ab, int lines_samples, int p){
	int idx=blockDim.x*blockIdx.x+threadIdx.x;
	if (idx<lines_samples*p){
		d_Ab[idx]=1;
	}
}

/*
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */
__global__ void Update_Ab (double *d_Num, double *d_Den, double *d_Ab, int lines_samples, int p){

	int idx=blockDim.x*blockIdx.x+threadIdx.x;

	if (idx < lines_samples*p)
		d_Ab[idx]= d_Ab[idx]*(d_Num[idx]/d_Den[idx]);
}


/*
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char** argv)//(float *image, int targets, int MAX_ITER, int lines, int samples, int bands, float *h_end, float * Ab){
{
	// Variables
	char header_filename[MAXCAD];

	/*
	 * PARAMETERS
	 * argv[1]: Input image file
	 * argv[2]: Endmembers file
	 * argv[3]: Max iterations
	 * argv[4]: Output abundances file
	 * */
	if(argc !=  5)
	{
		printf("EXECUTION ERROR ISRA Parallel: Parameters are not correct.");
		printf("./ISRA [Image Filename] [Endmembers file] [Number of iterations] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}

	// Input parameters:
	//READ IMAGE
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
		printf("EXECUTION ERROR ISRA Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR ISRA Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double *image_vector = (double*)malloc(lines*samples*bands*sizeof(double));
	error = loadImage(argv[1], image_vector, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR ISRA Iterative: Error reading image file: %s.", argv[1]);
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
		printf("EXECUTION ERROR ISRA Iterative: Error reading endmembers header file: %s.", header_filename);
		fflush(stdout);
		return error;
	}
	double *endmembers = (double*)malloc(targets*bandsEnd*sizeof(double));
	error = loadImage(argv[2], endmembers, targets, samplesE, bandsEnd, dataType, interleaveE);
	if(error != 0)
	{
		printf("EXECUTION ERROR ISRA Iterative: Error reading endmembers file: %s.", argv[2]);
		fflush(stdout);
		return error;
	}

	//CHECK CUDA DEVICE*******************************************************************************************
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	cudaDeviceProp properties;
	int max_multiprocessors = 0, max_device = 0;

	if (num_devices > 1)
	{
		for (device = 0; device < num_devices; device++)
		{
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors <= properties.multiProcessorCount)
			{
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		cudaSetDevice(max_device);
	}
	else if (num_devices < 1)
	{
		exit(0);
	}

	cudaGetDeviceProperties(&properties, max_device);
	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************

	int lines_samples = lines*samples;

	double *h_Num = (double*) malloc(lines_samples * targets * sizeof(double));
	double *h_aux = (double*) malloc(lines_samples * bands * sizeof(double));
	double *h_Den = (double*) malloc(lines_samples * targets * sizeof(double));
	double *abundanceVector = (double*)malloc(targets*lines_samples*sizeof(double));

	double *image_vectorD;
	double *d_end;
	double *d_Num;
	double *abundanceVectorD;
	double *d_aux;
	double *d_Den;

	cudaMalloc((void**)&image_vectorD, lines*samples*bands*sizeof(double));
	cudaMalloc((void**)&d_end, targets*samplesE*bandsEnd*sizeof(double));
	cudaMalloc((void**)&d_Num, lines*samples*targets*sizeof(double));
	cudaMalloc((void**)&abundanceVectorD, lines*samples*targets*sizeof(double));
	cudaMalloc((void**)&d_aux, lines*samples*bands*sizeof(double));
	cudaMalloc((void**)&d_Den, lines*samples*targets*sizeof(double));


	dim3 dimGrid(ceil((lines*samples*bands) / properties.maxThreadsPerBlock)+1,1,1);
	dim3 dimBlock(properties.maxThreadsPerBlock,1,1);


	cudaMemcpy((void*)image_vectorD, image_vector, lines*samples*bands*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_end, endmembers, targets*samplesE*bandsEnd*sizeof(double), cudaMemcpyHostToDevice);


	int i;

	for(i=0; i<lines_samples*targets; i++)
		abundanceVector[i]=1;



	cudaMemcpy((void*)abundanceVectorD, abundanceVector, lines*samples*targets*sizeof(double), cudaMemcpyHostToDevice);

	int it = atoi(argv[3]);
	double alpha = 1, beta = 0;

	cublasDgemm('N', 'T', lines_samples, targets, bands, alpha, image_vectorD, lines_samples, d_end, targets, beta, d_Num, lines_samples);

	for(i=0; i<it; i++)
	{
		cublasDgemm('N', 'N', lines_samples, bands, targets, alpha, abundanceVectorD, lines_samples, d_end, targets, beta, d_aux, lines_samples);
		cublasDgemm('N', 'T', lines_samples, targets, bands, alpha, d_aux, lines_samples, d_end, targets, beta, d_Den, lines_samples);

		Update_Ab<<<dimGrid, dimBlock>>>(d_Num, d_Den, abundanceVectorD, lines*samples, targets);
	}

	cudaMemcpy((void*)abundanceVector, abundanceVectorD, lines*samples*targets*sizeof(double), cudaMemcpyDeviceToHost);


	//END CLOCK*****************************************
	end = clock();
	printf("Parallel ISRA: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	strcpy(header_filename, argv[4]);
	strcat(header_filename, ".hdr");
	error = writeHeader(header_filename, lines ,samples, targets, 4, interleave, 0, NULL, NULL);
	if(error != 0)
	{
		printf("EXECUTION ERROR ISRA Parallel: Error writing endmembers header file: %s.", header_filename);
		fflush(stdout);
		return error;
	}
	error = writeResult(abundanceVector,argv[4],lines ,samples, targets, 4, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR ISRA Parallel: Error writing endmembers file: %s.", argv[4]);
		fflush(stdout);
		return error;
	}

	free(h_Den);
	free(h_Num);
	free(h_aux);
	free(abundanceVector);
	free(image_vector);
	free(endmembers);
	free(wavelength);
	free(interleave);
	free(interleaveE);
	free(waveUnit);
	cudaFree(abundanceVectorD);
	cudaFree(d_Den);
	cudaFree(d_Num);
	cudaFree(d_aux);
	cudaFree(image_vectorD);
	cudaFree(d_end);

	return 0;

}
