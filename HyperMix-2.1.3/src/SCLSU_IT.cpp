#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAXLINE 200
#define MAXCAD 90

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
		printf("EXECUTION ERROR SCLSU Iterative: Parameters are not correct.");
		printf("./SCLSU [Image Filename] [Endmembers file] [Output Result File]");
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
		printf("EXECUTION ERROR SCLSU Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double *image = (double*)malloc(lines*samples*bands*sizeof(double));
	error = loadImage(argv[1], image, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading image file: %s.", argv[1]);
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
		printf("EXECUTION ERROR SCLSU Iterative: Error reading endmembers header file: %s.", header_filename);
		fflush(stdout);
		return error;
	}
	double *endmembers = (double*)malloc(targets*bandsEnd*sizeof(double));
	error = loadImage(argv[2], endmembers, targets, samplesE, bandsEnd, dataType, interleaveE);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading endmembers file: %s.", argv[2]);
		fflush(stdout);
		return error;
	}

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
//	MatrixType I(N_endmembers, N_endmembers);
//	I.fill(0);
//	for(i=0; i<N_endmembers;i++)
//		I.put(i,i,1);
//
//	//INVERSE (E_t * E)⁻¹
//	//E_t
//	MatrixType END_T = END.transpose();//(n_end x bands)
//	//E_t * E
//	MatrixType PRODUCT = END_T * END; //(n_end x n_end)
//	//(E_t * E)⁻¹
//	MatrixType INVERSE = vnl_matrix_inverse<double>(PRODUCT);//(n_end x n_end)
//
//	//[1_t * (E_t *E)⁻¹ *1]⁻¹ == X
//	//ONE
//	MatrixType ONE(N_endmembers,1);
//	ONE.fill(1);
//	//ONET
//	MatrixType ONE_T = ONE.transpose(); (1, N_endmembers)
//	MatrixType X = vnl_matrix_inverse<double>(ONE_T * INVERSE * ONE); (1,1)
//
//	//SCLS = [I - INV*ONE*X*ONET] [INV*ENDT*PIXEL(i,j)] + [INV*ONE*X]
//	MatrixType A = I-(INVERSE*ONE*X*ONE_T);
//	VectorType B = INVERSE*ONE*X.get_row(0);
//
//	ABUNDANCES.set_size(N_endmembers,N_samples*N_lines);
//
//	for(i=0;i<N_samples*N_lines;i++)
//		ABUNDANCES.set_column(i,A*(INVERSE*END_T*PIXELS.get_column(i)) + B);
	int i, j, one = 1;
	double alpha = 1, beta = 0;
	double *Et_E = (double*)malloc(targets*targets*sizeof(double));

	dgemm_("N", "T", &targets, &targets, &bandsEnd, &alpha, endmembers, &targets, endmembers, &targets, &beta, Et_E, &targets);

	int *ipiv = (int*)malloc(targets*sizeof(int));
	int info;
	int lwork = targets;
	double *work = (double*)malloc(lwork*sizeof(double));
	dgetrf_(&targets,&targets,Et_E,&targets,ipiv, &info);
	dgetri_(&targets, Et_E, &targets, ipiv,work, &lwork, &info);

	double* ONE = (double*)malloc(targets*sizeof(double));
	for(i=0; i<targets; i++) ONE[i] = 1;
	double* AUX = (double*)calloc(targets,sizeof(double));
	dgemm_("N", "N", &one, &targets, &targets, &alpha, ONE, &one, Et_E, &targets, &beta, AUX, &one);

	double Y = 0;
	for(i=0; i<targets; i++) Y += AUX[i];
	Y = 1 / Y;
	for(i=0; i<targets; i++) AUX[i] = Y;

	double* AUX2 = (double*)calloc(targets,sizeof(double));
	dgemm_("N", "N", &one, &targets, &targets, &alpha, AUX, &one, Et_E, &targets, &beta, AUX2, &one);


	double* I = (double*)calloc(targets*targets,sizeof(double));
	for(i=0; i<targets; i++)
		for(j=0; j<targets; j++)
			if(i == j) I[i*targets+j] = 1 - AUX2[j];
			else I[i*targets+j] = -AUX2[j];



	double* A = (double*)calloc(targets*targets,sizeof(double));
	dgemm_("N", "N", &targets, &targets, &targets, &alpha, I, &targets, Et_E, &targets, &beta, A, &targets);


	double* B = (double*)calloc(targets*bands,sizeof(double));
	dgemm_("N", "N", &targets, &bands, &targets, &alpha, A, &targets, endmembers, &targets, &beta, B, &targets);


	dgemm_("N", "N", &targets, &one, &targets, &alpha, Et_E, &targets, ONE, &targets, &beta, AUX, &targets);
	for(i=0; i<targets; i++) AUX[i] *= Y;


	double* ABUN = (double*)calloc(lines*samples*targets, sizeof(double));
	double* PIXEL = (double*)calloc(bands, sizeof(double));

	for(i=0; i<lines*samples; i++)
	{
		for(j=0; j<bands; j++) PIXEL[j] = image[j*lines*samples+i];
		dgemm_("N", "N", &targets, &one, &bands, &alpha, B, &targets, PIXEL, &bands, &beta, AUX2, &targets);

		for(j=0; j<targets; j++) ABUN[j*lines*samples + i] = AUX2[j] + AUX[j];
	}
	//END CLOCK*****************************************
	end = clock();
	printf("Iterative SCLSU: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************

	strcpy(header_filename, argv[3]);
	strcat(header_filename, ".hdr");
	error = writeHeader(header_filename, lines,samples, targets, 4, interleave, 0, NULL, NULL);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error writing endmembers header file: %s.", header_filename);
		fflush(stdout);
		return error;
	}

	error = writeResult(ABUN,argv[3],lines,samples, targets, 4, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error writing endmembers file: %s.", argv[3]);
		fflush(stdout);
		return error;
	}

	//FREE MEMORY***************************************
	free(Et_E);
	free(ipiv);
	free(work);
	free(ONE);
	free(AUX);
	free(AUX2);
	free(I);
	free(A);
	free(B);
	free(PIXEL);
	free(ABUN);
	free(wavelength);
	free(image);
	free(endmembers);
	free(interleaveE);
	free(interleave);
	free(waveUnit);

	return 0;
}
