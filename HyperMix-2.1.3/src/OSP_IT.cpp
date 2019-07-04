//*********************************************************************
//@ATGP_FCLSU.c
//********************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

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
int writeHeader(const char* filename, int lines, int samples, int bands, int dataType,
		char* interleave, int byteOrder, char* waveUnit, double* wavelength)
{
    FILE *fp = NULL;
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
long int getPixelMaxBright(double *image_vector, int lines, int samples, int bands){
	int i, j;
	double max_local=0;
	double bright = 0, value = 0.0;
	int lines_samples = lines*samples;

	int pos;

	for(i=0; i < lines_samples; i+=1){
		for (j = 0; j < bands; j+=1){
			value = image_vector[i+(j*lines_samples)];
			bright +=value;
		}
		bright*=bright;
		if(bright > max_local){
			max_local = bright;
			pos = i;
		}
		bright=0;
	}

	return pos;
}


/*
 * Author: Sergio Bernabe Garcia
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char** argv ){

	// Variables
	char header_filename[MAXCAD] = "";
	int	i, j, k, iter;
	double max_local = 0;

	/*
	 * PARAMETERS
	 * argv[1]: Input image file
	 * argv[2]: number of endmembers to be extracted
	 * argv[3]: Output endmembers file
	 * */

	if(argc != 4)
	{
		printf("EXECUTION ERROR ATGP Iterative: Parameters are not correct.");
		printf("./PCA [Image Filename] [Number of endmembers] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}
	// Input parameters:
	strcpy(header_filename,argv[1]);
	strcat(header_filename,".hdr");

	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)calloc(MAXCAD,sizeof(char));
	waveUnit = (char*)calloc(MAXCAD,sizeof(char));

	// Load image
	int error = readHeader1(header_filename, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("EXECUTION ERROR ATGP Iterative: Error 1 reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double* wavelength = (double*)malloc(bands*sizeof(double));

	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR ATGP Iterative: Error 2 reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double *image_vector = (double *) malloc (sizeof(double)*(lines*samples*bands));

	error = loadImage(argv[1],image_vector, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR ATGP Iterative: Error reading image file: %s.", argv[1]);
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
			printf("EXECUTION ERROR ATGP Iterative: Targets was not set correctly file: %s.", argv[2]);
			fflush(stdout);
			exit(-1);
		}
	}
	//**********************************

	// Initialize data structures
	double *pixel = (double *) calloc (bands, sizeof(double));
	double *umatrix=(double *) calloc ((bands*targets),sizeof(double)); // Matriz de proyección (creciente)

	//Gram-Schmidt
	double *w=(double *) calloc (bands,sizeof(double));
	double *u=(double *) calloc (bands*targets,sizeof(double));
	double *c1=(double *) calloc (targets, sizeof(double));
	double *c2=(double *) calloc (targets, sizeof(double));
	double *f=(double *) calloc ((bands*targets),sizeof(double));
	double *auxV1=(double *) calloc ((bands*targets),sizeof(double));
	double *mult=(double *) calloc ((targets*bands),sizeof(double));
	double valorNum=0, valorFa=0, valor=0, valor_aux=0, valorFin=0;
	double valorDen=0;

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************

	for(j=0; j<bands*targets; j+=1)
		umatrix[j]=0;
	for(j=0;j<bands;j+=1)
		w[j]=1;


	//-->ATGP
	//////////////////////////////FIRST TARGET - The brightest pixel is calculated/////////////////////////////////
	int pos_abs = getPixelMaxBright(image_vector, lines, samples, bands);

	for (k = 0; k < bands; k+=1)
		umatrix[k*targets]= image_vector[pos_abs+(k*lines*samples)]; //max bright image_vector


	//////////////////////////////////////////SECOND TARGET//////////////////////////////////////////////////////
	for(k=0;k<bands-1;k+=1){
		f[k]=1;
		valorFa += umatrix[k*targets];
	}
	for(k=0;k<bands;k+=1){
		u[k]=umatrix[k*targets];
		valorNum+=w[k]*u[k];
		valorDen+=u[k]*u[k];
	}
	f[bands-1]=-valorFa/(umatrix[(bands-1)*targets]);
	c2[1]=valorNum/valorDen;
	for(iter=0; iter < lines*samples; iter+=1){
		valor=0;
		valorFin=0;
		for (j = 0; j < bands; j+=1){
			pixel[j]=image_vector[iter+j*lines*samples];
			valor=valor+pixel[j]*f[j];
		}
		valorFin=valor*valor;
		if(valorFin>max_local){
			max_local=valorFin;
			pos_abs=iter;
		}
	}

	if(1 < (targets-1)){
		for(j=0; j<bands; j+=1){
			umatrix[j*targets+1]=image_vector[pos_abs+(j*lines*samples)]; //max bright image_vector
		}
	}

	////////////////////////////////////////////THIRD TARGET UNTIL P TARGETS/////////////////////////////////////
	// Launch the ATGP algorithm to find i-1 targets
	for(i=2;i<targets;i+=1){
		for(j=0;j<bands;j+=1)
			w[j]=1;

		//c1=(u(:,1:i-1)'*UC(:,i))'./(sum(u(:,1:i-1).*u(:,1:i-1)));
		for(j=1;j<i;j+=1){
			valor=0;
			valor_aux=0;
			for(k=0;k<bands;k+=1){
				valor=valor+u[k+bands*(j-1)]*umatrix[k*targets +(i-1)];
				auxV1[k+bands*(j-1)]=u[k+bands*(j-1)]*u[k+bands*(j-1)];
				valor_aux=valor_aux+auxV1[k+bands*(j-1)];
			}
			c1[j]=valor/valor_aux;
		}

		//u(:,i)=UC(:,i)-sum(c1(ones(1,nb),:).*u(:,1:i-1),2);
		for(j=1;j<i;j+=1){
			for(k=0;k<bands;k+=1){
				auxV1[k+bands*(j-1)]=c1[j];
				mult[k+bands*(j-1)]=auxV1[k+bands*(j-1)]*u[k+bands*(j-1)];
			}
		}
		for(k=0;k<bands;k+=1){
			valor=0;
			for(j=1;j<i;j+=1){
				valor=valor+mult[k+bands*(j-1)];
			}
			u[k+bands*(i-1)]=umatrix[k*targets+(i-1)]-valor;
		}
		//c2(i)=(w'*u(:,i))/(u(:,i)'*u(:,i));
		valor=0;
		valor_aux=0;
		for(k=0;k<bands;k+=1){
			valor=valor+w[k]*u[k+bands*(i-1)];
			valor_aux=valor_aux+u[k+bands*(i-1)]*u[k+bands*(i-1)];
		}
		c2[i]=valor/valor_aux;

		//f(:,i)=w-sum(c2(ones(1,nb),:).*u(:,1:i),2);
		for(j=1;j<=i;j+=1){
			for(k=0;k<bands;k+=1){
				auxV1[k+bands*(j-1)]=c2[j];
				mult[k+bands*(j-1)]=auxV1[k+bands*(j-1)]*u[k+bands*(j-1)];
			}
		}
		for(k=0;k<bands;k+=1){
			valor=0;
			for(j=1;j<=i;j+=1){
				valor=valor+mult[k+bands*(j-1)];
			}
			f[k+bands*(i-1)]=w[k]-valor;
		}

		//Calculation most different pixel
		max_local = 0;
		for(iter=0; iter < lines*samples; iter+=1){
			valor=0;
			valorFin=0;
			for (j = 0; j < bands; j+=1){
				pixel[j]=image_vector[iter+j*lines*samples];
				valor=valor+pixel[j]*f[j+bands*(i-1)];
			}
			valorFin=valor*valor;
			if(valorFin>max_local){
				max_local=valorFin;
				pos_abs=iter;
			}
		}

		for(j=0; j<bands; j+=1)
			umatrix[j*targets+i]=image_vector[pos_abs+(j*lines*samples)]; //max bright image_vector



	} // All i-1 targets are obtained at this point*/

	//END CLOCK*****************************************
	end = clock();
	printf("Iterative ATGP: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************

	strcpy(header_filename, argv[3]);
	strcat(header_filename, ".hdr");

	error = writeHeader(header_filename, targets, 1, bands, dataType, interleave, byteOrder, waveUnit, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR ATGP Iterative: Error writing header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	error = writeResult(umatrix, argv[3], targets, 1, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR ATGP Iterative: Error writing image file: %s.", argv[3]);
		fflush(stdout);
		exit(-1);
	}

	// Free memory ATGP

	free(image_vector);
	free(pixel);
	free(umatrix);
	free(w);
	free(u);
	free(c1);
	free(c2);
	free(f);
	free(auxV1);
	free(mult);
	free(interleave);
	free(waveUnit);
	free(wavelength);

	return 0;
}
