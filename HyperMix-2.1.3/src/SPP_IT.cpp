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
 * Author: Jaime Delgado
 * Centre: Universidad de Extremadura
 * */
float getP(double *image, int samples, int lines, int ws, int x, int y, int bands, double C, int d, float *B)
{

	int i;
	long double sum = 0.0;
	float G = 0.0;
	int r,s, auxr, auxs;

	double cos= 0.0;

	double dotPro;
	double EucNormA;
	double EucNormB;

	// CALCULATE A
	// For every window coordenate
	for (r = x-d; r <= x+d ; r++)
	{
		for(s = y-d; s <= y+d; s++)
		{
			auxs = s;
			auxr = r;
			if(auxs < 0) auxs = auxs + d + 1;
			if(auxr < 0) auxr = auxr + d + 1;
			if(auxs >= lines) auxs = auxs - d - 1;
			if(auxr >= samples) auxr = auxr - d - 1;

			//	CALCULATE G
			dotPro = 0.0;
			EucNormA = 0.0;
			EucNormB = 0.0;
			for(i=0;i<bands;i++)
			{
				dotPro = dotPro + image[y*samples + x + samples*lines*i]*image[auxs*samples + auxr + samples*lines*i];
				EucNormA = EucNormA + image[y*samples + x + samples*lines*i]*image[y*samples + x + samples*lines*i];
				EucNormB = EucNormB + image[auxs*samples + auxr + samples*lines*i]*image[auxs*samples + auxr + samples*lines*i];
			}
			EucNormA= sqrt(EucNormA);
			EucNormB= sqrt(EucNormB);

			cos = dotPro /  (EucNormA * EucNormB);

			if (cos >= 1)
			  G=0.0;
			else
			{
			  G = acos(cos);
			  if (G<0)
				G=G*(-1);
			}
			sum = sum + B[(auxr-x+d)*ws + auxs-y+d]*G;
		}
	}
    //******************************************************
	float res= 1 + sqrt(sum);
	return res*res;
}

/*
 * Author: Jaime Delgado
 * Centre: Universidad de Extremadura
 * */
int main (int argc, char* argv[])
{

	/*
	 * ARGUMENTS
	 *
	 * argv[1]: Input image filename
	 * argv[2]: Window size
	 * argv[3]: Output image filename
	 * */


	if(argc > 4 || argc < 4)
	{
		printf("EXECUTION ERROR SPP Iterative: Parameters are not correct.");
		printf("./SPP [Image Filename] [Window size] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}

	int ws = atoi(argv[2]); // window size
	if((ws%2)==0)
	  ws=ws-1;

	char header_filename[MAXLINE];
	double* image;
	double* imageOut;
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
		printf("EXECUTION ERROR SPP Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR SPP Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	image = (double*)malloc ( samples*lines * bands*sizeof(double) );
	imageOut = (double*)malloc ( samples*lines * bands*sizeof(double) );
	error = loadImage(argv[1], image, samples, lines, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SPP Iterative: Error image header file: %s.", argv[1]);
		fflush(stdout);
		return error;
	}

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	// SPATIAL PREPROCESSING
	float P = 0;
	double *I = (double*)malloc(bands*sizeof(double));
	int d = (ws-1) / 2;
	float *B = (float*)malloc(ws*ws*sizeof(float));


	//CALCULATE C  = sum of the square distance between central pixel and the rest of the pixel
	//in the window
	float C=0.0;
	int i,j,z,n;
	float aux;
	for (i = 0; i <= d; i++)
	{
		for ( j = 1; j <= d; j++)
		{
			aux=(float)(j*j + i*i);
			C = C + (1/aux);
		}
	}
	C = C * 4;  // Is multiplied by 4 because there are 4 simetric regions


	//CALCULATE B
	int r,s;
	float aux1, aux2;
  	for (r = -d ; r <= d; r++)
  	{
		for (s = -d ; s <= d; s++)
		{
			if( (s == 0) && (r == 0))
				B[d*ws+d]=0;
			else
			{
				aux1=(r*r + s*s);
				aux1=1/aux1;
				aux2=1/C;
				B[(r+d)*(ws)+s+d] = (aux2) * (aux1) ;
			}
		}
	}

	for (z=0; z< bands; z++)
	{
	  	I[z] = 0.0;
		for(i=0; i<lines; i++)
		      for(j=0; j<samples; j++)
			      I[z] = I[z] + image[i*samples + j + samples*lines*z];

		I[z] = I[z]/(lines*samples);

	}


	for ( i=0; i<lines;i++)
	{
		for(j=0; j<samples;j++)
		{
		    P = getP(image, samples, lines, ws, j, i, bands, C, d,B);
		    for (n = 0; n < bands; n++)
		    	imageOut[i*samples + j + samples*lines*n] = (1/P) * ( image[i*samples + j + samples*lines*n] - I[n]) + I[n];

		}
	}

	//END CLOCK*****************************************
	end = clock();
	printf("Iterative SPP: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************

	char headerOut[MAXLINE];
	strcpy(headerOut, argv[3]);
	strcat(headerOut, ".hdr");

	error = writeHeader(headerOut, lines, samples, bands, dataType, interleave, byteOrder, waveUnit, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR SPP Iterative: Error writing header file: %s.", headerOut);
		fflush(stdout);
		return error;
	}
	error = writeResult(imageOut, argv[3], lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SPP Iterative: Error writing image file: %s.", argv[3]);
		fflush(stdout);
		return error;
	}

	free(I);
	free(B);
	free(image);
	free(imageOut);
	free(wavelength);
	free(interleave);
	free(waveUnit);

}
