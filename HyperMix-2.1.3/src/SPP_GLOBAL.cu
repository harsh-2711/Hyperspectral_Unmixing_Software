#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <cuda.h>


#define dimSize 64
#define MAX_THREADS 512
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
            //samples
            if(strstr(line, "samples")!=NULL && samples !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *samples = atoi(value);
            }

            //lines
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
 * Author: Jaime Delgado Granados
 * Centre: Universidad de Extremadura
 * */
double calculateC(int d){

	double C=0.0; // Constant = sum of the square distance between central pixel and the rest of the pixel in the window
	int i,j;
	double aux;
	for (i = 0; i <= d; i++){  // recorremos el cuadrante
		for ( j = 1; j <= d; j++){
			aux=(double)(j*j + i*i);
			C = C + (1/aux);  // sumamos la distancia al cuadrado es decir i^2 + j^2
		}
	}
	C = C * 4;  // Se muliplica por 4 porque hay 4 cuadrantes simetricos.

	return C;
}

/*
 * Author: Jaime Delgado Granados
 * Centre: Universidad de Extremadura
 * */
void calculateI(double *image, double *I, int  samples, int  lines, int bands){

 	int i,j,z;
 	int LS=samples*lines;
	for (z=0; z< bands; z++){
	  	I[z] = 0.0;
		for(i=0; i<lines; i++){
		      for(j=0; j<samples; j++){
			      I[z] = I[z] + image[i*samples + j + samples*lines*z];
		      }
		}
		I[z]=(I[z]/(LS));
	}
}

/*
 * Author: Jaime Delgado Granados
 * Centre: Universidad de Extremadura
 * */
void calculateB(double *B, int d, double C, int ws){

  	int r,s;

  	for (r = -d ; r <= d; r++){  // recorremos la matriz B
		for (s = -d ; s <= d; s++){
			if( (s == 0) && (r == 0)) {  // si estamos en el pixel central le ponemos 0
				B[d*ws+d]=0;
			}
			else{  // sino calculamos B[r,s]   (B[r+d,s+d] porque c++ no permite indices negativos en las matrices.

				double aux1=(r*r + s*s);
				aux1=1/aux1;
				double aux2=1/C;
				B[(r+d)*(ws)+s+d] = (aux2) * (aux1) ;
			}
		}
	}
}



/*
 * Author: Jaime Delgado Granados
 * Centre: Universidad de Extremadura
 * */
__global__ void getP_dev12(double *image,double * imageA, int samples, int lines, int bands, int ws, double C, int d, double *B){

	int x= threadIdx.x +  blockIdx.x * blockDim.x;
	extern __shared__ double sdata[];
	double EucNormA=0;
	long int vectA=0;
	unsigned int i;

	if ((x >= 0) && (x < (samples*lines))){ // si caen dentro de la imagen
// si caen dentro de la imagen
			for (i=0; i< bands; i++){
				vectA = image[x + (samples)*(lines)*i];//central
				EucNormA = EucNormA + (vectA*vectA);
			}
			imageA[x]  =  EucNormA;
	}
}


/*
 * Author: Jaime Delgado Granados
 * Centre: Universidad de Extremadura
 * */
__global__ void getP_dev(double *image,double * imageP, double * imageA, int samples, int lines, int bands, int ws, double C, int d, double *B, int valor, int cordX){

	int globalX = blockDim.x*blockIdx.x +threadIdx.x +(threadIdx.z-d) + (threadIdx.y -d)*samples;
	int central = blockDim.x*blockIdx.x + threadIdx.x;
	extern __shared__ double sdata[];
	double G=0.0;
	double vectA=0.0;
	double vectB=0.0;
	double dotPro=0.0;
	unsigned int i;
	double cose;

	if(central >=0 && central <samples*lines  ){

		if( globalX >= 0 && globalX< (samples*lines)){
			for (i=0; i< (bands); i++){
				vectA=image[globalX + (samples)*(lines)*i];
				vectB=image[central + (samples)*(lines)*i];
				dotPro=dotPro + vectA*vectB;
			}

			cose = (dotPro/(imageA[globalX] * imageA[central]));
			if (cose >= 1)
			  G=0.0;
			else{
			  G = acos(cose );   // calculamos el angulo como el arcoseno
			  if (G<0)
			    G=G*(-1);
			}

			sdata[threadIdx.x*ws*ws + threadIdx.y*ws + threadIdx.z]=B[threadIdx.y*ws + threadIdx.z]*G;
			__syncthreads();

			for (i=valor/2; i>0; i= i /2) {
				if(((threadIdx.y*ws + threadIdx.z) < i) && ((threadIdx.y*ws + threadIdx.z + i) < (ws*ws))) {
					sdata[threadIdx.x*ws*ws + threadIdx.y*ws + threadIdx.z] += sdata[threadIdx.x*ws*ws + threadIdx.y*ws + threadIdx.z + i];
				}
				__syncthreads();
			}

			if ((threadIdx.z==0) && (threadIdx.y==0)){
				imageP[central]  =  sdata[threadIdx.x*ws*ws];
			}
		}
	}

}


/*
 * Author: Jaime Delgado Granados
 * Centre: Universidad de Extremadura
 * */
__global__ void getPreprocessing_dev(double *image, double *imageP, double *imageOut, int samples, int lines, int bands, double *I){

	int x= threadIdx.x +  blockIdx.x * blockDim.x;
	int n= 0;
	extern __shared__ double fdata[];


	if ( threadIdx.x < bands){ // si caen dentro de la imagen
		fdata[threadIdx.x]=I[threadIdx.x];
		__syncthreads();
	}
	if ( x < (samples)*(lines)){ // si caen dentro de la imagen
		while(n<(bands)){
			imageOut[(x + (samples)*(lines)*n)] = ((1/imageP[x]) * ( image[x + ((samples)*(lines)*n)] - fdata[n]) + fdata[n]);
			n++;
		}
	}
}

/*
 * Author: Jaime Delgado Granados
 * Centre: Universidad de Extremadura
 * */
__global__ void getI(double *image, double *dev_I, int samples, int lines){

	int i;
	double acum = 0.00;
	unsigned int iteraciones = ((lines*samples-1)/blockDim.x) +1;
	extern __shared__ double sdata[];

	for (i = 0; i < iteraciones; i++){
		if (blockDim.x*i + threadIdx.x < (samples*lines) )
			acum = acum + image[blockIdx.x*(samples*lines) +  blockDim.x*i + threadIdx.x ];
	}
	sdata[threadIdx.x] = acum;
	__syncthreads();

	for (i = blockDim.x/2; i>0; i = i /2 ){
		if (threadIdx.x < i)
		sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + i];
		__syncthreads();
	}

	if(threadIdx.x == 0){
		dev_I[blockIdx.x] = sdata[0] / (samples*lines);
	}

}

/*
 * Author: Jaime Delgado Granados
 * Centre: Universidad de Extremadura
 * */
int main (int argc, char* argv[]){


	/*
	 * ARGUMENTS
	 *
	 * argv[1]: Input image filename
	 * argv[2]: Window size
	 * argv[3]: Output image filename
	 * */


	if(argc > 4 || argc < 4)
	{
		printf("EXECUTION ERROR SPP Parallel: Parameters are not correct.");
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
		printf("EXECUTION ERROR SPP Parallel: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR SPP Parallel: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	image = (double*)malloc ( samples*lines * bands*sizeof(double) );
	imageOut = (double*)malloc ( samples*lines * bands*sizeof(double) );
	error = loadImage(argv[1], image, samples, lines, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SPP Parallel: Error image header file: %s.", argv[1]);
		fflush(stdout);
		return error;
	}

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************

	//2º STEP: PREPROCESSING
	double *I = (double*)malloc(bands*sizeof(double));
	int d = (ws-1) / 2;
	double *B = (double*)malloc(ws*ws*sizeof(double));
	double *imageA = (double*)malloc ( samples*lines * bands*sizeof(double) );
	double *imageP = (double*)malloc ( samples*lines * bands*sizeof(double) );
	float C=0.0;

	//2º STEP: PREPROCESSING
	    C=calculateC(d);
	  //  calculateI(image, I, samples, lines, bands);
	    calculateB(B, d, C, ws);


	    dim3 Blocks((dimSize),(dimSize),1);
	    dim3 Threads((ws),(ws),1);
	// GETP KERNELL
	    double *dev_image;// device copies of image, samples, lines, ws, C,
	    double *dev_B;
	    double *dev_imageP;
	    double *dev_imageA;
		double *dev_I;

	// GETP KERNELL: allocate device copies
	    cudaMalloc( (void**)&dev_image, (samples)*(lines)*(bands)*sizeof(double));
	    cudaMalloc( (void**)&dev_B, (ws)*(ws)*sizeof(double));
	    cudaMalloc( (void**)&dev_imageP, (samples)*(lines)*sizeof(double) );
	    cudaMalloc( (void**)&dev_imageA, (samples)*(lines)*sizeof(double) );
		cudaMalloc( (void**)&dev_I, (bands)*sizeof(double) );

	// GETP KERNELL: copy inputs to device
	    cudaMemcpy( dev_image, image, (samples)*(lines)*(bands)*sizeof(double), cudaMemcpyHostToDevice );
	    cudaMemcpy( dev_B, B, (ws)*(ws)*sizeof(double), cudaMemcpyHostToDevice );

		  // CalculateI kernel

		  dim3 BlocksI(bands);
	      dim3 ThreadsI(512);

		  getI<<<BlocksI,ThreadsI,512*sizeof(double)>>>(dev_image, dev_I, samples, lines);



	// GETP KERNELL
	//    int x,y;
	/*    int valor=1;
	    while(valor<(ws*ws)){
	    	valor=valor*2;
	    }
	*/

		  int valor = exp2f(ceilf(log2f(ws*ws)));

	    int blocks= (samples*lines)/512 +1;
	    dim3 Blocks12(blocks);
	    dim3 Threads12(512);


	    getP_dev12<<<Blocks12,512>>>(dev_image,dev_imageA,samples,lines,bands,ws,C,d,dev_B);

	    cudaMemcpy( (void*)imageA, dev_imageA, (samples)*(lines)*sizeof( double ) , cudaMemcpyDeviceToHost );

	    int s,o;
	    for(s=0;s<lines;s++){
	       	for(o=0;o<samples;o++){
	       		imageA[s*samples + o]=sqrtf(imageA[s*samples + o]);
	      	}
	    }


	    cudaMemcpy( dev_imageA, imageA, (samples)*(lines)*sizeof(double), cudaMemcpyHostToDevice );

		///// VERSION VARIOS PIXELES UN BLOQUE
		/*
	    int x,y;

	   for(y=0;y<((lines-1)/dimSize+1 );y++){
	      for(x=0;x<((samples-1)/dimSize+1);x++){
	   		  getP_dev2<<<Blocks,Threads, valor*sizeof(double)>>>(dev_image,dev_imageP, dev_imageA,samples,lines,bands,ws,C,d,dev_B,x,y,valor);
	   	  }
	   }
	   cudaMemcpy( (void*)imageP, dev_imageP, (samples)*(lines)*sizeof( double ) , cudaMemcpyDeviceToHost );
	   */

	     int i,j;
	  	  // GETP KERNELL VERSION VARIOS PIXELES BLOQUE
		  int cordX=1;
		  int cordY=ws;
		  int cordZ=ws;
		  cordX=MAX_THREADS/(cordY*cordZ);
		  int nBlocks=((samples*lines)-1)/cordX+1;
		  dim3 Blocks1(nBlocks,1,1);
		  dim3 Threads1(cordX, cordY, cordZ);


		   cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//	   printf("cordX %d cordY %d cordZ %d, nBlocks %d",cordX,cordY,cordZ,nBlocks);


	//PASAR samples Y lines COMO samples*lines Y AHORRO UN PARAMETRO
		  getP_dev<<<Blocks1,Threads1, MAX_THREADS*sizeof(double)>>>(dev_image,dev_imageP, dev_imageA,samples,lines,bands,ws,C,d,dev_B, valor, cordX);
	 	  cudaMemcpy( imageP, dev_imageP, (samples)*(lines)*sizeof( double ) , cudaMemcpyDeviceToHost );

		  	   cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);


	   for(s=0;s<lines;s++){
		   for(o=0;o<samples;o++){
			   double hola=1 + sqrtf(imageP[s*samples + o]);
			   imageP[s*samples + o]=hola*hola;
		   }
	   }
	   cudaMemcpy( dev_imageP, imageP, (samples)*(lines)*sizeof(double), cudaMemcpyHostToDevice );


	   dim3 Blocks2((ceil((samples)*(lines))/bands) + 1);
	   dim3 Threads2(bands);

	// GETPREPROCESSING KERNELL
		double *dev_imageOut;


	// GETPREPROCESSING KERNELL: allocate device copies
		cudaMalloc( (void**)&dev_imageOut, (samples)*(lines)*(bands)*sizeof(double) );

	// operate over the image dev_image,dev_imageP,dev_s,dev_l,dev_bands,dev_ws,dev_c,dev_d,dev_B
		getPreprocessing_dev<<<Blocks2,Threads2,(bands)*sizeof(double)>>>(dev_image, dev_imageP, dev_imageOut, samples, lines, bands, dev_I);
	// GETP KERNELL: copy device result back to host copy of c
		cudaMemcpy( imageOut, dev_imageOut, (samples)*(lines)*(bands)*sizeof( double ) , cudaMemcpyDeviceToHost );


	//END CLOCK*****************************************
	end = clock();
	printf("Parallel SPP: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************

	char headerOut[MAXLINE];
	strcpy(headerOut, argv[3]);
	strcat(headerOut, ".hdr");

	error = writeHeader(headerOut, lines, samples, bands, dataType, interleave, byteOrder, waveUnit, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR SPP Parallel: Error writing header file: %s.", headerOut);
		fflush(stdout);
		return error;
	}
	error = writeResult(imageOut, argv[3], lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SPP Parallel: Error writing image file: %s.", argv[3]);
		fflush(stdout);
		return error;
	}

	// GETP KERNELL:
	cudaFree( dev_image );
	cudaFree( dev_imageP );
	cudaFree( dev_I);
	cudaFree( dev_imageOut);
	cudaFree( dev_B);
	cudaFree( dev_imageA);

	free(B);
	free(imageP);
	free(I);
	free(image);
	free(imageOut);
	free(imageA);
}
