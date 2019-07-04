#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#define MAXLINE 200
#define MAXCAD 200
#define FPS 5
#define TILE_WIDTH 32

#define MAXTHREADS 512
#define BLOCKSIZE_MEDIA 1024

extern "C" int dgesvd_(char *jobu, char *jobvt, int *m, int *n,
	double *a, int *lda, double *s, double *u, int *
	ldu, double *vt, int *ldvt, double *work, int *lwork,
	int *info);

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

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
    char line[MAXLINE]="";
    char value [MAXLINE]="";

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
    char line[MAXLINE]="";
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
__global__ void meanImage(double* d_image, double* d_pixel, int N, int iterations){
	__shared__ double sdata[BLOCKSIZE_MEDIA];
	int it,  s;
	unsigned int tid = threadIdx.x;
	int element;
	if(tid==0){
		d_pixel[blockIdx.x]=0;
	}

	for (it=0; it<iterations; it++){
		element=(N*blockIdx.x)+(blockDim.x*it);
		if((it*blockDim.x)+tid<N){
			sdata[tid]=d_image[element+tid];
		}
		else{
			sdata[tid]=0;
		}
		__syncthreads();

		for(s=blockDim.x/2; s>0; s=s/2){
			if (tid < s){
				sdata[tid]+=sdata[tid+s];
			}
			__syncthreads();
		}

		if(tid==0){
			d_pixel[blockIdx.x]+=sdata[0];
		}
		__syncthreads();

	}
	if(tid==0){
		d_pixel[blockIdx.x]/=N;
		sdata[0]=d_pixel[blockIdx.x];

	}
	__syncthreads();
	for (it=0; it<iterations; it++){
		element=(N*blockIdx.x)+(blockDim.x*it);
		if((it*blockDim.x)+tid<N){
			d_image[element+tid]-=sdata[0];
		}
	}


}

/*
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */
__global__ void correlation(double* Cov, double* Corr, int size, double* meanSpect)
{
	int row = blockIdx.x*blockDim.x+threadIdx.x;
	int col = blockIdx.y*blockDim.y+threadIdx.y;

	if(row<size && col<size)
	{
		Corr[(row*size)+col] = Cov[(row*size)+col]+(meanSpect[row] * meanSpect[col]);
	}

}

/*
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */
__global__ void estimation(double* CovEigVal, double* CorrEigVal, int* count, int N, int bands)
{
	int band = blockIdx.x*blockDim.x+threadIdx.x;

	double sigmaSquareTest, sigmaTest, TaoTest;
	double e;

	if(band < bands)
	{
    	sigmaSquareTest = (CovEigVal[band]*CovEigVal[band]+CorrEigVal[band]*CorrEigVal[band])*2/N;
    	sigmaTest = sqrt(sigmaSquareTest);

    	e = 0.906193802436823;
    	TaoTest = sqrt((double)2) * sigmaTest * e;
    	if((CorrEigVal[band]-CovEigVal[band]) > TaoTest)
    		atomicAdd(&count[0], 1);
    	e = 1.644976357133188;
    	TaoTest = sqrt((double)2) * sigmaTest * e;
    	if((CorrEigVal[band]-CovEigVal[band]) > TaoTest)
    		atomicAdd(&count[1], 1);
    	e = 2.185124219133003;
    	TaoTest = sqrt((double)2) * sigmaTest * e;
    	if((CorrEigVal[band]-CovEigVal[band]) > TaoTest)
    		atomicAdd(&count[2], 1);
    	e = 2.629741776210312;
    	TaoTest = sqrt((double)2) * sigmaTest * e;
    	if((CorrEigVal[band]-CovEigVal[band]) > TaoTest)
    		atomicAdd(&count[3], 1);
    	e = 3.015733201402701;
    	TaoTest = sqrt((double)2) * sigmaTest * e;
    	if((CorrEigVal[band]-CovEigVal[band]) > TaoTest)
    		atomicAdd(&count[4], 1);
	}
}

/*
 * Author: Sergio Sanchez Martinez
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char* argv[])
{
	int N;


    double *meanSpect;
    double *Cov;
    double *Corr;
    double *CovEigVal;
    double *CorrEigVal;
    double *U;
    double *VT;

	if(argc != 4)
	{
		printf("\nEXECUTION ERROR VD: Parameters are not correct.");
		printf(" ./VD [Image Filename] [Approximation] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}

	// Load image

	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)malloc(MAXCAD*sizeof(char));
	waveUnit = (char*)malloc(MAXCAD*sizeof(char));

	char cad[MAXLINE];
	strcpy(cad,argv[1]); // Second parameter: Header file:
	strcat(cad,".hdr");
	int error = readHeader1(cad, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("EXECUTION ERROR VD Iterative: Error 1 reading header file: %s.", cad);
		fflush(stdout);
		exit(-1);
	}

	double* wavelength = (double*)malloc(bands*sizeof(double));


	error = readHeader2(cad, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR VD Iterative: Error 2 reading header file: %s.", cad);
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

    N=lines*samples;
	//COVARIANCE
    meanSpect		= (double*) malloc(bands * sizeof(double));
    Cov			= (double*) malloc(bands * bands * sizeof(double));
    Corr			= (double*) malloc(bands * bands * sizeof(double));
    CovEigVal		= (double*) malloc(bands * sizeof(double));
    CorrEigVal	= (double*) malloc(bands * sizeof(double));
    U		= (double*) malloc(bands * bands * sizeof(double));
    VT	= (double*) malloc(bands * bands * sizeof(double));

    CUDA_CHECK_RETURN(cudaDeviceReset());

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	dim3 dimGrid_meanI(bands,1,1);
	dim3 dimBlock_meanI(MAXTHREADS, 1, 1);

	double* imageD;
	double* meanSpectD;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&imageD, sizeof(double)*N*bands));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&meanSpectD, sizeof(double)*bands));

	CUDA_CHECK_RETURN(cudaMemcpy((void*)imageD, image, N*bands*sizeof(double), cudaMemcpyHostToDevice));

	int iterations=ceil((N/MAXTHREADS)+1);

	meanImage<<<dimGrid_meanI,dimBlock_meanI, 0, 0>>>(imageD, meanSpectD, N, iterations);

	CUDA_CHECK_RETURN(cudaMemcpy((void*)meanSpect, meanSpectD, bands*sizeof(double), cudaMemcpyDeviceToHost));

    //******************************************
    double* CovD;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&CovD, sizeof(double)*bands*bands));

    double alpha = (double)1/N, beta = 0;
    cublasDgemm('T', 'N', bands, bands, N, alpha, imageD, N, imageD, N, beta, CovD, bands);
    CUDA_CHECK_RETURN(cudaFree(imageD));
    //******************************************
	//CORRELATION
    //******************************************
	dim3 dimGrid_Corr(ceil(bands/TILE_WIDTH)+1,ceil(bands/TILE_WIDTH)+1,1);
	dim3 dimBlock_Corr(TILE_WIDTH, TILE_WIDTH, 1);

	double* CorrD;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&CorrD, sizeof(double)*bands*bands));

	correlation<<<dimGrid_Corr, dimBlock_Corr, 0 , 0>>>(CovD, CorrD, bands, meanSpectD);

	CUDA_CHECK_RETURN(cudaMemcpy((void*)Corr, CorrD, bands*bands*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)Cov, CovD, bands*bands*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)meanSpect, meanSpectD, bands*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(meanSpectD));
    free(meanSpect);
	//******************************************
	//SVD
    int lwork = max(1,max(3*min(bands, bands)+max(bands,bands),5*min(bands,bands)));
    int info;
    double *work = (double*)malloc(lwork*sizeof(double));
    dgesvd_("N", "N", &bands, &bands, Cov, &bands, CovEigVal, U, &bands, VT, &bands, work, &lwork, &info);
    dgesvd_("N", "N", &bands, &bands, Corr, &bands, CorrEigVal, U, &bands, VT, &bands, work, &lwork, &info);

    free(Cov);
    free(Corr);
    free(U);
    free(VT);
    free(work);

    //ESTIMATION
    //******************************************
    int* count = (int*)malloc(FPS * sizeof(int));
    int *countD;
    double* CovEigValD;
    double* CorrEigValD;

    for(unsigned int i=0; i<FPS; i++) count[i] = 0;

    CUDA_CHECK_RETURN(cudaMalloc((void**)&countD, sizeof(int)*FPS));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&CovEigValD, sizeof(double)*bands));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&CorrEigValD, sizeof(double)*bands));

    CUDA_CHECK_RETURN(cudaMemcpy((void*)countD, count, FPS*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy((void*)CovEigValD, CovEigVal, bands*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy((void*)CorrEigValD, CorrEigVal, bands*sizeof(double), cudaMemcpyHostToDevice));

	dim3 dimGrid_Est(ceil(bands/TILE_WIDTH)+1,1,1);
	dim3 dimBlock_Est(TILE_WIDTH, 1, 1);

    estimation<<<dimGrid_Est, dimBlock_Est, 0, 0>>>(CovEigValD, CorrEigValD, countD, N, bands);

    CUDA_CHECK_RETURN(cudaFree(CorrEigValD));
    CUDA_CHECK_RETURN(cudaFree(CovEigValD));
    free(CovEigVal);
    free(CorrEigVal);

    CUDA_CHECK_RETURN(cudaMemcpy((void*)count, countD, FPS*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(countD));

    int res = count[atoi(argv[2])-1];

	//END CLOCK*****************************************
	end = clock();
	printf("Parallel VD: %f segundos", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************

	error = writeValueResults(argv[3], res);
	if(error != 0)
	{
		printf("EXECUTION ERROR VD Parallel: Error writing results file: %s.\n", argv[3]);
		fflush(stdout);
		exit(-1);
	}
	//FREE MEMORY
    free(count);
	free(image);
	free(interleave);
	free(waveUnit);

	return 0;
}

