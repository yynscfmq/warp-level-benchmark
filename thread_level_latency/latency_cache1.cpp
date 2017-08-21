#include <stdlib.h>
#include <stdio.h>
#define DATATYPE int
#define ARRAYLEN 1024*1024*256
#define REP 128
//#define PRINTNEED
#define TIMETESTEVENT
#include <cuda_runtime.h>

void call_test_latency(DATATYPE *h_array,DATATYPE *d_array,int step,int its,double *h_time,double *d_time,DATATYPE *d_out,DATATYPE *h_out);

int  main()
{
	double *h_time;
	h_time=(double*)malloc(sizeof(double)*6*1024);
	printf("step\t global\t texture\n");
	DATATYPE *h_array;
	h_array=(DATATYPE*)malloc(sizeof(DATATYPE)*ARRAYLEN);
	DATATYPE *d_array;
	cudaMalloc((void**)&d_array,sizeof(DATATYPE)*ARRAYLEN);
	double *d_time;
	cudaMalloc((void**)&d_time,sizeof(double)*6);
	DATATYPE *d_out,*h_out;
	h_out=(DATATYPE *)malloc(sizeof(DATATYPE)*6);
	cudaMalloc((void**)&d_out,sizeof(DATATYPE)*6);


	for (int i=89088+1024;i<=ARRAYLEN;i+=1024)
	{
#pragma omp parallel for
		for (int j=0;j<ARRAYLEN;j++)
		{
			h_array[j]=(i+j)%ARRAYLEN;
		}
		call_test_latency(h_array,d_array,i,1,&h_time[(i-1)*6],d_time,d_out,h_out);
	}

	cudaFree(d_array);
	cudaFree(d_time);
	cudaFree(d_out);
	free(h_array);
	free(h_out);

//	call_test_latency(1024,1,h_time);
	//printf("average:\t");
	//for (int i=0;i<2;i++)
	//{
	//	double average=0.0;
	//	for (int j=0;j<1024;j++)
	//	{
	//		average+=h_time[j*6+i];
	//	}
	//	average/=1024.0;
	//	printf("%f\t",average);
	//}
	printf("\n");
	return 0;
}


