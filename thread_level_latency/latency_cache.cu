#include <stdlib.h>
#include <stdio.h>
#define DATATYPE int
#define ARRAYLEN 1024*1024*256
#define REP 128
//#define PRINTNEED
#define TIMETESTEVENT
#include <cuda_runtime.h>
#include "repeat.h"

__global__ void test_global_latency(double *time,DATATYPE *out,int its,DATATYPE *array)
{
	int p=0;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

//	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(p=array[p];)
		stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0;
	out[0] =p;
	time[0] = time_tmp;
}

texture <int,1,cudaReadModeElementType> texref;
__global__ void test_texture_latency(double *time,DATATYPE *out,int its)
{
	int p=0;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

//	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(p=tex1Dfetch(texref,p);)
		stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0;
	out[1] =p;
	time[1] = time_tmp;
}



void call_test_latency(int step,int its,double *h_time)
{
	DATATYPE *h_array;
	h_array=(DATATYPE*)malloc(sizeof(DATATYPE)*ARRAYLEN);
	for (int i=0;i<ARRAYLEN;i++)
	{
		h_array[i]=(i+step)%ARRAYLEN;
	}
	DATATYPE *d_array;
	cudaMalloc((void**)&d_array,sizeof(DATATYPE)*ARRAYLEN);
//	cudaMemcpy(d_array,h_array,sizeof(DATATYPE)*ARRAYLEN,cudaMemcpyHostToDevice);
	if (cudaSuccess != cudaMemcpy(d_array,h_array,sizeof(DATATYPE)*ARRAYLEN,cudaMemcpyHostToDevice)){ printf("1\n"); return; }

	/*texture*/

	double *d_time;
	cudaMalloc((void**)&d_time,sizeof(double)*6);
	DATATYPE *d_out,*h_out;
	h_out=(DATATYPE *)malloc(sizeof(DATATYPE)*6);
	cudaMalloc((void**)&d_out,sizeof(DATATYPE)*6);

	test_global_latency		<<<1,1>>>(d_time,d_out,its,d_array);
	if (cudaDeviceSynchronize() != cudaSuccess){
		printf("3\n");
		return;
	}


	cudaMemcpy(h_out,d_out,sizeof(DATATYPE)*6,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_time,d_time,sizeof(double)*6,cudaMemcpyDeviceToHost);
	printf("%d:\t%f\t\n",step,h_time[0]);



	cudaUnbindTexture(texref);
	cudaFree(d_array);
	cudaFree(d_time);
	cudaFree(d_out);
	free(h_array);
	free(h_out);
}


int  main()
{
	double *h_time;
	h_time=(double*)malloc(sizeof(double)*6*1024);
	printf("step\t global\t texture\n");
	for (int i=1024;i<=ARRAYLEN;i+=1024)
	{
		call_test_latency(i,1,&h_time[(i-1)*6]);
	}
	call_test_latency(1024,1,h_time);
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


