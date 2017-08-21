#include <stdlib.h>
#include <stdio.h>
#define DATATYPE int
#define ARRAYLEN 1024*1024*256
#define REP 128
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



void call_test_latency(DATATYPE *h_array,DATATYPE *d_array,int step,int its,double *h_time,double *d_time,DATATYPE *d_out,DATATYPE *h_out)
{
	printf("111 111\n");

	if (cudaSuccess != cudaMemcpy(d_array,h_array,sizeof(DATATYPE)*ARRAYLEN,cudaMemcpyHostToDevice)){ printf("1\n"); return; }

	printf("111 222\n");


	test_global_latency		<<<1,1>>>(d_time,d_out,its,d_array);
	if (cudaDeviceSynchronize() != cudaSuccess){
		printf("3\n");
		return;
	}
	printf("111 333\n");


	cudaMemcpy(h_time,d_time,sizeof(double),cudaMemcpyDeviceToHost);
	printf("%d:\t%f\t\n",step,h_time[0]);
//	printf("111 444\n");

}

