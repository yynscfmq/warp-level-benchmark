/************************************************************************/
/* writen by fang minquan (fmq@hpc6.com)                                */
/************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#define DATATYPE int
#define ARRAYLEN 2048
#define REP 128
//#define PRINTNEED
#define TIMETESTEVENT
#include <cuda_runtime.h>
#include "repeat.h"
__global__ void test_registerarray_latency(double *time,DATATYPE *out,int its,DATATYPE *array)
{
	DATATYPE register_array[4];
	int i;
	for (i=0;i<4;i++)
	{
		register_array[i]=(i+1)%4;
	}
	int p=0;
	double time_tmp=0.0;
	unsigned int start_time=0, stop_time=0;

	for (int i=0;i<its;i++)										
	{									
		__syncthreads();
		start_time = clock();	
		repeat128(p=register_array[p];)
		stop_time = clock();
		time_tmp+=(stop_time-start_time);
	}
	time_tmp=time_tmp/128.0/its;
	out[0] =p;
	time[0] = time_tmp;
}