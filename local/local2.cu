#include <cuda_runtime.h>

//#define NUMS 64
//#define num_size 8
//#define NUM 49

#define local_1(NUMS) \
__global__ void local_1_##NUMS(float *a) \
{\
	float tmp[NUMS];\
	int i;\
	for(i=0;i<NUMS;i++)\
	{\
		tmp[i]=a[i];\
	}\
	for(i=0;i<NUMS;i++)\
	{\
		a[i]+=tmp[i];\
	}\
}
//local_1(29)	//sm_10
//local_1(30)

//local_1(73)	//sm_35
//local_1(74)

#define local_2(num_size) \
	__global__ void local_2_##num_size(float *a,float *b,float *c)\
{\
	float tmp_a[num_size*num_size];\
	float temp;\
	int i,j,k;\
	for (i=0;i<num_size*num_size;i++)\
	{\
		tmp_a[i]=a[i];\
	}\
	for (i=0;i<num_size;i++)\
	{\
		for (j=0;j<num_size;j++)\
		{\
			temp=0.0;\
			for (k=0;k<num_size;k++)\
			{\
				temp+=tmp_a[i*num_size+k]*b[k*num_size+j];\
			}\
			c[i*num_size+j]=temp;\
		}\
	}\
}
//local_2(2) //sm_10
//local_2(3)
//local_2(4)
//local_2(5)
//local_2(6) //sm_35
//local_2(7)
//local_2(8)
//local_2(9)

#define local_3(NUM) \
__global__ void local_3_##NUM(float *a)\
{\
	float tmp[NUM];\
	float minf=0.0,temp;\
	int mind;\
	int i,j;\
	for(i=0;i<NUM;i++)\
	{\
		tmp[i]=a[i];\
	}\
	for(i=0;i<NUM;i++)\
	{\
		minf=tmp[i];\
		mind=i;\
		for (j=i;j<NUM;j++)\
		{\
			if (minf>tmp[j])\
			{\
				minf=tmp[j];\
				mind=i;\
			}			\
		}\
		if (mind!=i)\
		{\
			temp=tmp[i];\
			tmp[i]=tmp[mind];\
			tmp[mind]=temp;\
		}\
	}\
	a[0]=tmp[NUM-1];\
}
				//sm_10
local_3(2)
local_3(4)
local_3(8)
local_3(16)
local_3(32)
local_3(64)
local_3(128)
