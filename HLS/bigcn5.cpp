#include"bigcn4.h"
#include <cmath>
#include<stdlib.h>
#include<iostream>
#include"ap_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>




template<unsigned N_P,unsigned N_B,unsigned N_node,unsigned N_feature >
void Extraction(data_bi in[N_node][N_feature], data_bi weight[N_feature][N_P][N_B],float beta[N_node],float alpha[N_P][N_B],float out[N_node][N_P][N_B])
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=alpha
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
	for(int l=0;l<N_node;l++)
	  {
		for(int m=0;m<N_B;m++)
		{
			for(int i=0;i<N_P;i++)
			{
#pragma HLS UNROLL
			    int bitcount =0;
				for(int k=0;k<N_feature;k++)
				{
#pragma HLS UNROLL
					bitcount += in[l][k]*weight[k][i][m];
				}
				float temp;
#pragma HLS BIND_OP variable=temp op=sub impl=dsp
				temp = alpha[i][m]*beta[l]* (2*bitcount-N_feature);
				out[l][i][m]=temp;
			 }
		  }
	  }
}







template<unsigned N_edge,unsigned N_P,unsigned N_B,unsigned N_node >
void Aggregation(int valu[N_edge],int col[N_edge],int row[N_edge], float in[N_node][N_P][N_B],float out[N_node][N_P][N_B])
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=col
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=row
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=out
#pragma HLS DEPENDENCE variable=out inter false
    for(int i=0;i<N_edge;i++)
	{
    	int temp=row[i];
    	int temp1=col[i];
    	float temp3;
    	for(int j=0;j<N_B;j++)
    	{
#pragma HLS UNROLL
    		for (int k=0;k<N_P;k++)
    		{
#pragma HLS UNROLL
    			out[temp][k][j]+=in[temp1][k][j]*valu[i];
    		}
    	}
	 }
}

template<unsigned N_feature,unsigned N_P,unsigned N_B >
void wei_Buf(data_bi in[N_feature][N_P][2*N_B],data_bi out[N_feature][N_P][N_B],int index)
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
#pragma HLS DEPENDENCE variable=out inter false
	    for(int k=0;k<N_B;k++)
		{
	    	for(int j=0;j<N_feature;j++)
	   			{
				for(int i=0;i<N_P;i++)
				{
#pragma HLS UNROLL
					out[j][i][k]=in[j][i][index*N_B+k];
				}
			}
		}
}




void TopFun(data_bi in[node][feature], data_bi weight[feature][P][2*B],float beta[node],float alpha[P][B],int valu[edge],int col[edge],int row[edge], float out[node][P][B])
{

	float out1[node][P][B];
	data_bi weight1[feature][P][B];
	for(int i=0;i<2;i++)
	{
	   int index=i;
#pragma HLS DATAFLOW
	   wei_Buf<feature,P,B>(weight,weight1,index);
	   Extraction<P,B,node,feature>(in,weight1,beta,alpha,out1);
	   Aggregation<edge,P,B,node>(valu,col,row,out1,out);
	}

}
