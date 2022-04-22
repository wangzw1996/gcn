#include"cora2.h"


#include <stdint.h>
#include <cmath>
#include<stdlib.h>
#include<iostream>
#include"ap_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>




template<unsigned N_node,unsigned N_feature >
void read_bi(hls::stream <ap_uint<N_feature>> &in,data_t out1[N_node][N_feature],data_t out2[N_node][N_feature])
{

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out1
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out2

	ap_uint<1433> in_local[N_node];
	for (int j = 0; j < N_node; j ++)
	{
		in_local[j]=in.read();
		for (int i = 0; i < N_feature; i++)
		{
#pragma HLS UNROLL
              if ((i+1)*32<1433)
              {
            	  out1[j][i] =in_local[j].range( (32* (i + 1) -1), 32 * i) ;
            	  out2[j][i] =in_local[j].range( (32* (i + 1) -1), 32 * i) ;


              }
              else
              {
            	  out1[j][i]=in_local[j].range( (32* (i + 1) -8), 32 * i)	;
            	  out2[j][i]=in_local[j].range( (32* (i + 1) -8), 32 * i)	;
              }

		 }
     }
}




template<unsigned N_edge,unsigned N_feature,unsigned N_node,unsigned N_node1,unsigned N_node2,unsigned N_node3,unsigned N_node4 >
void Aggregation( int col[4][N_edge/4],int row[4][N_edge/4],data_t in[N_node][N_feature],data_t in1[N_node][N_feature],float out1[N_node1][N_feature],float out2[N_node2][N_feature],float out3[N_node3][N_feature],float out4[N_node4][N_feature])
{

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in1


#pragma HLS DEPENDENCE variable=out1 inter false
#pragma HLS DEPENDENCE variable=out2 inter false
#pragma HLS DEPENDENCE variable=out3 inter false
#pragma HLS DEPENDENCE variable=out4 inter false
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=row
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=col

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out2
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out1
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out3
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out4

	int temp1_row1=row[0][(N_edge/4)-1];
	int temp2_row1=row[1][(N_edge/4)-1];
	int temp3_row1=row[2][(N_edge/4)-1];

    for (int i=0;i<N_edge/4;i++)
	{

    	 int temp1_row=row[0][i];
    	 int temp1_col=col[0][i];

    	 int temp2_row=row[1][i];
    	 int temp2_col=col[1][i];

    	 int temp3_row=row[2][i];
    	 int temp3_col=col[2][i];

    	 int temp4_row=row[3][i];
    	 int temp4_col=col[3][i];


    		for (int k=0;k<N_feature;k++)
    		{
#pragma HLS UNROLL

    			out1[temp1_row][k]+=in[temp1_col][k];
    		    out2[temp2_row-temp1_row1][k]+=in[temp2_col][k];
    		    out3[temp3_row- temp2_row1][k]+=in1[temp3_col][k];
    		    out4[temp4_row- temp3_row1][k]+=in1[temp4_col][k];

    		}
    }
    for (int l=0;l<N_feature;l++)
    {

      	out1[temp1_row1][l]=out1[temp1_row1][l]+out2[0][l];
      	out2[temp2_row1-temp1_row1][l]=out2[temp2_row1-temp1_row1][l]+out3[0][l];
       	out3[temp3_row1-temp2_row1][l]=out3[temp3_row1-temp2_row1][l]+out4[0][l];
    }

}





void m(float in1[node1][feature],float in2[node2][feature],float in3[node3][feature],float in4[node4][feature],float out[node][feature])
{
	for(int i=0;i<N_node1;i++)
	{
		for(int j=0;j<N_feature;j++)
		{
			out[i][j]=in1[i][j];
		}

	}
	for(int i=0;i<N_node2;i++)
		{
			for(int j=0;j<N_feature;j++)
			{
				out[i+N_node1][j]=in1[i][j];
			}

		}
	for(int i=0;i<N_node3;i++)
		{
			for(int j=0;j<N_feature;j++)
			{
				out[i+N_node1+N_node2][j]=in1[i][j];
			}

		}
	for(int i=0;i<N_node4;i++)
		{
			for(int j=0;j<N_feature;j++)
			{
				out[i+N_node1+N_node2+N_node4][j]=in1[i][j];
			}

		}
}



template<unsigned N_block,unsigned N_node>
void bi(float in[N_node][N_block],data_bi out[N_node][N_block],float beta[N_node])
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=out
	for(int i=0;i<<N_node;i++)
	{
		float sum1=0;
#pragma HLS DEPENDENCE variable=sum1 inter false
		float mean1=0;
			for(int k=0; k<N_block;k++)
			{
#pragma HLS UNROLL
				sum1 +=in[k][i];
			}


		mean1=sum1/N_node;
		float sum2=0;
		float mean2=0;

			for(int k=0;k<N_block;k++)
			{
#pragma HLS UNROLL
				in[k][i]=in[k][i]-mean1;
				sum2+=in[k][i];
			}


		mean2=sum2/N_node;
		float temp=0;
		float std=0;

			for(int k=0;k<N_block;k++)
			{
#pragma HLS UNROLL
				temp+=(in[k][i]-mean2)*(in[k][i]-mean2);
			}


		std=sqrt(temp/N_node);
		float sum3=0;

			for(int k=0;k<N_block;k++)
		    {
#pragma HLS UNROLL
				in[k][i]=in[k][i]/std;
				if (in[k][i]>0)
					out[k][i]=1;
				else
					out[k][i]=0;
				sum3+=abs(in[j][k][i]);
		    }

		beta[i]=sum3/N_node;
	}
}



template<unsigned N_P,unsigned N_node,unsigned N_feature >
void Extraction(data_t in[N_node][N_feature], data_t weight[N_feature][N_P],float beta[N_node],float alpha[N_P],float out[N_node][N_P])
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out1


#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=alpha
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in

	for(int l=0;l<N_node;l++)
	  {

#pragma HLS PIPELINE
			for(int i=0;i<N_P;i++)
			{
#pragma HLS UNROLL
				ap_uint<32> temp=0;
				data_bi temp2;
			    int bitcount1 =0;
				for(int k=0;k<N_feature;k++)
				{
#pragma HLS UNROLL
					temp = ~(in[l][k]^weight[k][i]);
			      for(int p=0;p<32;p++)
			   	  {

#pragma HLS UNROLL
			    	if (k*32+p<1433)
			    	{
			   		   temp2=temp.range( (1* (p + 1) -1), p * 1);
			   		   bitcount1+=temp2;
			    	}
			    	else
			    	{
			    		bitcount1+=0;
			    	}
			   	  }
			   }
			   float temp1;
			   temp1 = alpha[i]*beta[l]* (2*bitcount1-1433);
			   out[l][i]=temp1;

			 }
		  }
}







template<unsigned N_P,unsigned N_B,unsigned N_node1 >
void wr(float in1[N_node1][N_P*N_B], hls::stream <ap_uint<N_P*N_B*32>> &out)
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in1

	ap_uint<N_P*32> result_local1;
	for(int l=0;l<N_node1;l++)
	{
		float res;
		ap_uint<N_P> tmpOut1 = 0;
		for (int i = 0; i < N_P; i++)
		{
#pragma HLS UNROLL
			res = in1[l][i];
			tmpOut1.range(32 * (i + 1) - 1, i * 32) = res;
            }
		result_local1 = tmpOut1;
        out.write( result_local1);
	}
}


void TopFun( hls::stream <ap_uint<feature>> &in, data_t weight[feature][P],float beta[node],float alpha[P],int valu[4][edge/4],int col[4][edge/4],int row[4][edge/4], hls::stream <ap_uint<P*B*32>> &out)
{
	data_t temp_out[node][feature];
#pragma HLS BIND_STORAGE variable=temp_out type=ram_2p impl=uram



	data_t in_buf1[node][feature]={0};
	data_t in_buf2[node][feature]={0};


	float out5[node][feature];
	read_bi<node,feature>(in,in_buf1,in_buf2);
    for(int k=0;k<6;k++)
    {
    	float temp_out1[node][feature];
        float beta2[node];
    	float out1[node1][feature];
#pragma HLS BIND_STORAGE variable=out1 type=ram_2p impl=bram
    	float out2[node2][feature];
#pragma HLS BIND_STORAGE variable=out2 type=ram_2p impl=bram
    	float out3[node3][feature];
    	#pragma HLS BIND_STORAGE variable=out3 type=ram_2p impl=uram
    	float out4[node4][feature];
    	#pragma HLS BIND_STORAGE variable=out4 type=ram_2p impl=uram
#pragma HLS DATAFLOW

    	Aggregation<edge,feature,node,node1,node2,node3,node4>(col,row,in_buf1,in_buf2,out1,out2,out3,out4);
    	m(out1,out2,out3,out4,out5);
    	bi<feature,node>(out5,temp_out,beta2);
    	Extraction<P,node,feature>(temp_out,weight,beta2,alpha,temp_out1);

        wr<P,B,node1>(temp_out1,out);
    }
}
