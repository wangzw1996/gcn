#include"flickr2.h"
#include <stdint.h>
#include <cmath>
#include<stdlib.h>
#include<iostream>
#include"ap_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>







template<unsigned N_node,unsigned N_B,unsigned N_B2,unsigned N_feature >
void read_bi(hls::stream <ap_uint<N_feature>> &in,data_bi out1[N_node/N_B2/N_B][N_B][N_B2][N_feature])
{

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out1
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out2

	ap_uint<500> in_local[N_node/N_B2/N_B][N_B][N_B2];
	for (int j = 0; j < N_node/N_B2/N_B; j ++)
	{
		for (int m=0;m<N_B;m++)
		{
			for(int k=0;k<N_B2;k++)
			{
				in_local[j][m][k]=in.read();
				for (int i = 0; i < N_feature; i++)
				{
#pragma HLS UNROLL
					if ((i+1)*32<500)
					{
						out1[j][m][k][i] =in_local[j][m][k].range( (32* (i + 1) -1), 32 * i) ;
					}
					else
					{
						out1[j][m][k][i]=in_local[j][m][k].range( (32* (i + 1) -8), 32 * i)	;
					}
				}
			}
		}
	}

}




template<unsigned N_B,unsigned N_node,unsigned N_feature,unsigned N_B2,unsigned N_block >
void Extraction(data_bi in[N_node/N_B2/N_B][N_B][N_B2][N_feature], data_bi weight[N_feature][N_block],float beta[N_node/N_B2/N_B][N_B][N_B2],float alpha[N_block],float out[N_node/N_B2/N_B][N_B][N_B2][N_block])
{

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=alpha
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=in

	for(int l=0;l<N_node/N_B2/N_B;l++)
	  {
		for (int m=0;m<N_B2;m++)
		{
#pragma HLS PIPELINE
			for (int j=0;j<N_B;j++)
			{
#pragma HLS UNROLL
				for(int i=0;i<N_block;i++)
				{
#pragma HLS UNROLL
					ap_uint<32> temp=0;
					data_bi temp2;
					int bitcount1 =0;
					for(int k=0;k<N_feature;k++)
					{
#pragma HLS UNROLL
						temp = ~(in[l][j][m][k]^weight[k][i]);
						for(int p=0;p<32;p++)
						{
#pragma HLS UNROLL
							if (k*32+p < 500)
							{
								temp2=temp.range( (1* (p + 1) -1), p * 1);
								bitcount1+=temp2;
							}
							else
							{
								bitcount1 += 0;
							}
						}
					}
					float temp1=0;
					temp1 = alpha[i]*beta[l][j][m]* (2*bitcount1-500);
					out[l][j][m][i]=temp1;
				}

			 }
		  }
	  }
}




template<unsigned N_B,unsigned N_node,unsigned N_B2,unsigned N_block >
void bi(float in[N_node/N_B2/N_B][N_B][N_B2][N_block],data_bi out[N_node/N_B2/N_B][N_B][N_B2][N_block],float beta[2][2][2][N_block])
{
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=out

	float mean=0;
    float mean_1[N_node/N_B2/N_B][N_B][N_block]={0};
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=mean_1

    for(int j=0; j<N_node/N_B2/N_B;j++)
	{
     for(int m=0;m<N_B;m++)
     {
#pragma HLS UNROLL
		for(int i=0;i<N_B2;i++)
		{
#pragma HLS UNROLL
			for(int k=0;k<N_block;k++)
			{
#pragma HLS UNROLL
				mean_1[j][m][k] += in[j][m][i][k];
			}

		}
	 }
    }

	float mean_3[N_block]={0};
	for(int j=0; j<N_node/N_B2/N_B;j++)
	{
		for (int m=0;m<N_B;m++)
		{
#pragma HLS UNROLL
			for(int k=0;k<N_block;k++)
			{
#pragma HLS UNROLL
				mean_3[k]+=mean_1[j][m][k]/N_node;
			}

		}
	}


	for(int j=0; j<N_node/N_B2/N_B;j++)
	{
		for(int m=0;m<N_B;m++)
		{
#pragma HLS UNROLL
			for(int i=0;i<N_B2;i++)
			{
#pragma HLS UNROLL
				for(int k=0;k<N_block;k++)
				{
#pragma HLS UNROLL
					in[j][m][i][k]=in[j][m][i][k]-mean_3[k];
					if (in[j][m][i][k]>0)
					    out[j][m][i][k]=1;
					else
						out[j][m][i][k]=0;
				}
			}
		}
	}

	float temp1[N_node/N_B2/N_B][N_B][N_block]={0};
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=temp1
	float sum_1[N_node/N_B2/N_B][N_B][N_block]={0};
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable= sum_1

    float sum[N_block]={0};
    float sum3[N_block]={0};
	for(int j=0; j<N_node/N_B2/N_B;j++)
	{
		for(int m=0;m<N_B;m++)
		{
#pragma HLS UNROLL
			for(int i=0;i<N_B2;i++)
			{
#pragma HLS UNROLL
				for(int k=0;k<N_block;k++)
				{
					temp1[j][m][k]+=in[j][m][i][k]*in[j][m][i][k];
					sum_1[j][m][k]+=abs(in[j][m][i][k]);
				}
			}
		 }
	}
	for(int j=0; j<N_node/N_B2/N_B;j++)
	{
	  for(int m=0;m<N_B;m++)
	  {
#pragma HLS UNROLL
		 for(int k=0;k<N_block;k++)
		 {
#pragma HLS UNROLL
			 sum[k]+=temp1[j][m][k];
			 sum3[k]+=abs(sum_1[j][m][k]);
		 }
	   }
	}
	float std[N_block]={0};
	for(int k=0;k<N_block;k++)
	{
			std[k]=sqrt(sum[k]/N_node);
			beta[0][0][0][k]=sum3[k]/std[k]/N_node;
	}
}


template<unsigned N_B,unsigned N_node,unsigned N_B2,unsigned N_block >
void cc(data_bi in[N_node/N_B2/N_B][N_B][N_B2][N_block],data_bi out[N_node][N_block],data_bi out2[N_node][N_block],data_bi out3[N_node][N_block],data_bi out4[N_node][N_block],data_bi out5[N_node][N_block])
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out2
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out3
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out4
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out5
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=in

	int c=0;
	for(int j=0; j<N_node/N_B2/N_B;j++)
	{
		for(int m=0;m<N_B;m++)
		{
			for(int i=0;i<N_B2;i++)
			{
				for(int k=0;k<N_block;k++)
				{
#pragma HLS UNROLL
					if (c<N_node)
					{
					out[c][k]=in[j][m][i][k];
					out2[c][k]=in[j][m][i][k];
					out3[c][k]=in[j][m][i][k];
					out4[c][k]=in[j][m][i][k];
					out5[c][k]=in[j][m][i][k];
					c++;
					}
				}
			}
		}
	}
}






template<unsigned N_edge,unsigned N_B,unsigned N_node,unsigned N_node1,unsigned N_node2,unsigned N_node3,unsigned N_node4,unsigned N_node5,unsigned N_node6,unsigned N_node7,unsigned N_node8,unsigned N_node9,unsigned N_node10>
void Aggregation( hls::stream <ap_uint<10*32>> &col1,hls::stream <ap_uint<10*32>> &row1,data_bi in[N_node][N_B],data_bi in4[N_node][N_B],data_bi in1[N_node][N_B],data_bi in2[N_node][N_B],data_bi in3[N_node][N_B],float out1[N_node1][N_B],float out2[N_node2][N_B],float out3[N_node3][N_B],float out4[N_node4][N_B],float out5[N_node5][N_B],float out6[N_node6][N_B],float out7[N_node7][N_B],float out8[N_node8][N_B],float out9[N_node9][N_B],float out10[N_node10][N_B])

{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in1

#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in2
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in3
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in4

#pragma HLS DEPENDENCE variable=out1 inter false
#pragma HLS DEPENDENCE variable=out2 inter false
#pragma HLS DEPENDENCE variable=out3 inter false
#pragma HLS DEPENDENCE variable=out4 inter false
#pragma HLS DEPENDENCE variable=out5 inter false
#pragma HLS DEPENDENCE variable=out6 inter false
#pragma HLS DEPENDENCE variable=out7 inter false
#pragma HLS DEPENDENCE variable=out8 inter false
#pragma HLS DEPENDENCE variable=out9 inter false
#pragma HLS DEPENDENCE variable=out10 inter false


#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out2
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out1
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out3
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out4
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out5
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out6
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out7
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out8
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out9
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out10

	    int temp1_row1=1681;
		int temp2_row1=5050;
		int temp3_row1=10427;
		int temp4_row1=17573;
	    int temp5_row1=26205;
	    int temp6_row1=36145;
	    int temp7_row1=47352;
	    int temp8_row1=59878;
	    int temp9_row1=73758;


    for (int i=0;i<N_edge/10;i++)
	{
    	ap_uint<10*32> row_local;
    	row_local=row1.read();
    	int row[10];
    	for (int i = 0; i < 10; i++)
    	{
#pragma HLS UNROLL

    	    row[i] =row_local.range( (32* (i + 1) -1), i * 32) ;

    	}
    	ap_uint<10*32> col_local;
    	col_local=col1.read();
      	int col[10];
     	for (int i = 0; i < 10; i++)
    	{
    	#pragma HLS UNROLL
    	      col[i] =col_local.range( (32* (i + 1) -1), i * 32) ;

     	}

    	 int temp1_row=row[0];
    	 int temp1_col=col[0];

    	 int temp2_row=row[1];
    	 int temp2_col=col[1];

    	 int temp3_row=row[2];
    	 int temp3_col=col[2];

    	 int temp4_row=row[3];
    	 int temp4_col=col[3];

    	 int temp5_row=row[4];
    	 int temp5_col=col[4];

    	 int temp6_row=row[5];
    	 int temp6_col=col[5];

    	 int temp7_row=row[6];
    	 int temp7_col=col[6];

    	 int temp8_row=row[7];
    	 int temp8_col=col[7];

    	 int temp9_row=row[8];
    	 int temp9_col=col[8];

    	 int temp10_row=row[9];
    	 int temp10_col=col[9];


    		for (int k=0;k<N_P;k++)
    		{
#pragma HLS UNROLL

    			out1[temp1_row][k]+=in[temp1_col][k];
    		    out2[temp2_row-temp1_row1][k]+=in[temp2_col][k];
    		    out3[temp3_row- temp2_row1][k]+=in1[temp3_col][k];
    		    out4[temp4_row- temp3_row1][k]+=in1[temp4_col][k];
    		    out5[temp5_row- temp4_row1][k]+=in2[temp5_col][k];
    		    out6[temp6_row- temp5_row1][k]+=in2[temp6_col][k];
    		    out7[temp7_row- temp6_row1][k]+=in3[temp7_col][k];
    		    out8[temp8_row- temp7_row1][k]+=in3[temp8_col][k];
    		    out9[temp9_row- temp8_row1][k]+=in4[temp9_col][k];
    		    out10[temp10_row- temp9_row1][k]+=in4[temp10_col][k];

    		}

    }
    for (int l=0;l<N_P;l++)
    {

      	out1[temp1_row1][l]=out1[temp1_row1][l]+out2[0][l];
      	out2[temp2_row1-temp1_row1][l]=out2[temp2_row1-temp1_row1][l]+out3[0][l];
       	out3[temp3_row1-temp2_row1][l]=out3[temp3_row1-temp2_row1][l]+out4[0][l];
       	out4[temp4_row1-temp3_row1][l]=out4[temp4_row1-temp3_row1][l]+out5[0][l];
       	out5[temp5_row1-temp4_row1][l]=out5[temp5_row1-temp4_row1][l]+out6[0][l];
       	out6[temp6_row1-temp5_row1][l]=out6[temp6_row1-temp5_row1][l]+out7[0][l];
      	out7[temp7_row1-temp6_row1][l]=out7[temp7_row1-temp6_row1][l]+out8[0][l];
    	out8[temp8_row1-temp7_row1][l]=out8[temp8_row1-temp7_row1][l]+out9[0][l];
        out9[temp9_row1-temp8_row1][l]=out9[temp9_row1-temp8_row1][l]+out10[0][l];
    }

}

ap_uint<256> sampler( ap_uint<256> seed, int load) {
  static ap_uint<256> mask;
  if (load ==1 )
    mask = seed;
  bool b_32 = mask.get_bit(256-32);
  bool b_104 = mask.get_bit(256-104);
  bool b_248 = mask.get_bit(256-248);
  bool b_1 = mask.get_bit(256-1);
  bool new_bit = b_32 ^ b_104 ^ b_248 ^ b_1;
  mask = mask >> 1;
  mask.set_bit(255, new_bit);

  return mask.to_uint();

}







template<unsigned N_P,unsigned N_node1,unsigned N_node2,unsigned N_node3,unsigned N_node4,unsigned N_node5,unsigned N_node6,unsigned N_node7,unsigned N_node8,unsigned N_node9,unsigned N_node10 >
void wr(float in1[N_node1][N_P],float in2[N_node2][N_P],float in3[N_node3][N_P],float in4[N_node4][N_P],float in5[N_node5][N_P],float in6[N_node6][N_P],float in7[N_node7][N_P],float in8[N_node8][N_P],float in9[N_node9][N_P],float in10[N_node10][N_P], hls::stream <ap_uint<N_P*32>> &out)
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in1
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in2
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in3
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in4
	ap_uint<N_P*32> result_local1;
	ap_uint<N_P*32> result_local2;
	ap_uint<N_P*32> result_local3;
	ap_uint<N_P*32> result_local4;
	ap_uint<N_P*32> result_local5;
    ap_uint<N_P*32> result_local6;
    ap_uint<N_P*32> result_local7;
    ap_uint<N_P*32> result_local8;
    ap_uint<N_P*32> result_local9;
    ap_uint<N_P*32> result_local10;



	for(int l=0;l<N_node1;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut1 = 0;
		for (int i = 0; i < N_P; i++)
		{
#pragma HLS UNROLL
			bool temp=mask.get_bit(i);
			res = in1[l][i]*temp;
			tmpOut1.range(32 * (i + 1) - 1, i * 32) = res;
            }
		result_local1 = tmpOut1;
        out.write( result_local1);
	}
	for(int l=1;l<N_node2;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut2 = 0;
        for (int i = 0; i < N_P; i++)
        {
#pragma HLS UNROLL
        	bool temp=mask.get_bit(i);
        	res = in2[l][i]*temp;
	        tmpOut2.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local2 = tmpOut2;
        out.write( result_local2);
    }
	for(int l=1;l<N_node3;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut3 = 0;
        for (int i = 0; i < N_P; i++)
        {
#pragma HLS UNROLL
        	bool temp=mask.get_bit(i);
            res = in3[l][i]*temp;
            tmpOut3.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local3 = tmpOut3;
        out.write( result_local3);
    }
	for(int l=1;l<N_node4;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut4 = 0;
        for (int i = 0; i < N_P; i++)
       {
#pragma HLS UNROLL
        	bool temp=mask.get_bit(i);
            res = in4[l][i]*temp;
            tmpOut4.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local4 = tmpOut4;
        out.write( result_local4);
    }
	for(int l=1;l<N_node5;l++)
		{
		ap_uint<256> mask;
		sampler(mask,0);
			float res;
			ap_uint<N_P*32> tmpOut5 = 0;
	        for (int i = 0; i < N_P; i++)
	       {
	#pragma HLS UNROLL
	        	bool temp=mask.get_bit(i);
	            res = in5[l][i]*temp;
	            tmpOut5.range(32 * (i + 1) - 1, i * 32) = res;
	        }
	        result_local5 = tmpOut5;
	        out.write( result_local5);
	    }
	for(int l=1;l<N_node6;l++)
		{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut6 = 0;
        for (int i = 0; i < N_P; i++)
	    {
#pragma HLS UNROLL
	        	bool temp=mask.get_bit(i);
	            res = in6[l][i]*temp;
	            tmpOut6.range(32 * (i + 1) - 1, i * 32) = res;
	     }
	     result_local6 = tmpOut6;
	     out.write( result_local6);
	    }
	for(int l=1;l<N_node7;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut7 = 0;
        for (int i = 0; i < N_P; i++)
		{
#pragma HLS UNROLL
        	bool temp=mask.get_bit(i);
        	res = in7[l][i]*temp;
		    tmpOut7.range(32 * (i + 1) - 1, i * 32) = res;
		 }
		        result_local7 = tmpOut7;
		        out.write( result_local7);
	}
	for(int l=1;l<N_node8;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut8 = 0;
        for (int i = 0; i < N_P; i++)
        {
#pragma HLS UNROLL
           bool temp=mask.get_bit(i);
           res = in8[l][i]*temp;
           tmpOut8.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local8 = tmpOut8;
        out.write( result_local8);
	}
	for(int l=1;l<N_node9;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut9 = 0;
        for (int i = 0; i < N_P; i++)
        {
#pragma HLS UNROLL
           bool temp=mask.get_bit(i);
           res = in9[l][i]*temp;
           tmpOut9.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local9 = tmpOut9;
        out.write( result_local9);
	}
	for(int l=1;l<N_node10;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<N_P*32> tmpOut10 = 0;
        for (int i = 0; i < N_P; i++)
        {
#pragma HLS UNROLL
           bool temp=mask.get_bit(i);
           res = in10[l][i]*temp;
           tmpOut10.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local10 = tmpOut10;
        out.write( result_local10);
	}


 }


void TopFun( hls::stream <ap_uint<feature>> &in, hls::stream <ap_uint<10*32>> &col1,hls::stream <ap_uint<10*32>> &row1,data_bi weight[feature][block], float beta[node/B2/B][B][B2],float alpha[block],int valu[4][edge/4], hls::stream <ap_uint<P*32>> &out)
{




	sampler(10000, 1);

	data_bi temp_in[node/B2/B][B][B2][feature];
#pragma HLS BIND_STORAGE variable=temp_in type=ram_2p impl=uram
	read_bi<node,B,B2,feature>(in,temp_in);

    for(int k=0;k<13;k++)
    {

    	float out1[node1][block];
#pragma HLS BIND_STORAGE variable=out1 type=ram_2p impl=uram
    	float out2[node2][block];
#pragma HLS BIND_STORAGE variable=out2 type=ram_2p impl=uram
    	float out3[node3][block];
    	#pragma HLS BIND_STORAGE variable=out3 type=ram_2p impl=lutram
    	float out4[node4][block];
    	#pragma HLS BIND_STORAGE variable=out4 type=ram_2p impl=lutram
    	float out5[node5][block];
    	#pragma HLS BIND_STORAGE variable=out5 type=ram_2p impl=uram
    	float out6[node6][block];
    	#pragma HLS BIND_STORAGE variable=out6 type=ram_2p impl=uram
    	float out7[node7][block];
    	#pragma HLS BIND_STORAGE variable=out7 type=ram_2p impl=uram
    	float out8[node8][block];
     	#pragma HLS BIND_STORAGE variable=out8 type=ram_2p impl=bram
    	float out9[node9][block];
    	 #pragma HLS BIND_STORAGE variable=out9 type=ram_2p impl=uram
    	float out10[node10][block];
 #pragma HLS BIND_STORAGE variable=out10 type=ram_2p impl=lutram
#pragma HLS DATAFLOW
        data_bi temp2_out[node][block];
        data_bi temp2_out2[node][block];
#pragma HLS BIND_STORAGE variable=temp_out2 type=ram_2p impl=uram
         data_bi temp2_out3[node][block];
#pragma HLS BIND_STORAGE variable=temp_out3 type=ram_2p impl=lutram
        data_bi temp2_out4[node][block];
#pragma HLS BIND_STORAGE variable=temp_out4 type=ram_2p impl=uram
        data_bi temp2_out5[node][block];
#pragma HLS BIND_STORAGE variable=temp_out4 type=ram_2p impl=uram

        data_bi temp1_out1[node/B2/B][B][B2][block];
    	for(int i=0;i<4;i++)
    	{
#pragma HLS DATAFLOW
#pragma HLS BIND_STORAGE variable=temp_out1 type=ram_2p impl=uram
    		float temp_out1[node/B2/B][B][B2][block];
    		float beta2[2][2][2][block];
    		Extraction<B,node,feature,B2,block>(temp_in,weight,beta,alpha,temp_out1);
    		bi<B,node,B2,block>(temp_out1,temp1_out1,beta2);
    	}
    	cc<B,node,B2,block>(temp1_out1,temp2_out,temp2_out2,temp2_out3,temp2_out4,temp2_out5);
    	Aggregation<edge,block,node,node1,node2,node3,node4,node5,node6,node7,node8,node9,node10>(col1,row1,temp2_out,temp2_out2,temp2_out3,temp2_out4,temp2_out5,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10);
        wr<block,node1,node2,node3,node4,node5,node6,node7,node8,node9,node10>(out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out);
    }
}
