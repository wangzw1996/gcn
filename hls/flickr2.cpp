
#include"flickr2.h"
#include <stdint.h>
#include <cmath>
#include<stdlib.h>
#include<iostream>
#include"ap_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>




void read_bi(hls::stream <ap_uint<500>> &in,data_t out[node/B2/B][B][B2][feature])
{
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=out
	ap_uint<500> in_local[node/B2/B][B][B2];
	for (int j = 0; j < node/B2/B; j++)
	{
		for (int m=0;m<B;m++)
		{
			for(int k=0;k<B2;k++)
			{
				in_local[j][m][k]=in.read();
				for (int i = 0; i < feature; i++)
				{
#pragma HLS UNROLL
					if ((i+1)*32<500)
					{
						out[j][m][k][i] =in_local[j][m][k].range( (32* (i + 1) -1), 32 * i) ;
					}
					else
					{
						out[j][m][k][i]=in_local[j][m][k].range( (32* (i + 1) -13), 32 * i)	;
					}
				}
			}
		}
	}

}





void Extraction(data_t in[node/B2/B][B][B2][feature], data_t weight[feature][block],float beta[node/B2/B][B][B2],float alpha[block],float out[node/B2/B][B][B2][block])
{

#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=beta

	for(int l=0;l<node/B2/B;l++)
	  {

		for (int m=0;m<B2;m++)
		{
			for(int i=0;i<block;i++)
			{
#pragma HLS PIPELINE
				for (int j=0;j<B;j++)
				{
#pragma HLS UNROLL
					ap_uint<32> temp=0;
					data_bi temp2;
					int bitcount1 =0;
					for(int k=0;k<feature;k++)
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




void bi(float in[node/B2/B][B][B2][block],data_bi out[node/B2/B][B][B2][block],float gama[2][2][2][block])
{
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=out

	float mean=0;
    float mean_1[node/B2/B][block]={0};
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=mean_1
    for(int j=0; j<node/B2/B;j++)
	{
     for(int m=0;m<B;m++)
     {
#pragma HLS UNROLL
		for(int i=0;i<B2;i++)
		{
#pragma HLS UNROLL
			for(int k=0;k<block;k++)
			{
#pragma HLS UNROLL
				mean_1[j][k] += in[j][m][i][k];
			}
		}
	 }
    }

	float mean_3[block]={0};
	for(int j=0; j<node/B2/B;j++)
	{
		for(int k=0;k<block;k++)
		{
#pragma HLS UNROLL
				mean_3[k]+=mean_1[j][k]/node;
		}
	}


	float temp1[node/B2/B][block]={0};
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=temp1
		float sum_1[node/B2/B][block]={0};
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=sum_1
	    float sum[block]={0};
	    float sum3[block]={0};
	for(int j=0; j<node/B2/B;j++)
	{
		for(int m=0;m<B;m++)
		{
#pragma HLS UNROLL
			for(int i=0;i<B2;i++)
			{
#pragma HLS UNROLL
				for(int k=0;k<block;k++)
				{
#pragma HLS UNROLL
					float c;
					c=in[j][m][i][k]-mean_3[k];
					if (c>0)
					{
					    out[j][m][i][k]=1;
					}
					else
					{
						out[j][m][i][k]=0;
					}
					temp1[j][k]+=c*c;
					sum_1[j][k]+=abs(c);
				}
			}
		}
	}

	for(int j=0; j<node/B2/B;j++)
	{
		for(int k=0;k<block;k++)
	  {
#pragma HLS UNROLL
		  sum[k]+=temp1[j][k];
		  sum3[k]+=abs(sum_1[j][k]);
	   }
	}
	float std[block]={0};
	for(int k=0;k<block;k++)
	{
			std[k]=sqrt(sum[k]/node);
			gama[0][0][0][k]=sum3[k]/std[k]/node;
	}
}





void ddd(data_t in[feature][block*block2],data_t out[feature][block], int i,float alpha_in[block*block2],float alpha_out[block])
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
	for (int j=0;j<feature;j++)
	{
		for(int k=0;k<block;k++)
		{
	#pragma HLS UNROLL
			out[j][k]=in[j][k+i*block];
			alpha_out[k]=alpha_in[k+i*block];
		}
	}
}



void ccc(data_bi in[node/B2/B][B][B2][block], float in_gama[2][2][2][block], float out_gama[2][2][2][block*block2],data_bi out[node/B2/B][B][B2][block*block2],int index)
{
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=out
	for(int j=0; j<node/B2/B;j++)
	{
		for(int k=0;k<block;k++)
		{
#pragma HLS UNROLL
			for(int i=0;i<B;i++)
			{
#pragma HLS UNROLL
				for(int m=0;m<B2;m++)
			    {
#pragma HLS UNROLL
					out[j][i][m][k+index*block] = in[j][i][m][k];

				}
			}
		}
	}
	for(int j=0; j<2;j++)
	{
		for(int m=0;m<2;m++)
	    {
			for(int i=0;i<2;i++)
			{
				for(int k=0;k<block;k++)
				{
#pragma HLS UNROLL
					out_gama[j][i][m][k+index*block]=in_gama[j][i][m][k];

				}
			}
		}

	}
}







void exbi(data_t in[node/B2/B][B][B2][feature], data_t weight[feature][block*block2],float beta[node/B2/B][B][B2],float alpha[block*block2],data_bi out[node/B2/B][B][B2][block*block2],float gama[2][2][2][block*block2])
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight

	data_t temp_weight[feature][block];
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=temp_weight
	float temp_alpha[block];
	float temp_out1[node/B2/B][B][B2][block];

	data_bi temp_out2[node/B2/B][B][B2][block];
#pragma HLS BIND_STORAGE variable=temp_out2 type=ram_2p impl=lutram

	float temp_gama[2][2][2][block];
	data_bi temp_out3[node/B2/B][B][B2][block*block2];
	for (int i=0;i<block2;i++)
	{
#pragma HLS DATAFLOW

		ddd(weight,temp_weight,i,alpha,temp_alpha);
		Extraction(in,temp_weight,beta,temp_alpha,temp_out1);
		bi(temp_out1,temp_out2,temp_gama);
		ccc(temp_out2,temp_gama,gama,out,i);
	}
}



void cc(data_bi in[node/B2/B][B][B2][block*block2],data_bi out[node][block*block2][5])
{
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=out
	for(int i=0; i<node/B2/B;i++)
	{
		int cum=0;
		for(int k=0;k<B2;k++)
		{
			for(int j=0;j<B;j++)
			{
#pragma HLS PIPELINE
			for(int m=0;m<block*block2;m++)
				{
#pragma HLS UNROLL
					for(int l=0;l<5;l++)
					{
#pragma HLS UNROLL
						float c;
						c=in[i][j][k][m];
					    out[k+j*B2+i*(B+B2)][m][l]=c;
					}
				}
			}
		}
	}

}




void Aggregation( int col[10][edge/10],int row[10][edge/10],data_bi in[node][block*block2][5],float out1[node1][block*block2],float out2[node2][block*block2],float out3[node3][block*block2],float out4[node4][block*block2],float out5[node5][block*block2],float out6[node6][block*block2],float out7[node7][block*block2],float out8[node8][block*block2],float out9[node9][block*block2],float out10[node10][block*block2])

{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=in
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=col
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=row



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


    for (int i=0;i<edge/10;i++)
	{

    	 int temp1_row=row[0][i];
    	 int temp1_col=col[0][i];

    	 int temp2_row=row[1][i];
    	 int temp2_col=col[1][i];

    	 int temp3_row=row[2][i];
    	 int temp3_col=col[2][i];

    	 int temp4_row=row[3][i];
    	 int temp4_col=col[3][i];

    	 int temp5_row=row[4][i];
    	 int temp5_col=col[4][i];

    	 int temp6_row=row[5][i];
    	 int temp6_col=col[5][i];

    	 int temp7_row=row[6][i];
    	 int temp7_col=col[6][i];

    	 int temp8_row=row[7][i];
    	 int temp8_col=col[7][i];

    	 int temp9_row=row[8][i];
    	 int temp9_col=col[8][i];

    	 int temp10_row=row[9][i];
    	 int temp10_col=col[9][i];


    		for (int k=0;k<block*block2;k++)
    		{
#pragma HLS UNROLL

    			out1[temp1_row][k]+=in[temp1_col][k][0];
    		    out2[temp2_row-temp1_row1][k]+=in[temp2_col][k][0];
    		    out3[temp3_row- temp2_row1][k]+=in[temp3_col][k][1];
    		    out4[temp4_row- temp3_row1][k]+=in[temp4_col][k][1];
    		    out5[temp5_row- temp4_row1][k]+=in[temp5_col][k][2];
    		    out6[temp6_row- temp5_row1][k]+=in[temp6_col][k][2];
    		    out7[temp7_row- temp6_row1][k]+=in[temp7_col][k][3];
    		    out8[temp8_row- temp7_row1][k]+=in[temp8_col][k][3];
    		    out9[temp9_row- temp8_row1][k]+=in[temp9_col][k][4];
    		    out10[temp10_row- temp9_row1][k]+=in[temp10_col][k][4];

    		}

    }
    for (int l=0;l<block*block2;l++)
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








void wr(float in1[node1][block*block2],float in2[node2][block*block2],float in3[node3][block*block2],float in4[node4][block*block2],float in5[node5][block*block2],float in6[node6][block*block2],float in7[node7][block*block2],float in8[node8][block*block2],float in9[node9][block*block2],float in10[node10][block*block2], hls::stream <ap_uint<block*block2*32>> &out)
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in1
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in2
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in3
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in4
	ap_uint<block*block2*32> result_local1;
	ap_uint<block*block2*32> result_local2;
	ap_uint<block*block2*32> result_local3;
	ap_uint<block*block2*32> result_local4;
	ap_uint<block*block2*32> result_local5;
    ap_uint<block*block2*32> result_local6;
    ap_uint<block*block2*32> result_local7;
    ap_uint<block*block2*32> result_local8;
    ap_uint<block*block2*32> result_local9;
    ap_uint<block*block2*32> result_local10;



	for(int l=0;l<node1;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut1 = 0;
		for (int i = 0; i < block*block2; i++)
		{
#pragma HLS UNROLL
			bool temp=mask.get_bit(i);
			res = in1[l][i]*temp;
			tmpOut1.range(32 * (i + 1) - 1, i * 32) = res;
            }
		result_local1 = tmpOut1;
        out.write( result_local1);
	}
	for(int l=1;l<node2;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut2 = 0;
        for (int i = 0; i < block*block2; i++)
        {
#pragma HLS UNROLL
        	bool temp=mask.get_bit(i);
        	res = in2[l][i]*temp;
	        tmpOut2.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local2 = tmpOut2;
        out.write( result_local2);
    }
	for(int l=1;l<node3;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut3 = 0;
        for (int i = 0; i < block*block2; i++)
        {
#pragma HLS UNROLL
        	bool temp=mask.get_bit(i);
            res = in3[l][i]*temp;
            tmpOut3.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local3 = tmpOut3;
        out.write( result_local3);
    }
	for(int l=1;l<node4;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut4 = 0;
        for (int i = 0; i < block*block2; i++)
       {
#pragma HLS UNROLL
        	bool temp=mask.get_bit(i);
            res = in4[l][i]*temp;
            tmpOut4.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local4 = tmpOut4;
        out.write( result_local4);
    }
	for(int l=1;l<node5;l++)
		{
		ap_uint<256> mask;
		sampler(mask,0);
			float res;
			ap_uint<block*block2*32> tmpOut5 = 0;
	        for (int i = 0; i < block; i++)
	       {
	#pragma HLS UNROLL
	        	bool temp=mask.get_bit(i);
	            res = in5[l][i]*temp;
	            tmpOut5.range(32 * (i + 1) - 1, i * 32) = res;
	        }
	        result_local5 = tmpOut5;
	        out.write( result_local5);
	    }
	for(int l=1;l<node6;l++)
		{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut6 = 0;
        for (int i = 0; i < block; i++)
	    {
#pragma HLS UNROLL
	        	bool temp=mask.get_bit(i);
	            res = in6[l][i]*temp;
	            tmpOut6.range(32 * (i + 1) - 1, i * 32) = res;
	     }
	     result_local6 = tmpOut6;
	     out.write( result_local6);
	    }
	for(int l=1;l<node7;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut7 = 0;
        for (int i = 0; i < block*block2; i++)
		{
#pragma HLS UNROLL
        	bool temp=mask.get_bit(i);
        	res = in7[l][i]*temp;
		    tmpOut7.range(32 * (i + 1) - 1, i * 32) = res;
		 }
		        result_local7 = tmpOut7;
		        out.write( result_local7);
	}
	for(int l=1;l<node8;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut8 = 0;
        for (int i = 0; i < block*block2; i++)
        {
#pragma HLS UNROLL
           bool temp=mask.get_bit(i);
           res = in8[l][i]*temp;
           tmpOut8.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local8 = tmpOut8;
        out.write( result_local8);
	}
	for(int l=1;l<node9;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut9 = 0;
        for (int i = 0; i < block*block2; i++)
        {
#pragma HLS UNROLL
           bool temp=mask.get_bit(i);
           res = in9[l][i]*temp;
           tmpOut9.range(32 * (i + 1) - 1, i * 32) = res;
        }
        result_local9 = tmpOut9;
        out.write( result_local9);
	}
	for(int l=1;l<node10;l++)
	{
		ap_uint<256> mask;
		sampler(mask,0);
		float res;
		ap_uint<block*block2*32> tmpOut10 = 0;
        for (int i = 0; i < block; i++)
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



void TopFun( hls::stream <ap_uint<500>> &in, int col[10][edge/10],int row[10][edge/10],data_t weight[feature][block*block2], float beta[node/B2/B][B][B2],float alpha[block*block2],int valu[4][edge/4], hls::stream <ap_uint<block*block2*32>> &out)
{


	sampler(10000, 1);

	data_t temp_in[node/B2/B][B][B2][feature];
#pragma HLS BIND_STORAGE variable=temp_in type=ram_2p impl=uram

	read_bi(in,temp_in);

    for(int k=0;k<16;k++)
    {
    	float out1[node1][block*block2];
#pragma HLS BIND_STORAGE variable=out1 type=ram_2p impl=uram
    	float out2[node2][block*block2];
#pragma HLS BIND_STORAGE variable=out2 type=ram_2p impl=uram
    	float out3[node3][block*block2];
#pragma HLS BIND_STORAGE variable=out3 type=ram_2p impl=uram
    	float out4[node4][block*block2];
#pragma HLS BIND_STORAGE variable=out4 type=ram_2p impl=uram
    	float out5[node5][block*block2];
#pragma HLS BIND_STORAGE variable=out5 type=ram_2p impl=uram
    	float out6[node6][block*block2];
#pragma HLS BIND_STORAGE variable=out6 type=ram_2p impl=uram
    	float out7[node7][block*block2];
#pragma HLS BIND_STORAGE variable=out7 type=ram_2p impl=uram
    	float out8[node8][block*block2];
    	float out9[node9][block*block2];
    	float out10[node10][block*block2];


#pragma HLS DATAFLOW

        float gama[2][2][2][block*block2];
        data_bi temp_out1[node/B2/B][B][B2][block*block2];
#pragma HLS BIND_STORAGE variable=temp_out1 type=ram_2p impl=lutram

        data_bi temp_out2[node][block*block2][5];



        exbi(temp_in,weight,beta,alpha,temp_out1,gama);
    	cc(temp_out1,temp_out2);
    	Aggregation(col,row,temp_out2,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10);
        wr(out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out);
    }
}
