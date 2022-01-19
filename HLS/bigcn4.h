#define node 1024
#define feature 500
#define P 13
#define B 10
#define hidden 256

#define edge 10000


#include<hls_stream.h>
#include"ap_int.h"




typedef ap_uint<1> data_bi;

void TopFun(data_bi in[edge][feature], data_bi weight[feature][P][B],float beta[edge],float alpha[P][B],int valu[edge],int col[edge],int row[edge], float out[node][P][B]);




