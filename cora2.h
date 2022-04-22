#define node 2708
#define node1 500
#define node2 501
#define node3 855
#define node4 855
#define feature 45
#define P 44
#define B 1
#define edge 10556
#define hidden 256


#include<hls_stream.h>
#include"ap_int.h"




typedef ap_uint<1> data_bi;
typedef ap_uint<32> data_t;

void TopFun( hls::stream <ap_uint<feature>> &in, data_t weight[feature][P],float beta[node],float alpha[P],int valu[4][edge/4],int col[4][edge/4],int row[4][edge/4], hls::stream <ap_uint<P*B*32>> &out);
