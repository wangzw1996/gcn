#define node 2700
#define node_all 89250
#define node1 1682
#define node2 3370
#define node3 5378
#define node4 7147
#define node5  8634
#define node6 9941
#define node7  11208
#define node8 12527
#define node10  15492
#define feature 500
#define P 5
#define B 1
#define B2 20
#define B3 4
#define edge 892500
#define hidden 4
#define block 20
#define block1 10

#include<hls_stream.h>
#include"ap_int.h"



typedef ap_uint<1> data_bi;
typedef ap_uint<32> data_t;


void TopFun( hls::stream <ap_uint<feature>> &in, hls::stream <ap_uint<10*32>> &col1,hls::stream <ap_uint<10*32>> &row1,data_bi weight[feature][P], float beta[node],float alpha[P],int valu[4][edge/4], hls::stream <ap_uint<P*32>> &out);
