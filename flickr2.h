#define node 89250
#define node1 1682
#define node2 3370
#define node3 5378
#define node4 7147
#define node5  8634
#define node6 9941
#define node7  11208
#define node8 12527
#define node9 13900
#define node10  15492
#define feature 500/32+1
#define B 20
#define B2 5
#define edge 892500
#define hidden 256
#define block 4
#define block2 4

#include<hls_stream.h>
#include"ap_int.h"



typedef ap_uint<1> data_bi;
typedef ap_uint<32> data_t;


void TopFun( hls::stream <ap_uint<500>> &in, int col[10][edge/10],int row[10][edge/10],data_t weight[feature][block*block2], float beta[node/B2/B][B][B2],float alpha[block*block2],int valu[4][edge/4], hls::stream <ap_uint<block*block2*32>> &out);
void cc(data_bi in[node/B2/B][B][B2][block*block2],data_bi out[node][block*block2][5]);
