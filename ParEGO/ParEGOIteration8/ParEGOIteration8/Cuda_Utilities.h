extern "C" void matrixMul(int HA, int WA, int HB, int WB, int HC, int WC,
                          float* A, float* B, float* C);
extern "C" void myalloc(int iter);
extern "C" void mydealloc();