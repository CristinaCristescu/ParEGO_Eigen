
#include <stdio.h>
#include "Vector.h"

using namespace MATPACK;

int main()
{
    int n =10;
    Matrix R(0,9,0,9);
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            R[i][j] = rand()%1000;
    
    double s2;
    
    Matrix InvR = R.Inverse();
    
    Vector one(0,9);
    for(int i=0;i<n;i++)
        one(i)=1;
    
    Vector r(0,9);
    for(int i=0;i<n;i++)
    {
        r[i] = rand()%10;
    }
    
    //s2 = sigma * (1 - r*InvR*r + pow((1-one*InvR*r),2)/(one*InvR*one) );
    s2 = 1 - r*InvR*r;
    printf("res:%f", s2);

    return 0;
 }   