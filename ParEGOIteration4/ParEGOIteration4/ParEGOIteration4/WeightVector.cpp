//
//  WeightVector.cpp
//  ParEGOIteration4
//
//  Created by Bianca Cristina Cristescu on 04/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#include "WeightVector.h"

#include <math.h>

WeightVector::WeightVector()
{
    fobjectives = 0;
    N = 0;
    next = -1;
    add = 1;
    change = true;
}

WeightVector::WeightVector(int noobjectives)
{
    fobjectives = noobjectives;
    
    next = -1;
    add = 1;
    change = true;
    
    if(fobjectives==2)
        N=10;  // gives 11 weight vectors  - the formula is: number of weight vectors = (N+k-1)!/N!(k-1)! where k is number of objectives
    else if(fobjectives==3)
        N=4;  // gives 15 weight vectors
    else if(fobjectives==4)
        N=3;   // gives 20 weight vectors
    
    snake_wv();
}

int static mypow(int x, int expon)
{
    return ((int)(pow((double)x, (double)expon)));
}

void WeightVector::changeWeights(int iter, double* newvector)
{
    //  double recomp_prob=1.0;
    
    if(iter%5==2)
        change = true;  // change the weight vector
    
    if(change)
    {
        if((next==9)||((next==0)&&(add==-1)))
            add*=-1;
        next+=add;
        for(int k=0;k<fobjectives;k++)
        {
            newvector[k]=normwv[next][k];
        }
    }
    
    change = false;
    
          //fprintf(stdout, "%.2lf %.2lf weightvectors %d %d\n", newvector[0], newvector[1], next, add);
}

// function to create evenly spaced normalized weight vectors
void WeightVector::snake_wv()
{
    // This funtion uses the method of generating reflected k-ary Gray codes
    // in order to generate every normalized weight vector of k fSearchSpaceDimensions and
    // s divisions, so that they `snake' in the space, i.e., each wv is near the
    // previous one. This is useful for MOO search using weight vectors.
    
    int i,j;
    int n;
    int m;
    int d=fobjectives-1;
    //int b;
    int sum;
    int count=0;
    
    m=N;
    
    
    for(i=0;i<N;i++)
        wv[i][d]=i;
    
    n=N;
    
    while(m<mypow(N,fobjectives))
    {
        reverse(m);
        d--;
        m=N*m;
        for(i=0;i<m;i++)
            wv[i][d]=i/(m/N);
        
        //      for(i=0;i<m;i++)
        //	print_vec(wv[i],5);
        
    }
    //  for(i=0;i<pow(s,k);i++)
    //  print_vec(wv[i],k);
    
    for(i=0;i<mypow(N,fobjectives);i++)
    {
        for(j=0;j<fobjectives;j++)
            dwv[i][j]=(double)wv[i][j]/(N-1.0);
    }
    
    count=0;
    for(i=0;i<mypow(N,fobjectives);i++)
    {
        sum=0;
        for(j=0;j<fobjectives;j++)
            sum+=wv[i][j];
        if(sum==N-1)
        {
            //  print_vec_double(dwv[i],k);
            for(j=0;j<fobjectives;j++)
                normwv[count][j]=dwv[i][j];
            count++;
        }
    }
    //  printf("\n\n");
    //  printf("%d weight vectors generated\n",count);
    nwvs=count;
}

void WeightVector::reverse(int n)
{
    int h,i,j;
    for(i=0;i<n;i++)
    {
        for(h=0;h<N-1;h++)
        {
            for(j=0;j<fobjectives;j++)
            {
                if(h%2==0)
                    wv[h*n+n+i][j]=wv[n-1-i][j];
                else
                    wv[h*n+n+i][j]=wv[i][j];
            }
        }
    }
    
}
