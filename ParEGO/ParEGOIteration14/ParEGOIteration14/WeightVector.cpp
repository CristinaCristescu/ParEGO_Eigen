/*
 * \class WeightVector
 *
 *
 * \brief Representation of weight for scalarizinf multiple dimension solutions.
 *
 * This class models the weighting system that enables use to scalarize
 * a multi-dimensional solution to be used in the optimization problem to create
 * a surrogate model function for the real expensive function.
 *
 * \note Copyright (c) 2006 Joshua Knowles. All rights reserved.
 *
 * \author (last to touch it) Bianca-Cristina Cristescu
 *
 * \version $Revision: 13
 *
 * \date $Date: 04/02/15.
 *
 */

#include "WeightVector.h"

#include <math.h>

#define MAX_K 5
#define NORM_WEIGHT 300

/**
  * Creates an instance of a weight vector.
  *
  * @param[in] objectivesNumber - The number of objectives the function has.
  */
WeightVector::WeightVector(int objectivesNumber):
normwv(NORM_WEIGHT, std::vector<double>(MAX_K))
{
    fobjectives = objectivesNumber;
    next = -1;
    add = 1;
    change = true;
    
    //the formula is: number of weight vectors = (N+k-1)!/N!(k-1)!
    //first dimension wv: #weight vectors^nobjs
    //first dimension normwv: N+k-1 choose k-1 ($^{N+k-1}C_{k-1}$)
    if(fobjectives == 2){
        N = 10;  // gives 11 weight vectors  - of objectives
        wvSize = pow(11, fobjectives);
        wv.resize(wvSize, std::vector<int>(MAX_K));
    }
    else if(fobjectives == 3)
    {
        N = 4;  // gives 15 weight vectors
        wvSize = pow(15, fobjectives);
        wv.resize(wvSize, std::vector<int>(MAX_K));

    }
    else if(fobjectives == 4)
    {
        N = 3;   // gives 20 weight vectors
        wvSize = pow(20, fobjectives);
        wv.resize(wvSize, std::vector<int>(MAX_K));
    }

    snake_wv();
}

/** Changes the weight vector.
  * The weights are changes every 5 iterations.
  *
  * @param[in] iter - The current iteration number.
  * @param[out] newVector - The new weights vector for the next iterations.
  */
void WeightVector::changeWeights(int iter, std::vector<double>& newVector)
{
    // Change the weight vector.
    if(iter%5 == 2)
        change = true;
    
    if(change)
    {
        if((next == 9)||((next == 0)&&(add == -1)))
            add *= -1;
        next += add;
        for(int k = 0; k < fobjectives; k++)
        {
            newVector[k] = normwv[next][k];
        }
    }
    
    change = false;
}

/** Creates evenly spaced normalized weight vectors.
  * This funtion uses the method of generating reflected k-ary Gray codes
  * in order to generate every normalized weight vector of 
  * k fSearchSpaceDimensions and s divisions, so that they `snake' in the space,
  * i.e., each weight vector is near the previous one. 
  * This is useful for MOO search using weight vectors.
  *
  */
void WeightVector::snake_wv()
{
    // First dimension needs to be #weight vectors^nobjs.
    std::vector<std::vector<double> > dwv(wvSize, std::vector<double>(MAX_K));
    int wvsSize;
    
    int i,j;
    int n;
    int m;
    int d = fobjectives-1;
    int sum;
    int count = 0;
    
    m = N;
    
    
    for(i = 0;i < N; i++)
        wv[i][d] = i;
    
    n = N;
    
    while(m < pow(N,fobjectives))
    {
        reverse(m);
        d--;
        m = N*m;
        for(i = 0; i < m; i++)
            wv[i][d] = i/(m/N);
    }
    
    for(i = 0; i < pow(N,fobjectives); i++)
    {
        for(j = 0; j < fobjectives; j++)
            dwv[i][j] = (double)wv[i][j]/(N-1.0);
    }
    
    count = 0;
    for(i = 0; i < pow(N,fobjectives); i++)
    {
        sum = 0;
        for(j = 0; j < fobjectives; j++)
            sum += wv[i][j];
        if(sum == N-1)
        {
            for(j = 0; j < fobjectives; j++)
                normwv[count][j] = dwv[i][j];
            count++;
        }
    }
    wvsSize = count;
}

/** Auxiliary method of the snake methods.
 *
 * @param[in] n - Internal paramenter.
 */
void WeightVector::reverse(int n)
{
    int h,i,j;
    for(i = 0; i < n; i++)
    {
        for(h = 0; h < N-1; h++)
        {
            for(j = 0; j < fobjectives; j++)
            {
                if(h%2 == 0)
                    wv[h*n+n+i][j] = wv[n-1-i][j];
                else
                    wv[h*n+n+i][j] = wv[i][j];
            }
        }
    }
    
}
