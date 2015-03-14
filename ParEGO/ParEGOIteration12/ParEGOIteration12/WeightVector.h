//
//  WeightVector.h
//  ParEGOIteration4
//
//  Created by Bianca Cristina Cristescu on 04/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __ParEGOIteration4__WeightVector__
#define __ParEGOIteration4__WeightVector__

#define MAX_K 5

#include <stdio.h>

class WeightVector
{
public:
    int N; // parameter relating to the number of divisions
    int wv[40402][MAX_K];      // first dimension needs to be #weight vectors^nobjs
    double normwv[300][MAX_K]; // first dimension needs to be N+k-1 choose k-1 ($^{N+k-1}C_{k-1}$) , where N is a parameter and k=nobjs
    int fobjectives;
    int next;
    int add;
    bool change;
    
    WeightVector();
    WeightVector(int noobjectives);
    void changeWeights(int iter, double* newvector);
    void snake_wv();
    void reverse(int n);


};

#endif /* defined(__ParEGOIteration4__WeightVector__) */
