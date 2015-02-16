//
//  LatinHypercube.cpp
//  ParEGOIteration3
//
//  Created by Bianca Cristina Cristescu on 01/12/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//

#include "LatinHypercube.h"
#include <stdlib.h>

#define RN rand()/(RAND_MAX+1.0)

LatinHypercube::LatinHypercube(int iter, unsigned dim, double* xmin,
                                double* xmax)
{
    fIter = iter;
    fDim = dim;
    fXmin = xmin;
    fXmax = xmax;
}

void LatinHypercube::latin_hyp(double **ax)
{
    
    int v;
    
    double L[fDim][fIter];
    bool viable[fDim][fIter];
    
    
    for(int d=0; d<fDim; d++)
    {
        for(int i=0;i<fIter;i++)
        {
            viable[d][i]=true;
            
            L[d][i] = fXmin[d+1] + i*((fXmax[d+1]-fXmin[d+1])/double(fIter));
        }
    }
    for(int i=0; i<fIter; i++)
    {
        for(int d = 0; d < fDim; d++)
        {
            do
                v = int(RN*fIter);
            while(!viable[d][v]);
            viable[d][v]=false;
            ax[i+1][d+1] = L[d][v]+RN*((fXmax[d+1]-fXmin[d+1])/double(fIter));
        }
        
    }
}

