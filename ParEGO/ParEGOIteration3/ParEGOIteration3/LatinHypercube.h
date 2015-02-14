//
//  LatinHypercube.h
//  ParEGOIteration3
//
//  Created by Bianca Cristina Cristescu on 01/12/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __ParEGOIteration3__LatinHypercube__
#define __ParEGOIteration3__LatinHypercube__

#include <stdio.h>

class LatinHypercube
{
private:
    unsigned fDim; // Dimensions of the search space
    double* fXmin; // Lower bound on the search space
    double* fXmax; // Upper bound on the search space
    int fIter; // Number of iterations
    
public:
    LatinHypercube(int iter, unsigned dim, double* xmin, double* xmax);
    void latin_hyp(double **ax);
};
#endif /* defined(__ParEGOIteration3__LatinHypercube__) */
