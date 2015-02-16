//
//  GeneticAlgorithm.h
//  ParEGOIteration4
//
//  Created by Bianca Cristina Cristescu on 03/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __ParEGOIteration4__GeneticAlgorithm__
#define __ParEGOIteration4__GeneticAlgorithm__

#include <stdio.h>

#include "SearchSpace.h"

#define INFTY 1.0e35;

class DACE;

class GeneticAlgorithm
{
private:
    int gapopsize;
    int chromosomedim;
    double **popx;
    double *popy;
    double *mutx;
    double muty;
    int parA;
    int parB;
public:
    GeneticAlgorithm(int pop_size, int dim);
    ~GeneticAlgorithm();
    
    void run(SearchSpace* space, DACE* model, int iter, double* bestparam, double* best_imp);
    
private:
    void cross(double *child, double *par1, double *par2, SearchSpace* space);
    int tourn_select(double *xsel, double **ax, double *ay, int iter, int t_size, SearchSpace* space);
    void mutate(double *x, SearchSpace* space);
    
};

#endif /* defined(__ParEGOIteration4__GeneticAlgorithm__) */

