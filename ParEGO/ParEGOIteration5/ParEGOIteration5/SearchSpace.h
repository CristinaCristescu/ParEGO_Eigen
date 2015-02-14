//
//  SearchSpace.h
//  ParEGOIteration4
//
//  Created by Bianca Cristina Cristescu on 03/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __ParEGOIteration4__SearchSpace__
#define __ParEGOIteration4__SearchSpace__

#define MAX_K 5

#include <stdio.h>

class DACE;

// The class models the search space and sets the fitness function.
class SearchSpace
{
//change to private and make setters and getters
private:
    const char* fObjectiveFunctionName; // name of the objective function

public:
    
    int fNoObjectives; //the number of objectives nobjs
    int fSearchSpaceDim;  // dimension of the actual search space, X dim
    double **fCostVectors; // two-dimensional array of all multiobjective cost vectors ff
    double **fXVectors;  // two-dimensional array of all x vectors ax
    double **fSelectedXVectors; // two dimensional array of selected x vectors tmpax
    double *fXMin; // upper and lower constraints on the search space
    double *fXMax;
    double *fMeasuredFit; // array of true/measured y fitnesses ay
    double *fSelectedMeasuredFit;
    
    double fAbsMax[MAX_K]; //absolute values of the maximum and minimum of each
                          //objective space dimension
    double fAbsMin[MAX_K];
    double fIdealObjective[MAX_K]; //objective ideal points
    double fWeightVectors[MAX_K]; //weight vectors
        
    double alph = 0.0; // power in the DTLZ2 and DTLZ4 test functions


public:
    SearchSpace();
    SearchSpace(const char* function_name, int max_iter);
    void SetSearch();
    void init_arrays(int n);
    void applyFit(int i, int j);
    void chooseUpdateSolutions(int iter, int correlation_size);
    double Tcheby(double* vec);
    double myfit(int i, int j);
    
private:

    void f_dtlz1a(double *x, double *y);
    void f_dtlz2a(double *x, double *y);
    void f_dtlz7a(double *x, double *y);
    void f_vlmop3(double *x, double *y);
    void f_oka1(double *x, double *y);
    void f_oka2(double *x, double *y);
    void f_vlmop2(double *x, double *y);
    void f_kno1(double *x, double *y);
    
    
    
};

#endif /* defined(__ParEGOIteration4__SearchSpace__) */
