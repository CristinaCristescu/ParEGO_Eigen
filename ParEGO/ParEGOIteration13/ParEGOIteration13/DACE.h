//
//  DACE.h
//  ParEGOIteration4
//
//  Created by Bianca Cristina Cristescu on 05/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __ParEGOIteration4__DACE__
#define __ParEGOIteration4__DACE__

#include <stdio.h>
#include <vector>

#include "SearchSpace.h"
#include "Matrix.h"
#include "Vector.h"

class DACE
{
private:
    SearchSpace* daceSpace;
    int fNoParamDACE; // number of parameters in the DACE model pdim
    double gmu;
    double gsigma;
    double *gtheta;
    double *gp;
    double gymin;
    //TO DO: Pointer or objects?
    MyMatrix pInvR;
    MyVector pgy;
    double glik;
    double gz[76]; // an array holding the z values for the gaussian distribution
    MyMatrix one_pInvR;
    MyMatrix onetransp_pInvR;
    double onetransp_pInvR_one;
    MyVector predict_y_constant;
    

public:
    int fCorrelationSize; // global giving the size of the current correlation matrix R
    // being used titer
    //take them outttttt
    double ***pax; // a pointer 
    double **pay;
    double ymin;
    int debug_iter;

    DACE(SearchSpace* space);
    ~DACE();
    void buildDACE(bool change, int iter);
    double wrap_ei(double *x, int iter); // returns -(expected improvement), given the solution x
    
private:
    double correlation(double *xi, double *xj, double *theta, double *p, int dim);
    double mu_hat(MyVector& y, int iter);
    double sigma_squared_hat(MyVector& y);
    double weighted_distance(double *xi, double *xj);
    double predict_y(double **ax);
    double s2(double **ax);
    void build_R(double **ax,MyMatrix& R);
    void build_y(double *ay, MyMatrix& y);
    long double posdet(MyMatrix& R, int n);
    void init_gz();
    double standard_density(double z);
    double standard_distribution(double z);
    double expected_improvement(double yhat, double ymin, double s);
    double likelihood(double *param, double** pax, double* pay);
    
};

#endif /* defined(__ParEGOIteration4__DACE__) */
