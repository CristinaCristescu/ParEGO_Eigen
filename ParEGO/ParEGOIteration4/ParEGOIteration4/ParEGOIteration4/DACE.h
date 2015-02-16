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

#include </Users/cristina/developer/eigen/eigen-eigen-1306d75b4a21/Eigen/Dense>
#include "SearchSpace.h"

using namespace::Eigen;

class DACE
{
public:
    SearchSpace* daceSpace;
    double gmu;
    double gsigma;
    double *gtheta;
    double *gp;
    double ymin;
    double gymin;
    MatrixXd pgR;
    MatrixXd pInvR;
    VectorXd pgy;
    
    double glik;

    
    //take them outttttt
    double ***pax; // a pointer 
    double **pay;

    DACE(SearchSpace* space);
    double correlation(double *xi, double *xj, double *theta, double *p, int dim);
    double mu_hat(VectorXd& y);
    double sigma_squared_hat(VectorXd& y);
    double weighted_distance(double *xi, double *xj);
    double predict_y(double **ax);
    double s2(double **ax);
    void build_R(double **ax,MatrixXd& R);
    void build_y(double *ay, VectorXd& y);
    long double posdet(MatrixXd& R, int n);
    
    double likelihood(double *param, double** pax, double* pay);
    
    void get_params(double** param);
    
    void buildDACE(double** param, double** bestparam, bool change, int iter);



};

#endif /* defined(__ParEGOIteration4__DACE__) */
