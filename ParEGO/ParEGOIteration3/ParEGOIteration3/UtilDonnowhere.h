//
//  UtilDonnowhere.h
//  ParEGOIteration3
//
//  Created by Bianca Cristina Cristescu on 03/12/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef ParEGOIteration3_UtilDonnowhere_h
#define ParEGOIteration3_UtilDonnowhere_h

#include </Users/cristina/developer/eigen/eigen-eigen-1306d75b4a21/Eigen/Dense>

using namespace Eigen;

void cwr(int **target, int k, int n);  // choose without replacement k items from n
extern double myabs(double v);
extern void mysort(int *idx, double *val, int num);
extern double standard_density(double z);
extern double standard_density(double z);
extern double standard_distribution(double z, double gz[76]);
extern double expected_improvement(double yhat, double ymin, double s, double gz[76]);
extern double weighted_distance(double *xi, double *xj, double *theta, double *p, int dim,
                                double* xmin, double* xmax);

extern double correlation(double *xi, double *xj, double *theta, double *p, int dim, double* xmin, double* xmax);

extern double predict_y(double **ax, MatrixXd& InvR, VectorXd& y, double mu_hat,
                        double *theta, double *p, int n, unsigned dim, double* xmin, double* xmax);
extern double s2(double **ax, double *theta, double *p, double sigma, int dim, int n, MatrixXd& InvR, double* xmin, double* xmax);
extern double wrap_ei(double *x, unsigned dim, int titer, double** pax, MatrixXd* pInvR,
                      double* gtheta, double* gp, double gsigma, double gymin,
                      double* xmin, double* xmax, VectorXd* pgy, double globalmu, double gz[76]);

#endif
