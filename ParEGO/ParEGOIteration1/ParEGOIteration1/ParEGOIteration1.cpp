//
//  ParEGOIteration1.cpp
//  ParEGOIteration1
//
//  Created by Bianca Cristina Cristescu on 30/10/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//

#include "ParEGOIteration1.h"

/****************************************************************
 
 The Pareto Efficient Global Optimization Algorithm: ParEGO
 (C) Joshua Knowles, October 2004. j.knowles@manchester.ac.uk
 
 
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or (at
 your option) any later version.
 
 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 
 
 ******************************************************************
 MORE INFO
 ParEGO is described in the following paper:
 J. Knowles (2004) ParEGO: A Hybrid Algorithm with On-line
 Landscape Approximation for Expensive Multiobjective Optimization
 Problems. Technical Report TR-COMPSYSBIO-2004-01, University of
 Manchester, Manchester, UK. September 2004.
 Available from  http://dbkweb.ch.umist.ac.uk/knowles/parego/
 Please read this first.
 ******************************************************************
 
 
 ******************************************************************
 COMPILE
 To compile this source code, you will need to install the open
 source Matpack libraries available from http://www.matpack.de/
 
 Edit the matpack path in the makefile to point to the appropriate
 directory.
 
 Then:
 make ParEGO
 ******************************************************************
 
 
 ******************************************************************
 RUN
 ./ParEGO 3563563 f_oka2
 
 runs ParEGO on the OKA2 fitness function with random seed 3563563.
 
 Other fitness functions included in this code are called with
 the following parameters:
 
 f_oka1
 f_kno1
 f_vlmop2 (also the default if no params are given)
 f_vlmop3
 f_dtlz1a
 f_dtlz2a
 f_dtlz4a
 f_dtlz7a
 ******************************************************************
 
 
 ******************************************************************
 OTHER FITNESS FUNCTIONS
 To add your own fitness function to this code, search for the text
 "HERE" which appears several times in this file. Instructions on
 adding the fitness function appear there.
 ******************************************************************
 
 
 ******************************************************************
 OUTPUT
 The output of ParEGO to standard out has the following form:
 
 1 -1.264347 0.870591 decision
 1 0.980027 0.939169 objective
 2 -1.154172 1.878366 decision
 2 0.992063 0.998977 objective
 ...
 .
 .
 21 1.959805 1.678096 decision
 21 0.9189 0.999997 objective
 ei= -0.021539
 22 -0.769814 -0.907925 decision
 22 0.991684 0.0432951 objective
 ...
 .
 .
 ei= -0.022139
 200 0.755171 0.558005 decision
 200 0.0242429 0.976216 objective
 
 i.e., there are three types of output line. A decision vector
 line has the iteration, a list of the decision variable values, and
 the identifier "decision". An objective vector line follows a similar
 format. The first 21 lines above (= 11d-1) consist of only decision
 and objective vectors. After that, a third type of line also appears.
 This third type of output line gives the
 expected improvement in the (scalarized) fitness for the *next*
 solution being proposed by ParEGO. A negative value means an
 improvement.
 ******************************************************************
 
 
 
 ******************************************************************
 NOTES
 1. The code here has very little structure as it is a quick and
 dirty conversion to C++ from C code that originally had lots of
 global variables. I am sorry. It really needs to be restructured
 in a modular way but I don't have time now.
 
 2. Because of 1, there is no current easy way to set parameters.
 At present, ParEGO uses 11d-1 initial solutions, where d is the
 number of decision space dimensions. Then it runs for a maximum
 of MAX_ITERS evaluations. The latter can be changed by setting
 MAX_ITERS in the set_search() function. For everything else, you
 will have to edit this source code at your peril.
 
 3. In an earler version, ParEGO used a downhill simplex method
 to find the
 maximum likelihood model. However, I later found that using a
 random search was just as good - and simplified the code a lot.
 So this version does not use downhill simplex.
 
 4. ParEGO gets progressively slower as number of previous
 points visited, and hence the dimension of the
 correlation matrix R, increases. For this reason, we do not
 use information from all previous points visited; just a
 selection of them. This reduces the performance slightly, and
 means that ParEGO won't necessarily
 converge accurately to a Pareto front with more iterations. With
 larger numbers of iterations, it would be best to use ParEGO to
 do the first 200 evaluations and then continue on with a standard
 MOEA, eg NSGA-II or PESA-II etc. I have experimented with
 restarting ParEGO but this does not seem to yield particularly
 good results.
 
 5. Please email me if you have comments, criticisms or requests.
 Joshua Knowles j.knowles@manchester.ac.uk
 ******************************************************************/


#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <set>
#include <cmath>
#include <cstdio>
#include <cstdlib>


#define LARGE 2147000000
#define MAX_K 5
#define INFTY 1.0e35;
#define Euler 2.71828182
#define PI 3.141592653
#define RN rand()/(RAND_MAX+1.0)
#define IndexRangeChecking
#define VectorIndexRangeChecking
#define MatrixIndexRangeChecking
#define ComplexVectorIndexRangeChecking
#define ComplexMatrixIndexRangeChecking
// Use the Eigen library as a first trial for comparison and profiling.
#include </Users/cristina/developer/eigen/eigen-eigen-1306d75b4a21/Eigen/Dense>
//#include "nrutil.h"

using namespace std;
using namespace Eigen;

// numerical integration defines
#define FUNC(x) ((*func)(x))
#define EPS 1.0e-6
#define JMAX 20

void mysort(int *idx, double *val, int num);

FILE* out;

class universe{
    
public:
    int MAX_ITERS;
    char fitfunc[100];
    int iter; // global iteration counter
    FILE* plotfile;

    
private:
    int nobjs;
    int wv[40402][MAX_K];      // first dimension needs to be #weight vectors^nobjs
    double dwv[40402][MAX_K];  // first dimension needs to be #weight vectors^nobjs
    double normwv[300][MAX_K]; // first dimension needs to be N+k-1 choose k-1 ($^{N+k-1}C_{k-1}$) , where N is a parameter and k=nobjs
    int N; // parameter relating to the number of divisions
    int nwvs;
    
    double alph; // power in the DTLZ2 and DTLZ4 test functions
    
    int improvements[300];
    int dim;  // dimension of the actual search space, X
    int pdim; // number of parameters in the DACE model
    int titer; // global giving the size of the current correlation matrix R being used
    double **ff; // two-dimensional array of all multiobjective cost vectors
    double **ax;  // two-dimensional array of all x vectors
    double ***pax; // a pointer to the above, or the below
    double **tmpax; // two dimensional array of selected x vectors
    double *xmin; // upper and lower constraints on the search space
    double *xmax;
    double *ay; // array of true/measured y fitnesses
    double *tmpay;
    double **pay;
    
    double minss;  // a minimum standard error of the predictor to force exploration
    
    double glik;
    double altlik;
    bool Debug;
    
    double global_min;
    double gmu;
    double gsigma;
    double **gtheta;
    double **gp;
    double ymin;
    double gymin;
    MatrixXd *pgR;
    MatrixXd *pInvR;
    VectorXd *pgy;
    int best_ever;
    double niche;
    
    double gz[76]; // an array holding the z values for the gaussian distribution
    
    double absmax[MAX_K];
    double absmin[MAX_K];
    double gideal[MAX_K];
    double gwv[MAX_K];
    
    clock_t start, end;
    double cpu_time_used;
    set<int> debugBestParam;
    
public:
    
    
    universe(int x);
    void init_ParEGO();
    void iterate_ParEGO();
private:
    void snake_wv(int s, int k);
    void reverse(int k, int n, int s);
    void init_gz();
    void cwr(int **target, int k, int n);
    MatrixXd identM(int n);
    void pr_sq_mat(MatrixXd& m, int dim);
    void pr_vec(VectorXd& v, int dim);
    double mu_hat(MatrixXd& InvR, VectorXd& y, int n);
    double sigma_squared_hat(MatrixXd& InvR, VectorXd& y, double mu_hat, int n);
    double likelihood(double *param);
    double approx_likelihood(double *param);
    double weighted_distance(double *xi, double *xj, double *theta, double *p, int dim);
    double correlation(double *xi, double *xj, double *theta, double *p, int dim);
    void build_R(double **ax, double *theta, double *p, int dim, int n, MatrixXd& R);
    void build_y(double *ay, int n, VectorXd& y);
    void init_arrays(double ***ax, double **ay, int n, int dim);
    int mypow(int x, int expon);
    double predict_y(double **ax, MatrixXd& R, VectorXd& y, double mu_hat, double *theta, double *p, int n, int dim);
    void get_params(double** param, double *theta, double *p, double *sigma, double *mu);
    double myfit(double *x, double *ff);
    long double posdet(MatrixXd& R, int n);
    double s2(double **ax, double *theta, double *p, double sigma, int dim, int n, MatrixXd& InvR);
    double standard_density(double z);
    double standard_distribution(double z);
    double expected_improvement(double yhat, double ymin, double s);
    double wrap_ei(double *x); // returns -(expected improvement), given the solution x
    void latin_hyp(double **ax, int iter);
    void compute_n(int n, int *dig, int base);
    void set_search();
    void f_kno1(double *x, double *y); // 2 objectives test problem
    void f_vlmop2(double *x, double *y); // 2 objectives test problem
    void f_oka1(double *x, double *y); // 2 objectives test problem
    void f_oka2(double *x, double *y); // 2 objectives test  problem
    void f_dtlz1a(double *x, double *y); // 2 objectives test problem
    void f_dtlz2a(double *x, double *y); // 3 objectives test problem
    void f_dtlz7a(double *x, double *y); // 3 objectives test problem
    void f_vlmop3(double *x, double *y); // 3 objectives test problem
    double Tcheby(double *wv, double *vec, double *ideal);
    int tourn_select(double *xsel, double **ax, double *ay, int iter, int t_size);
    void mutate(double *x);
    void cross(double *child, double *par1, double *par2);
    
};

double static myabs(double v)
{
    if(v>=0.0)
        return v;
    else
        return -v;
}

universe::universe(int x)
{
    alph=1.0;
    minss = 1.0e-9;
    Debug=false;
    best_ever=1;
    niche=1.0;
}

void universe::cross(double *child, double *par1, double *par2)
{
    double xl, xu;
    double x1, x2;
    double alpha;
    double expp;
    double di=20;
    double beta;
    double betaq;
    double rnd;
    
    for(int d=1; d<=dim; d++)
    {
        
        /*Selected Two Parents*/
        
        xl = xmin[d];
        xu = xmax[d];
        
        /* Check whether variable is selected or not*/
        double rn = RN;
        ////fprintf(stderr,"rn1 = %lg\n", rn);
        if(rn <= 0.5)
        {
            //fprintf(stderr,"par1 %lg", par1[d]);
            //fprintf(stderr,"par2 %lg", par2[d]);
            //fprintf(stderr,"abs %.9lg", abs(par1[d] - par2[d]));
            if(myabs(par1[d] - par2[d]) > 0.000001)
            { //fprintf(stderr,"go to bed");
                if(par2[d] > par1[d])
                {
                    x2 = par2[d];
                    x1 = par1[d];
                }
                else
                {
                    x2 = par1[d];
                    x1 = par2[d];
                }
                
                /*Find beta value*/
                
                if((x1 - xl) > (xu - x2))
                {
                    beta = 1 + (2*(xu - x2)/(x2 - x1));
                }
                else
                {
                    beta = 1 + (2*(x1-xl)/(x2-x1));
                    
                }
                
                /*Find alpha*/
                
                expp = di + 1.0;
                
                beta = 1.0/beta;
                
                alpha = 2.0 - pow(beta,expp);
                
                if (alpha < 0.0)
                {
                    printf("ERRRROR %f %f %f\n",alpha,par1[d],par2[d]);
                    exit(-1);
                }
                
                rnd = RN;
                //fprintf(stderr,"rn2 = %lg\n", rnd);
                if (rnd <= 1.0/alpha)
                {
                    alpha = alpha*rnd;
                    expp = 1.0/(di+1.0);
                    betaq = pow(alpha,expp);
                }
                else
                {
                    alpha = alpha*rnd;
                    alpha = 1.0/(2.0-alpha);
                    expp = 1.0/(di+1.0);
                    if (alpha < 0.0)
                    {
                        printf("ERRRORRR \n");
                        exit(-1);
                    }
                    betaq = pow(alpha,expp);
                }
                
                /*Generating one child */
                child[d] = 0.5*((x1+x2) - betaq*(x2-x1));
            }
            else
            {
                
                betaq = 1.0;
                x1 = par1[d]; x2 = par2[d];
                
                /*Generating one child*/
                child[d] = 0.5*((x1+x2) - betaq*(x2-x1));
                
            }
            
            if (child[d] < xl) child[d] = xl;
            if (child[d] > xu) child[d] = xu;
        }
    }
}


int universe::tourn_select(double *xsel, double **ax, double *ay, int iter, int t_size)
{
    int *idx;
    double besty=INFTY;
    //ME: Uninitialised var!!!
    int bestidx = 0;
    cwr(&idx, t_size, iter);                    // fprintf(stderr,"cwr\n");
    for(int i=1;i<=t_size;i++)
    {
        //fprintf(stderr,"idx[i] %d ay[idx[i]]=%lg\n", idx[i], ay[idx[i]]);
        if(ay[idx[i]]<besty)
        {
            besty=ay[idx[i]];
            bestidx=i;
            //fprintf(stderr,"besty %lg bestidx %d\n", besty, bestidx);
        }
    }
    for(int d=1;d<=dim;d++)
    {
        xsel[d] = ax[idx[bestidx]][d];
    }
    //fprintf(stderr,"res %d \n", idx[bestidx]);

    return(idx[bestidx]);
}

void universe::mutate(double *x)
{
    // this mutation always mutates at least one gene (even if the m_rate is set to zero)
    // this is to avoid duplicating points which would cause the R matrix to be singular
    double m_rate=1.0/dim;
    double shift;
    bool mut[dim+1];
    int nmutations=0;
    for(int i=1;i<=dim;i++)
    {
        mut[i]=false;
        double rn = RN;
        ////fprintf(stderr,"rn3 = %lg\n", rn);
        if(rn<m_rate)
        {
            mut[i]=true;
            nmutations++;
        }
    }
    if(nmutations==0)
    {
        double rn = RN;
        ////fprintf(stderr,"rn4 = %lg\n", rn);
        mut[1+(int)(rn*dim)]=true;
    }
    
    
    for(int d=1;d<=dim;d++)
    {
        if(mut[d])
        {
            double rn = RN;
            ////fprintf(stderr,"rn5 = %lg\n", rn);
            shift = (0.0001+rn)*(xmax[d]-xmin[d])/100.0;
            rn = RN;
            ////fprintf(stderr,"rn6 = %lg\n", rn);
            if(rn<0.5)
                shift*=-1;
            
            
            x[d]+=shift;
            
            if(x[d]>xmax[d])
                x[d]=xmax[d]-myabs(shift);
            else if(x[d]<xmin[d])
                x[d]=xmin[d]+myabs(shift);
        }
    }
}


double universe::Tcheby(double *wv, double *vec, double *ideal)
{
    // the augmented Tchebycheff measure with normalization
    int i;
    double sum;
    double diff;
    double d_max=-LARGE;
    double norm[MAX_K];
    double nideal[MAX_K];
    
    sum=0.0;
    
    
    
    for(i=0;i<nobjs;i++)
    {
        norm[i] = (vec[i]-absmin[i])/(absmax[i]-absmin[i]);
        nideal[i] = ideal[i];
        diff = wv[i]*(norm[i]-nideal[i]);
        sum += diff;
        if(diff>d_max)
            d_max = diff;
    }
    
    
    // f//fprintf(stderr,out, "d_max= %.5lf + 0.5 * sum= %.5lf\n", d_max, sum);
    return(d_max + 0.05*sum);
}

void universe::set_search()
{
    /* HERE 1
     To define a new fitness function, first provide information about
     it in the same way as done below for f_vlmop2. Then search for "HERE" again.
     */
    
    MAX_ITERS=250; // default value
    
    if(strcmp(fitfunc,"f_vlmop2")==0) // fitness function name, as it will be given on command line
    {
        nobjs = 2;  // number of objectives in the ff
        dim = 2;    // number of decision variables
        xmin = (double *)calloc((dim+1),sizeof(double));
        xmax = (double *)calloc((dim+1),sizeof(double));
        for(int d=1;d<=dim;d++)
        {
            xmin[d]=-2.0;  // give the minimum and maximum value of every decision variable (notice the array offset)
            xmax[d]=2.0;
        }
        
        // There is no offset in the following arrays
        gideal[0]=0.0;  // set all objective ideal points to zero
        gideal[1]=0.0;
        gwv[0]=0.9;     // set the weight vectors of each objective to anything, so long as it sums to 1
        gwv[1]=0.1;
        absmax[0]=1.0;  // give the absolute values of the maximum and minimum of each objective space dimension. If this is not known, then make sure you overestimate the extent of the space.
        absmin[0]=0.0;
        absmax[1]=1.0;
        absmin[1]=0.0;
    }
    else if(strcmp(fitfunc,"f_dtlz1a")==0)
    {
        nobjs = 2;
        dim = 6;
        xmin = (double *)calloc((dim+1),sizeof(double));
        xmax = (double *)calloc((dim+1),sizeof(double));
        for(int d=1;d<=dim;d++)
        {
            xmin[d]=0;
            xmax[d]=1;
        }
        gideal[0]=0.0;
        gideal[1]=0.0;
        gwv[0]=0.9;
        gwv[1]=0.1;
        absmax[0]=500.0;
        absmin[0]=0.0;
        absmax[1]=500.0;
        absmin[1]=0.0;
    }
    else if(strcmp(fitfunc,"f_dtlz7a")==0)
    {
        nobjs=3;
        dim = 8;
        xmin = (double *)calloc((dim+1),sizeof(double));
        xmax = (double *)calloc((dim+1),sizeof(double));
        for(int d=1;d<=dim;d++)
        {
            xmin[d]=0;
            xmax[d]=1;
        }
        gideal[0]=0.0;
        gideal[1]=0.0;
        gideal[2]=0.0;
        gwv[0]=0.4;
        gwv[1]=0.3;
        gwv[2]=0.3;
        absmax[0]=1.0;
        absmin[0]=0.0;
        absmax[1]=1.0;
        absmin[1]=0.0;
        absmax[2]=30.0;
        absmin[2]=0.0;
    }
    else if((strcmp(fitfunc,"f_dtlz2a")==0)||(strcmp(fitfunc,"f_dtlz4a")==0))
    {
        nobjs=3;
        dim = 8;
        xmin = (double *)calloc((dim+1),sizeof(double));
        xmax = (double *)calloc((dim+1),sizeof(double));
        for(int d=1;d<=dim;d++)
        {
            xmin[d]=0;
            xmax[d]=1;
        }
        gideal[0]=0.0;
        gideal[1]=0.0;
        gideal[2]=0.0;
        gwv[0]=0.4;
        gwv[1]=0.3;
        gwv[2]=0.3;
        absmax[0]=2.5;
        absmin[0]=0.0;
        absmax[1]=2.5;
        absmin[1]=0.0;
        absmax[2]=2.5;
        absmin[2]=0.0;
        if(strcmp(fitfunc,"f_dtlz4a")==0)
            alph=100.0;
    }
    else if(strcmp(fitfunc,"f_vlmop3")==0)
    {
        nobjs=3;
        dim = 2;
        xmin = (double *)calloc((dim+1),sizeof(double));
        xmax = (double *)calloc((dim+1),sizeof(double));
        for(int d=1;d<=dim;d++)
        {
            xmin[d]=-3;
            xmax[d]=3;
        }
        gideal[0]=0.0;
        gideal[1]=0.0;
        gideal[2]=0.0;
        gwv[0]=0.4;
        gwv[1]=0.3;
        gwv[2]=0.3;
        absmax[0]=10.0;
        absmin[0]=0.0;
        absmax[1]=62.0;
        absmin[1]=15.0;
        absmax[2]=0.2;
        absmin[2]=-0.15;
    }
    else if(strcmp(fitfunc,"f_oka1")==0)
    {
        nobjs = 2;
        dim = 2;
        xmin = (double *)calloc((dim+1),sizeof(double));
        xmax = (double *)calloc((dim+1),sizeof(double));
        xmin[1]=6*sin(PI/12.0);
        xmax[1]=xmin[1]+2*PI*cos(PI/12.0);
        xmin[2]=-2*PI*sin(PI/12.0);
        xmax[2]=6*cos(PI/12.0);
        gideal[0]=0.0;
        gideal[1]=0.0;
        gwv[0]=0.9;
        gwv[1]=0.1;
        absmax[0]=8.0;
        absmin[0]=0.0;
        absmax[1]=5.0;
        absmin[1]=0.0;
    }
    else if(strcmp(fitfunc,"f_oka2")==0)
    {
        nobjs = 2;
        dim = 3;
        xmin = (double *)calloc((dim+1),sizeof(double));
        xmax = (double *)calloc((dim+1),sizeof(double));
        xmin[1]=-PI;
        xmax[1]=PI;
        xmin[2]=-5;
        xmax[2]=5;
        xmin[3]=-5;
        xmax[3]=5;
        gideal[0]=0.0;
        gideal[1]=0.0;
        gwv[0]=0.9;
        gwv[1]=0.1;
        absmax[0]=PI;
        absmin[0]=-PI;
        absmax[1]=5.1;
        absmin[1]=0.0;
    }
    else if(strcmp(fitfunc,"f_kno1")==0)
    {
        nobjs = 2;
        dim = 2;
        xmin = (double *)calloc((dim+1),sizeof(double));
        xmax = (double *)calloc((dim+1),sizeof(double));
        for(int d=1;d<=dim;d++)
        {
            xmin[d]=0.0;
            xmax[d]=3.0;
        }
        gideal[0]=0.0;
        gideal[1]=0.0;
        gwv[0]=0.9;
        gwv[1]=0.1;
        absmax[0]=20.0;
        absmin[0]=0.0;
        absmax[1]=20.0;
        absmin[1]=0.0;
    }
    else
    {
        fprintf(stderr,"Didn't recognise that fitness function\n");
        exit(0);
    }
    
    
    pdim=dim*2+2;
}


void universe::compute_n(int n, int *dig, int base)
{
    // given the number n, convert it to base base, and put the digits into dig
    // we already know the number n will be less than pow(base,dim) where dim is global
    int d=dim-1;
    int div = (int)pow((double)base,(double)(dim-1));
    for(int i=0; i < dim; i++)
    {
        dig[i] = n/div;
        n-=dig[i]*div;
        d--;
        div = (int)pow(double(base),double(d));
    }
}

void universe::latin_hyp(double **ax, int iter)
{
    
    //  double **L;
    // bool **viable;
    int v;
    
    /*
     L = (double **)calloc(dim*sizeof(double *));
     for(int d=0; d<dim; d++)
     L[d] = (double *)calloc(iter*sizeof(double));
     viable = (bool **)calloc(dim*sizeof(bool *));
     for(int d=0;d<dim;d++)
     viable[d] = (bool *)calloc(iter*sizeof(bool));
     */
    
    double L[dim][iter];
    bool viable[dim][iter];
    
    
    for(int d=0; d<dim; d++)
    {
        for(int i=0;i<iter;i++)
        {
            viable[d][i]=true;
            
            L[d][i] = xmin[d+1] + i*((xmax[d+1]-xmin[d+1])/double(iter));
        }
    }
    ////fprintf(stderr,"latin:ax\n");
    for(int i=0; i<iter; i++)
    {
        for(int d = 0; d < dim; d++)
        {
            do
            {
                double rn = RN;
                ////fprintf(stderr,"rn7 = %lg\n", rn);
                v = int(rn*iter);
            }
            while(!viable[d][v]);
            viable[d][v]=false;
            double rn = RN;
            ////fprintf(stderr,"rn8 = %lg\n", rn);
            ax[i+1][d+1] = L[d][v]+rn*((xmax[d+1]-xmin[d+1])/double(iter));
            ////fprintf(stderr,"%lg ", ax[i+1][d+1]);
        }
    }
    ////fprintf(stderr,"\n");
}

long double universe::posdet(MatrixXd& R, int n)
{
    // *** NB: this function changes R !!! ***
    // computes the determinant of R. If it is non-positive, then it adjusts R and recomputes.
    // returns the final determinant and changes R.
    long double detR;
    detR = R.determinant();
    double diag=1.01;
    unsigned int nr = 0;
    while(detR<=0.0)
    {
        for(int i=0;i<n;i++)
            R(i,i)=diag;
        detR = R.determinant();
        nr++;
        diag+=0.01;
    }
    if(isnan(detR))
        assert(!isnan(detR));
    return(detR);
}

MatrixXd universe::identM(int n)
{
    MatrixXd I = MatrixXd::Zero(n,n);
    for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
            I(i,j)=0;
    for(int j=1;j<=n;j++)
        I(j,j)=1;
    return I;
}

int universe::mypow(int x, int expon)
{
    return ((int)(pow((double)x, (double)expon)));
}



/* HERE 2
 Below are the current fitness function definitions. Add your functions here.
 The function should take an x and y array and compute the value of the y objectives
 from the x decision variables. Note that the first decision variable is x[1] and t
 he first objective is y[1], i.e. there is an offset to the array. */


void universe::f_dtlz1a(double *x, double *y)
{
    
    
    double g = 0.0;
    for(int i=2;i<=dim;i++)
        g+= (x[i]-0.5)*(x[i]-0.5) - cos(2*PI*(x[i]-0.5)); // Note this is 20*PI in Deb's dtlz1 func
    g += dim-1;
    g *= 100;
    
    
    y[1] = 0.5*x[1]*(1 + g);
    y[2] = 0.5*(1-x[1])*(1 + g);
}

void universe::f_dtlz2a(double *x, double *y)
{
    double g = 0.0;
    for(int i=3;i<=dim;i++)
        g+=(x[i]-0.5)*(x[i]-0.5);
    
    
    y[1] = (1 + g)*cos(pow(x[1],alph)*PI/2)*cos(pow(x[2],alph)*PI/2);
    y[2] = (1 + g)*cos(pow(x[1],alph)*PI/2)*sin(pow(x[2],alph)*PI/2);
    y[3] = (1 + g)*sin(pow(x[1],alph)*PI/2);
}

void universe::f_dtlz7a(double *x, double *y)
{
    double g,h,sum;
    y[1]=x[1];
    y[2]=x[2];
    
    g = 0.0;
    for(int i=3;i<=dim;i++)
    {
        g+=x[i];
    }
    g*=9.0/(dim-nobjs+1);
    g+=1.0;
    
    sum=0.0;
    for(int i=1;i<=nobjs-1;i++)
        sum += ( y[i]/(1.0+g) * (1.0+sin(3*PI*y[i])) );
    h = nobjs - sum;
    
    y[1]=x[1];
    y[2]=x[2];
    y[3]=(1 + g)*h;
}

void universe::f_vlmop3(double *x, double *y)
{
    y[1] = 0.5*(x[1]*x[1]+x[2]*x[2]) + sin(x[1]*x[1]+x[2]*x[2]);
    y[2] = pow(3*x[1]-2*x[2]+4.0, 2.0)/8.0 + pow(x[1]-x[2]+1, 2.0)/27.0 + 15.0;
    y[3] = 1.0 / (x[1]*x[1]+x[2]*x[2]+1.0) - 1.1*exp(-(x[1]*x[1]) - (x[2]*x[2]));
}


void universe::f_oka1(double *x, double *y)
{
    double x1p = cos(PI/12.0)*x[1] - sin(PI/12.0)*x[2];
    double x2p = sin(PI/12.0)*x[1] + cos(PI/12.0)*x[2];
    
    y[1] = x1p;
    y[2] = sqrt(2*PI) - sqrt(myabs(x1p)) + 2 * pow(myabs(x2p-3*cos(x1p)-3) ,0.33333333);
}

void universe::f_oka2(double *x, double *y)
{
    
    y[1]=x[1];
    y[2]=1 - (1/(4*PI*PI))*pow(x[1]+PI,2) + pow(myabs(x[2]-5*cos(x[1])),0.333333333) + pow(myabs(x[3] - 5*sin(x[1])),0.33333333);
}

void universe::f_vlmop2(double *x, double *y)
{
    
    double sum1=0;
    double sum2=0;
    
    for(int i=1;i<=2;i++)
    {
        sum1+=pow(x[i]-(1/sqrt(2.0)),2);
        sum2+=pow(x[i]+(1/sqrt(2.0)),2);
    }
    
    y[1] = 1 - exp(-sum1);
    y[2] = 1 - exp(-sum2);
}

void universe::f_kno1(double *x, double *y)
{
    double f;
    double g;
    double c;
    
    
    c = x[1]+x[2];
    
    f = 20-( 11+3*sin((5*c)*(0.5*c)) + 3*sin(4*c) + 5 *sin(2*c+2));
    //  f = 20*(1-(abs(c-3.0)/3.0));
    
    g = (PI/2.0)*(x[1]-x[2]+3.0)/6.0;
    
    y[1]= 20-(f*cos(g));
    y[2]= 20-(f*sin(g));
    
}


double universe::myfit(double *x, double *ff)
{
    /* HERE 3
     
     Here is where the fitness function is called. Just add in a call for your function here,
     following this format.
     
     */
    
    if(strcmp(fitfunc, "f_kno1")==0)
    {
        f_kno1(x, ff);
        return(Tcheby(gwv,&(ff[1]),gideal));
    }
    else if(strcmp(fitfunc, "f_vlmop2")==0)
    {
        f_vlmop2(x, ff);
        return(Tcheby(gwv,&(ff[1]),gideal));
    }
    else if(strcmp(fitfunc, "f_vlmop3")==0)
    {
        double T;
        f_vlmop3(x, ff);
        T=(Tcheby(gwv,&(ff[1]),gideal));
        return(T);
    }
    else if(strcmp(fitfunc, "f_dtlz1a")==0)
    {
        double T;
        f_dtlz1a(x, ff);
        T=(Tcheby(gwv,&(ff[1]),gideal));
        return(T);
    }
    else if(strcmp(fitfunc, "f_dtlz2a")==0)
    {
        double T;
        f_dtlz2a(x, ff);
        T=(Tcheby(gwv,&(ff[1]),gideal));
        return(T);
    }
    else if(strcmp(fitfunc, "f_dtlz4a")==0)
    {
        double T;
        f_dtlz2a(x, ff); // this is called but with global variable alph set = 100.0
        T=(Tcheby(gwv,&(ff[1]),gideal));
        return(T);
    }
    else if(strcmp(fitfunc, "f_dtlz7a")==0)
    {
        double T;
        f_dtlz7a(x, ff);
        T=(Tcheby(gwv,&(ff[1]),gideal));
        return(T);
    }
    
    else if(strcmp(fitfunc, "f_oka1")==0)
    {
        double T;
        f_oka1(x, ff);
        T=(Tcheby(gwv,&(ff[1]),gideal));
        return(T);
    }
    else if(strcmp(fitfunc, "f_oka2")==0)
    {
        f_oka2(x, ff);
        return(Tcheby(gwv,&(ff[1]),gideal));
    }
    else
    {
        fprintf(stderr, "Didn't recognise that fitness function.\n");
        exit(0);
    }
}


void universe::init_arrays(double ***ax, double **ay, int n, int dim)
{
    // using a 1 offset to make things easier for use with NR routines and matrix routines
    *ax = (double **)calloc((n+1),sizeof(double *));
    for(int i=0;i<n+1;i++)
        (*ax)[i]=(double *)calloc((dim+1),sizeof(double));
    (*ay)=(double *)calloc((n+1),sizeof(double));
    
}

void universe::get_params(double** param, double *theta, double *p, double *sigma, double *mu)
{
    // uses global dim value
    for (int i=1;i <= dim; i++)
    {
        assert(debugBestParam.find(i)!=debugBestParam.end());
        theta[i]=param[1][i];
        assert(debugBestParam.find(i+dim)!=debugBestParam.end());
        p[i]=param[1][i+dim];
    }
    assert(debugBestParam.find(2*dim+1)!=debugBestParam.end());
    *sigma=param[1][2*dim+1];
    assert(debugBestParam.find(2*dim+2)!=debugBestParam.end());
    *mu=param[1][2*dim+2];
    
}

void universe::cwr(int **target, int k, int n)  // choose without replacement k items from n
{
    int i,j;
    int l_t; //(length of the list at time t)
    
    if(k>n)
    {
        fprintf(out,"trying to choose k items without replacement from n but k > n!!\n");
        exit(-1);
    }
    
    (*target) = (int *)calloc((k+1),sizeof(int));
    int *to;
    to = &((*target)[1]);
    
    int from[n];
    
    for(i=0;i<n;i++)
        from[i]=i+1;
    
    l_t = n;
    for(i=0;i<k;i++)
    {
        double rn = RN;
        ////////fprintf(stderr,"rn9 = %lg\n", rn);
        j=(int)(rn*l_t);
        to[i]=from[j];
        from[j]=from[l_t-1];
        l_t--;
    }
}



double universe::wrap_ei(double *x)
{
    ////fprintf(stderr,"WRAP_BEGIN\n");
    for(int d=1;d<=dim; d++)
    {    (*pax)[titer+1][d] = x[d];
        //fprintf(stderr,stdout,"%.5lf %.5lf\n", x[d], (*pax)[titer+1][d]);
    }
    double fit;
    // predict the fitness
    fit=predict_y(*pax, *pInvR, *pgy, gmu, *gtheta, *gp, titer, dim);
    
    //fprintf(stderr,"predicted fitness in wrap_ei() = %lg\n", fit);
    
    
    // compute the error
    double ss;
    //ME: ss -least square error.
    ss=s2(*pax, *gtheta, *gp, gsigma, dim, titer, *pInvR);
     //fprintf(stderr,"s^2 error in wrap_ei() = %lg\n", ss);
     // //fprintf(stderr,"%.9lf %.9lf \n", x[1], ss);
    
    
    // compute the expected improvement
    double ei;
    ei=expected_improvement(fit, gymin, sqrt(ss));
    //fprintf(stderr,,"-ei in wrap_ei() = %lg\n", -ei);
    
    
    for(int d=1; d <= dim; d++)
    {
        if((x[d]>xmax[d])||(x[d]<xmin[d]))
            ei=-1000;
    }
    
    //fprintf(stderr,"%.9lf\n", ei);
    //printf("WRAP__END\n");
    
    // return the expected improvement
    return(-ei);
}


int main(int argc, char **argv)
{
    clock_t time1, time2;
    time1 = clock();
    //out = fopen("outputEigen.txt", "w");
    universe U(1);
    
    unsigned int seed = 47456536;
    //printf("seed: %d\n", seed);

    //if(argc>1)
    //seed = atoi(argv[1]);
    srand(47456536);
    
    sprintf(U.fitfunc, "f_dtlz2a");
    //printf("function_name: %s\n", U.fitfunc);

    //if(argc>2)
    // sprintf(U.fitfunc, argv[2]);
    
    //U.plotfile = fopen("plotdtlz4a.txt", "w");
    
    U.init_ParEGO(); // starts up ParEGO and does the latin hypercube, outputting these to a file
    
    int i = U.iter;
    while ( i < U.MAX_ITERS )
    {
        U.iterate_ParEGO(); // takes n solution/point pairs as input and gives 1 new solution as output
        i++;
    }
    
    fclose(U.plotfile);
    fclose(out);
    time2 = clock();
    float diff1 ((float)time2-(float)time1);
    float seconds1 = diff1 / CLOCKS_PER_SEC;
    cout<<seconds1<<endl;
    
}



void universe::iterate_ParEGO()
{
    int prior_it=0;
    int stopcounter=0;
    
    //double lik[pdim+2]; // array of [1..pdim+1] likelihoods for downhill simplex
    //  lik = (double *)calloc((pdim+2)*sizeof(double));
    
    //  double param[pdim+2][pdim+1];
    double **param;
    param = (double **)calloc((pdim+2),sizeof(double *));
    for(int i=0;i<=pdim+1;i++)
        param[i]=(double *)calloc((pdim+1),sizeof(double));
    
    
    
    //double bestparam[pdim+2][pdim+1];
    double **bestparam;
    bestparam = (double **)calloc((pdim+2),sizeof(double *));
    for(int i=0;i<=pdim+1;i++)
        bestparam[i]=(double *)calloc((pdim+1),sizeof(double));
    //vector<double> bestparam(pdim+1);
    
    double *p_last;
    double *theta_last;
    p_last=(double *)calloc((dim+1),sizeof(double));
    theta_last=(double *)calloc((dim+1),sizeof(double));
    double sigma_last;
    //printf("mem val of sigma %f\n", sigma_last);
    double mu_last;
    
    
    
    //double tryx[dim+2][dim+1];
    // double **tryx;
    // tryx = (double **)calloc((dim+2)*sizeof(double *));  // solutions tried in the search over the modelled search landscape
    // for(int i=0;i<dim+2;i++)
    //  tryx[i]=(double *)calloc((dim+1)*sizeof(double));
    
    //double expi[dim+2];
    //  double *expi;
    // expi = (double *)calloc((dim+2)*sizeof(double)); // expected improvement in the fitness
    
    int gapopsize=20;
    
    // double popx[gapopsize+1][dim+1];
    //matrix
    double **popx;
    popx = (double **)calloc((gapopsize+1),sizeof(double *));  // GA population for searching over modelled landscape
    for(int i=0;i<gapopsize+1;i++)
        popx[i]=(double *)calloc((dim+1),sizeof(double));
    
    //double popy[gapopsize+1];
      double *popy;
    popy = (double *)calloc((gapopsize+1),sizeof(double));
    
    //double mutx[dim+1];
    double *mutx;
    mutx =(double *)calloc((dim+1),sizeof(double));
    
    double muty;
    
    
    
    //double fit; // predicted fitness value
    //double mh; // mu hat
    //double sig; // sigma hat squared
    
    
    
    //  double recomp_prob=1.0;
    static bool change=true;
    static int next=-1;
    static int add=1;
    
    if(iter%5==2)
        change = true;  // change the weight vector
    
    if(change)
    {
        if((next==9)||((next==0)&&(add==-1)))
            add*=-1;
        next+=add;
        for(int k=0;k<nobjs;k++)
        {
            gwv[k]=normwv[next][k];
        }
    }
    
          ////fprintf(stderr,, "%.2lf %.2lf weightvectors %d %d\n", gwv[0], gwv[1], next, add);
    //printf("ay[i]\n");
    for(int i=1;i<=iter;i++)
    {
        ay[i] = Tcheby(gwv,&(ff[i][1]),gideal);
        //printf("%lg ", ay[i]);
        if(ay[i]<ymin)
        {
            ymin=ay[i];
            best_ever=i;
        }
    }
//    printf("\n ymin: %lg\n", ymin);
//    printf("best_ever: %d\n", best_ever);
    
    
    if(iter>11*dim+24)
    {
        titer =11*dim+24;
        
        int ind[iter+1];
        mysort(ind, ay, iter);
        for (int i=1; i<=titer/2; i++)
        {
            for(int d=1;d<=dim;d++)
            {
                tmpax[i][d] = ax[ind[i]][d];
                //printf("%lg", tmpax[i][d]);
            }
            tmpay[i] = ay[ind[i]];                   // fprintf(stderr, "evaluate\n");
            //printf("\n tmpay: %lg\n", tmpay[i]);

            //find the best fitness out of the selected initial solutions
            //	  if(ay[parA]<best_imp)
            //  best_imp=ay[parA];
        }
        
        
        int *choose;
        cwr(&choose,iter-titer/2,iter-titer/2); // cwr(k,n) - choose without replacement k items from n
        
        for(int i=titer/2+1;i<=titer;i++)
        {
            assert(choose[i-titer/2]+titer/2 <= iter && 0 < choose[i-titer/2]+titer/2);
            int j= ind[choose[i-titer/2]+titer/2];
            for(int d=1;d <=dim;d++)
                tmpax[i][d] = ax[j][d];
            tmpay[i] = ay[j];
        }
        
        pax=&tmpax;
        pay=&tmpay;
        free(choose);
    }
    else
    {
        titer=iter;
        pax=&ax;
        pay=&ay;
    }
    
    
    
    VectorXd y(titer);
    build_y(*pay, titer,y);
    //cout<<"y:"<<y;
    // pr_vec(y, titer);
    
    double best_lik=INFTY;
    int besti;
    // ME: changed from change=1
    if(change=1)
    {
        
        for(int i=0;i<30;i++)
        {
            for(int d=1; d<=dim; d++)
            {
                double rn1 = RN;
                ////printf("rn10 = %lg\n", rn1);
                double rn2 = RN;
                ////printf("rn11 = %lg\n", rn2);
                theta_last[d]=1+rn1*2;
                p_last[d]=1.01+rn2*0.98;
                //printf("rn1=%lg rn2=%lg theta_last[d]=%lg p_last[d]=%lg\n", rn1, rn2, theta_last[d], p_last[d]);

            }
            //ME: Carefull with the indexes!!! It starts from 1.
            MatrixXd R = MatrixXd::Zero(titer, titer);
            build_R(*pax, theta_last, p_last, dim, titer, R);
            //  fprintf(out,"NEW STARTING R Matrix\n");
            //	  pr_sq_mat(R,iter);
            
            double detR=posdet(R,titer);
            if (detR == 0.0)
            {
                printf("det is null");
                break;
            }
            //printf("det = %.5lf\n", R.determinant());
            // pr_sq_mat(R,titer);
            //cout<< "THe MATRIX:"<<R<<"\n";
            MatrixXd InvR = R.inverse();
            //cout<<"MATRIX INV IN CHANGE" <<InvR<<"\n";
            mu_last=mu_hat(InvR, y, titer);
            //fprintf(stderr,,"OK - mu %lg calculated\n", mu_last);
            sigma_last=sigma_squared_hat(InvR, y, mu_last, titer);
            //fprintf(stderr,,"OK - sigma %lg calculated\n", sigma_last);
            for(int j=1;j<=dim;j++)
            {
                param[1][j]=theta_last[j];
                param[1][j+dim]=p_last[j];
            }
            param[1][2*dim+1]=sigma_last;
            param[1][2*dim+2]=mu_last;
            
            for (int i = 1; i <= 2*dim+2; i++)
                //fprintf(stderr,, "%lg ", param[1][i]);
            //fprintf(stderr,, "\n");
//
            glik=likelihood(param[1]);
            //fprintf(stderr,"glik= %lg best_lik= %lg  BETTER....\n", glik, best_lik);

            if(glik<best_lik)
            {
                //printf("bestparam:\n");

                besti = i;
                best_lik=glik;
                for(int j=1;j<=pdim;j++)
                {
                    bestparam[1][j]=param[1][j];
                    debugBestParam.insert(j);
                    //printf("%lg ", bestparam[1][j]);

                }
                //printf("\n");

            }
            change=false;
            
            //ME: Free the memory from Eigen. Check if it is automatically freed.
            //~y;
            //~R;
            //~InvR;
        }
    }
    //printf(" theta_last[d]=%lg %lg p_last[d]=%lg %lg\n", theta_last[1], theta_last[2], p_last[1], p_last[2]);
    
    //fprintf(out, "last_sigma: %f\n", sigma_last);

    get_params(bestparam, theta_last, p_last, &sigma_last, &mu_last);
    
    //printf(" theta_last[d]=%lg %lg p_last[d]=%lg %lg\n", theta_last[1], theta_last[2], p_last[1], p_last[2]);

//    	//fprintf(stderr,,"FINAL DACE parameters = \n");
//    	for(int d=1; d<=dim; d++)
//    	//fprintf(stderr,,"%lg ", theta_last[d]);
//    	for(int d=1; d<=dim; d++)
//    	//fprintf(stderr,,"%lg ", p_last[d]);
//    	//fprintf(stderr,," %lg %lg\n", bestparam[1][2*dim+1], bestparam[1][2*dim+2]);
//    
//    
    /* Use the full R matrix */
    titer=iter;
    pax = &ax;
    //ME: Carefull with the indexes!!! It starts from 1.
    MatrixXd Rpred = MatrixXd::Zero(titer,titer);
    build_R(*pax, theta_last, p_last, dim, titer, Rpred);
    posdet(Rpred, titer);
    //scout<<Rpred;
    if (Rpred.determinant() == 0.0)
    {
        printf("det is null");
    }
    long double det =  Rpred.determinant();
    if(isnan(det))
        printf("is indeed nan");
    //printf("det = %lg\n",det);
    MatrixXd InvR = Rpred.inverse();
    //cout<<"MATRIX INV" <<InvR<<"\n";
    VectorXd fy(titer);
    build_y(ay, titer, fy);
    /* ***************************** */
    
    //   fprintf(out,"predicted R matrix built OK:\n");
    //   pr_sq_mat(Rpred,titer);
    
    
    double best_imp=INFTY;
    double best_x[dim+1];
    //double best_y=INFTY;
    
    
    // set the global variables equal to the local ones
    gmu = mu_last;
    gsigma = sigma_last;
    //printf("gsigma %f\n", gsigma);
    gtheta = &theta_last;
    gp = &p_last;
    pgR=&Rpred;
    pInvR=&InvR;
    pgy=&fy;
    
    //printf(" gtheta[d]=%lg %lg p_last[d]=%lg %lg\n", gtheta[1], gtheta[2], gp[1], gp[2]);

    
    
    gymin = INFTY;
    
    for(int i = 1;i < titer;i++)
        if((*pgy)[i]<gymin)
            gymin=(*pgy)[i];
    
    //printf("gymin %lg\n", gymin);
    
    start = clock();
    
    // BEGIN GA code
    best_imp=INFTY;
    int parA;
    int parB;
    
    
    // initialize using the latin hypercube method
    if(iter == 48)
        cout<<"bla";
    latin_hyp(popx, gapopsize);
//    printf("popx ");
//    for(int i = 1; i < gapopsize + 1; i++)
//        for (int j = 1; j < dim+1; j++)
//        {
//            printf("%lg ", popx[i][j]);
//            printf("\n");
//        }

   //fprintf(stderr,"popy ");
    for (int i=1; i<=gapopsize; i++)
    {
        popy[i] = wrap_ei(popx[i]);
        //fprintf(stderr,"%lg ", popy[i]);
    }
    //printf("POPY_END\n");
    
    // initialize with mutants of nearby good solutions
    
    int ind[iter+1];
    mysort(ind, ay, iter);
    
    //   for(int i=1;i<=titer;i++)
    //	printf("ay = %.5lf ; %.5lf\n", ay[ind[i]], ay[i]);
    
    //printf("Begin_MUTATION");
    for (int i=1; i<=5; i++)
    {
        //	  printf("parent = %.5lf\n", ay[ind[i]]);
        parA=ind[i];
        for(int d=1;d<=dim;d++)
            popx[i][d] = ax[parA][d];
        mutate(popx[i]);// fprintf(stderr, "mutate\n");
        //for(int j=1; j < dim+1 ;j++)
         //   printf("%lg ", popx[i][j]);
        popy[i] = wrap_ei(popx[i]);                   // fprintf(stderr, "evaluate\n");
        
    }
    //printf("End_MUTATION");

    double p_cross=0.2;
    for (int i=1; i<=10000; i++)
    {
//        printf("GA iter %d ", i);
//        printf("popx ");
//        for(int i = 1; i < gapopsize + 1; i++)
//            for (int j = 1; j < dim+1; j++)
//            {
//                printf("%lg ", popx[i][j]);
//                printf("\n");
//            }
        double rn = RN;
        //printf("rn12 = %lg\n", rn);
        if(rn < p_cross)
        {
            if(0.0235743<=rn && rn <= 0.0235744){
                cerr << "bla\n";
                //cerr << iter <<"\n";
                assert(rn != 0.127397);
            }
            parA=tourn_select(mutx, popx, popy, gapopsize, 2);
            do
            {
//                printf("TOURN");
//                for (int i=1; i<=gapopsize; i++)
//                {
//                    printf("%lg ", popy[i]);
//                }
//                printf("TOURN_END\n");
                parB=tourn_select(mutx, popx, popy, gapopsize, 2);
            }
            while(parB==parA);
            cross(mutx, popx[parA], popx[parB]);
        }
        else
            parA=tourn_select(mutx, popx, popy, gapopsize, 2);//  fprintf(stderr, "parent selected\n");
        mutate(mutx);                             // fprintf(stderr, "mutate\n");
        muty = wrap_ei(mutx);                      //  fprintf(stderr, "evaluate\n");
        if(muty<popy[parA])
        {
            for(int d=1;d<=dim;d++)
                popx[parA][d]=mutx[d];
            popy[parA]=muty;
        }
    }
    
    bool improved=false;
    for(int i=1;i<=gapopsize; i++)
    {
        if(popy[i]<best_imp)
        {
            improved=true;
            best_imp=popy[i];
            for(int d=1; d<=dim; d++)
                best_x[d]=popx[i][d];
        }
    }
    printf("ei= %lf\n", best_imp);
    if(improved==false)
    {
        fprintf(stderr, "GA found no improvement\n");
        for(int d=1; d<=dim; d++)
        {
            best_x[d]=popx[1][d];
        }
        mutate(best_x);
    }
    // END GA code
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // timestr << iter << " " << cpu_time_used << endl;
    
//    printf("popx ");
//    for(int i = 1; i < gapopsize + 1; i++)
//        for (int j = 1; j < dim+1; j++)
//        {
//            printf("%lg ", popx[i][j]);
//            printf("\n");
//        }
//    
//
//    printf("ax\n");
    for(int d=1;d<=dim;d++)
    {
        ax[iter+1][d]=best_x[d];
        //printf("%lg ", ax[iter+1][d]);
    }
    //printf("\n");
    ay[iter+1]=myfit(ax[iter+1], ff[iter+1]);
    
    
    fprintf(stdout, "%d ", iter+1+prior_it);
    for(int d=1;d <=dim; d++)
        printf("%lg ", ax[iter+1][d]);
    fprintf(stdout, "decision\n");
    
    fprintf(stdout, "%d ", iter+1);
    for(int i=1;i<=nobjs;i++)
    {
        fprintf(stdout, "%lg ", ff[iter+1][i]);
        //fprintf(plotfile, "%.5lf ", ff[iter+1][i]);
    }
    //fprintf(plotfile, "\n");
    fprintf(stdout, "objective\n");
    
    //cout<<"ymin"<<ymin<<"\n";
    improvements[iter+1]=improvements[iter];
    if (ay[iter+1]>=ymin)
    {
        //fprintf(stderr,"No actual improver found\n");
        stopcounter++;
    }
    
    else
    {
        improvements[iter+1]++;
        ymin = ay[iter+1];
        stopcounter=0;
        best_ever=iter+1;
    }
    
    //ME: Free the memory from Eigen. Check if it is automatically freed.
    //~InvR;
    //~Rpred;
    //~fy;
    iter++;
    
    free(p_last);
    free(theta_last);
    
    for(int i=0;i<gapopsize+1;i++)
        free(popx[i]);
    free(popx);
    
    for(int i=0;i<=pdim+1;i++)
        free(param[i]);
    free(param);
    
    
    
    
}


void universe::init_ParEGO()
{
    
    init_gz();
    
    set_search();
    
    
    if(nobjs==2)
        N=10;  // gives 11 weight vectors  - the formula is: number of weight vectors = (N+k-1)!/N!(k-1)! where k is number of objectives
    else if(nobjs==3)
        N=4;  // gives 15 weight vectors
    else if(nobjs==4)
        N=3;   // gives 20 weight vectors
    
    snake_wv(N,nobjs);  // function to create evenly spaced normalized weight vectors
    
    
    
    init_arrays(&ax, &ay, MAX_ITERS, dim);
    init_arrays(&tmpax, &tmpay, MAX_ITERS, dim);
    
    ff = (double **)calloc((MAX_ITERS+2),sizeof(double *));
    for(int i=0;i<=MAX_ITERS+1;i++)
        ff[i]=(double *)calloc((dim+1),sizeof(double));
    
    int prior_it=0;
    
    iter=10*dim+(dim-1);
    do
    {
        for(int i=1; i<=iter;i++)
            improvements[i]=0;
        ymin=INFTY;
        
        
        latin_hyp(ax, iter);  // select the first solutions using the latin hypercube
        
        
        
        for(int i=1;i<=iter;i++)
        {
            ay[i]=myfit(ax[i],ff[i]);
            printf( "%d ", i+prior_it);
            for(int d=1;d<=dim;d++)
                printf( "%lf ", ax[i][d]);
            printf("decision\n");
            printf("%d ", i+prior_it);
            for(int k=1;k<=nobjs;k++)
                printf("%lg ", ff[i][k]);
            printf("objective\n");
        }
    }while(0);
    
}




double universe::expected_improvement(double yhat, double ymin, double s)
{
    double E;
    double sdis;
    double sden;
    if(s<=0)
        return 0;
    //printf("expression: %lg", (ymin-yhat)/s);
    if((ymin-yhat)/s < -7.0)
        sdis = 0.0;
    else if((ymin-yhat)/s > 7.0)
        sdis = 1.0;
    else
        sdis = standard_distribution( (ymin-yhat)/s );
    
    //printf("sdis: %lg\n", sdis);
    sden = standard_density((ymin-yhat)/s);
    //printf("sdis: %lg\n", sden);
    
    E = (ymin - yhat)*sdis + s*sden;
    return E;
}

double universe::standard_density(double z)
{
    double psi;
    
    psi = (1/sqrt(2*PI))*exp(-(z*z)/2.0);
    return (psi);
    
}

/*
 double standard_distribution(double z)
 {
 double phi;
 if(z>0.0)
 phi=0.5+qsimp(&standard_density, 0.0, z);
 else if(z<0.0)
 phi=0.5-qsimp(&standard_density, 0.0, -z);
 else
 phi=0.5;
 return(phi);
 }
 */

double universe::predict_y(double **ax, MatrixXd& InvR, VectorXd& y, double mu_hat, double *theta, double *p, int n, int dim)
{
    double y_hat;
    //  Matrix InvR = R.Inverse();
    VectorXd one(n);
    for(int i=0;i<n;i++)
        one(i)=1;
    
    VectorXd r(n);
    for(int i=0;i<n;i++)
    {
        r(i) = correlation(ax[n+1],ax[i+1],theta,p,dim);
        //fprintf(stdout,"r[%di]=%.5lf ax[n+1][1]=%.5lf\n",i, r(i), ax[n+1][1]);
    }
    // fprintf(out,"\n");
    
    
    //ME: unsupported operation of Eigen to add a scalar coefficient-wise????
    //????????????????????
    //y_hat = mu_hat + r * InvR * (y - one*mu_hat);
    //double intermidiate = (();
    //cout<< "intermidiate"<< intermidiate<<"\n";(double)((r.transpose()*InvR)*(y-(one*mu_hat)));
    y_hat = mu_hat + ((r.transpose()*InvR)*(y-(one*mu_hat)));
    
    //fprintf(stderr,"y_hat=%f mu_hat=%f\n",y_hat, mu_hat);
    
    /*
     if((y_hat>100)||(y_hat<-100))
     {
     //      fprintf(out,"mu_hat=%.5lf theta=%.5lf p=%.5lf\n", mu_hat, theta[1], p[1]);
     for(int i=1;i<=n;i++)
     fprintf(out,"%.2f-%.2f log(r[i])=%le ", ax[n+1][1],ax[i][1] , log(r[i]));
     }
     */
    
    return(y_hat);
    
}

double universe::weighted_distance(double *xi, double *xj, double *theta, double *p, int dim)
{
    double sum=0.0;
    
    double nxi[dim+1];
    double nxj[dim+1];
    
    
    
    for(int h=1;h<=dim;h++)
    {
        nxi[h] = (xi[h]-xmin[h])/(xmax[h]-xmin[h]);
        nxj[h] = (xj[h]-xmin[h])/(xmax[h]-xmin[h]);
        //if(iter == 48)
            //printf("theta= %lg p= %lg\n", theta[h], p[h]);
        sum += theta[h]*pow(myabs(nxi[h]-nxj[h]),p[h]);
        //      sum += 4.0*pow(myabs(xi[h]-xj[h]),2.0);     // theta=4 and p=2
        
    }
    return(sum);
}

double universe::correlation(double *xi, double *xj, double *theta, double *p, int dim)
{
    
        //for(int d=1;d<=dim;d++)
            //fprintf(stdout, "CORRELATION: %.5lf %.5lf %.5lf %.5lf\n", xi[d],xj[d],theta[d],p[d]);
    return exp(-weighted_distance(xi,xj,theta,p,dim));
}

void universe::build_y(double *ay, int n, VectorXd& y)
{
    for(int i=0;i<n;i++)
        y(i)=ay[i+1];
}

void universe::build_R(double **ax, double *theta, double *p, int dim, int n, MatrixXd& R)
{
    // takes the array of x vectors, theta, and p, and returns the correlation matrix R.
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            if(Debug)
                fprintf(out,"%.5lf %.5lf %.5lf  %.5lf %d\n", ax[i][1],ax[j][1],theta[1], p[1], dim);
            //ME : Eigen way of accessing elements.
            R(i,j)=correlation(ax[i+1],ax[j+1], theta, p, dim);
            if(Debug)
                fprintf(out,"%.5lf\n", R(i,j));
        }
        if(Debug)
            fprintf(out,"\n");
    }
    //printf("FINISHED BUILDING R");
    
}

double universe::s2(double **ax, double *theta, double *p, double sigma, int dim, int n, MatrixXd& InvR)
{
    //if(iter == 80)
    //printf("sigma: %Lf\n", sigma);
    double s2;
    //  Matrix InvR = R.Inverse();
    VectorXd one(n);
    for(int i=0;i<n;i++)
        one(i)=1;
    
    VectorXd r(n);
    for(int i=0;i<n;i++)
    {
        //fprintf(stdout,"theta=%.5lf p=%.5lf ax[n+1]=%.5lf, ax[i+1]=%.5lf\n", theta[i],p[i],ax[n+1][i],ax[i+1][i]);
        r(i) = correlation(ax[n+1],ax[i+1],theta,p,dim);
        //fprintf(stdout,"r[i]=%lg \n",r(i));
    }
    
    //ME: Not sure still if that is dot product?????
    //???????????
    //s2 = sigma * (1 - r*InvR*r + pow((1-one*InvR*r),2)/(one*InvR*one) );
    //s2 = sigma * (1 - r*InvR*r + pow((1-one*InvR*r),2)/(one*InvR*one) );
    s2 = sigma * (1 - r.transpose()*InvR*r +
                  (pow((1-one.transpose()*InvR*r),2)/(one.transpose()*InvR*one)));

//    double f1 = (r.transpose()*InvR*r);
//    printf("f1= %lg\n", f1);
//    double f2 = 1 - r.transpose()*InvR*r;
//    printf("f2= %lg\n", f2);
//    double f3 = 1-one.transpose()*InvR*r;
//    printf("f3= %lg\n", f3);
//    double f4 = pow((1-one.transpose()*InvR*r),2);
//    printf("f4= %lg\n", f4);
//    double f5 = pow((1-one.transpose()*InvR*r),2)/(one.transpose()*InvR*one);
//    printf("f5= %lg\n", f5);
//    
//    printf("s2= %lg\n", s2);

    

    
//    double f2 = (one.transpose()*InvR*r);
//    double f4  = 1.0000000000000 - f2;
//    double f3 = (one.transpose()*InvR*one);
//    double f5 = pow(f4,2)/f3;
//    double f6 = abs(1.0000-f1);
//    double f7 = f6 + f5;
//    //printf("%f %f %f %f %f %f %f\n", f1, f2, f4, f3, f5, f6, f7);
//    double aa = one.transpose()*InvR*r;
//    double a1 = 1-one.transpose()*InvR*r;
//    double f1 = pow((1-one.transpose()*InvR*r),2)/(one.transpose()*InvR*one);
//    double f2 = r.transpose()*InvR*r;
//    double f3 = 1-f2+f1;
//    s2 = sigma * f3;
//    printf("iter: %d\n", iter);
//    printf("aa %lg\n", aa);
//    printf("a1 %lg\n", a1);
//        //printf("sigma %f\n", sigma);
//      printf("f1 %lg\n", f1);
//      printf("f2 %lg\n", f2);
//        printf("f3 %lg\n", f3);
//   
    //ME: Free the memory from Eigen. Check if it is automatically freed.
    //~one;
    //~r;
    assert(s2 >= 0);    return(s2);
}

double universe::sigma_squared_hat(MatrixXd& InvR, VectorXd& y, double mu_hat, int n)
{
    double numerator, denominator;
    
    VectorXd one(n);
    for(int i=0;i<n;i++)
        one(i)=1;
    
    VectorXd vmu=one*mu_hat;
    //cout<<"GMU:"<<mu_hat<<"\n";
    //cout<<vmu;
    VectorXd diff = y - vmu;
    //cout<<diff;
    //ME: Not sure still if that is dot product?????
    //???????????
    //numerator = diff*InvR*diff;
    numerator = diff.transpose()*InvR*diff;
    denominator = n;
    //printf("sigma: %f\n", numerator);
    return (numerator/denominator);
}

double universe::approx_likelihood(double *param)
{
    // uses global variable storing the size of R: iter
    // uses global variable storing the dimension of the search space: dim
    // uses global ax and ay values
    
    // constraint handling
    // constraint handling
    double sum=0.0;
    double ypred;
    for(int j=1;j<pdim;j++)
    {
        if(param[j]<0.0)
            sum+=param[j];
        
    }
    if(sum<0)
        return(-sum + 1e+80);
    sum=0.0;
    bool outofrange=false;
    for(int j=1;j<=dim;j++)
    {
        if(param[dim+j]>=2.0)
        {
            sum+=param[dim+j]-2.0;
            outofrange=true;
        }
        else if(param[dim+j]<1.0)
        {
            sum+=-(param[dim+j]-1.0);
            outofrange=true;
        }
    }
    if(outofrange)
        return(sum + 1e+80);
    
    int p,q;
    
    //double coefficient;
    //double exponent;
    double **tmpx;
    
    tmpx = (double **) calloc((titer+1),sizeof(double *));
    for(int i=0;i<=titer;i++)
        tmpx[i]=(double *)calloc((dim+1),sizeof(double));
    
    double mu=param[2*dim+2];
    //double sigma=param[2*dim+1];
    //double *theta=&param[0];
    //double *pp=&param[dim];
    
    
    MatrixXd R = MatrixXd::Zero(titer,titer);
    build_R(*pax, &param[0], &param[dim], dim, titer, R);
    
    
    VectorXd y = VectorXd::Zero(titer);
    build_y(*pay, titer, y);
    
    
    //  Debug=false;
    //  pr_sq_mat(likR,iter);
    // fprintf(out,"\n\n");
    sum=0.0;
    for(int i=0;i<titer;i++)
    {
        VectorXd tmpy(titer-1);
        MatrixXd Rr = MatrixXd::Zero(titer-1,titer-1);
        for(int j=0;j<titer;j++)
        {
            for(int k=0;k<titer;k++)
            {
                p = 0;
                q = 0;
                if(j<i)
                    p=j;
                if(k<i)
                    q=k;
                if(k>i)
                    q=k-1;
                if(j>i)
                    p=j-1;
                if((k==i)||(j==i))
                    p=0;
                if(p)
                {
                    //	  fprintf(out,"%d %d\n", p, q);
                    Rr(p,q)=R(j,k);
                }
            }
        }
        //      pr_sq_mat(Rr,iter-1);
        int k;
        for(int j=0;j<titer;j++)
        {
            
            if(j<i)
                k=j;
            else if(j==i)
                k=0;
            else
                k=j-1;
            if(k)
            {
                for(int d=0;d<dim;d++)
                    tmpx[k+1][d+1]=(*pax)[j+1][d+1];
                tmpy(k)=(*pay)[j];
            }
        }
        for(int d=0;d<dim;d++)
            tmpx[titer][d+1]=(*pax)[i+1][d+1];
        //      fprintf(out,"arzze\n");
        
        //   for(int j=1;j<=iter;j++)
        //	fprintf(out,"%.5lf@ ",tmpx[j][1]);
        
        posdet(Rr, titer-1);
        if(Rr.determinant() == 0)
        {
            printf("DET is null");
            
        }
        MatrixXd InvRr = Rr.inverse();
        //cout << "MATRIX INV R" << InvRr;
        
        ypred = predict_y(tmpx, InvRr, tmpy, mu, &(param[1]), &(param[dim+1]), titer-1, dim);
        
        
        if(1)//(ypred-y(i)>10)
            //      	fprintf(stderr,"%lg - %lg\n", ypred, y(i));
            
            sum+= pow(ypred - y(i), 2);
        //  fprintf(out,"arzze\n");
        
        //ME: Free the memory from Eigen. Check if it is automatically freed.
        //~Rr;
        //~tmpy;
    }
    
    //ME: Free the memory from Eigen. Check if it is automatically freed.
    //~R;
    //~y;
    for(int i=0;i<=titer;i++)
        free(tmpx[i]);
    free(tmpx);
    
    if(sum==0)
    {
        fprintf(out,"Error\n");
        exit(0);
    }
    
    return(sum/(double)titer);
    
    //  pr_sq_mat(R, iter);
    
}




double universe::likelihood(double *param)
{
    // uses global variable storing the size of R: iter
    // uses global variable storing the dimension of the search space: dim
    // uses global ax and ay values
    
    // return(approx_likelihood(param));
    
    
    double lik;
    
    // constraint handling
    double sum=0.0;
    for(int j=1;j<pdim;j++)
    {
        if(param[j]<0.0)
            sum+=param[j];
        
    }
    if(sum<0)
        return(-sum);
    sum=0.0;
    bool outofrange=false;
    for(int j=1;j<=dim;j++)
    {
        if(param[dim+j]>=2.0)
        {
            sum+=param[dim+j]-2.0;
            outofrange=true;
        }
        else if(param[dim+j]<1.0)
        {
            sum+=-(param[dim+j]-1.0);
            outofrange=true;
        }
    }
    if(outofrange)
        return(sum);
    
    double coefficient;
    double exponent;
    
    double mu=param[2*dim+2];
    double sigma=param[2*dim+1];
    
    MatrixXd R = MatrixXd::Zero(titer,titer);
    build_R(*pax, &param[0], &param[dim], dim, titer, R);
    
    
    VectorXd y(titer);
    build_y(*pay, titer, y);
    
    double detR = posdet(R,titer);
    if (R.determinant() ==0)
        printf("DET is null");
    //fprintf(out,"R (after determinant = \n");
    //  pr_sq_mat(R, titer);
    
    //  fprintf(out,"determinant=%lg\n", detR);
    MatrixXd InvR = R.inverse();
    //cout<<"MATRIX INV IN LIKELIHOOD" << InvR;
    // fprintf(out,"Inverse = \n");
    // pr_sq_mat(InvR, titer);
    
    VectorXd one(titer);
    for(int i=0;i<titer;i++)
        one(i)=1;
    
    VectorXd vmu=one*mu;
    VectorXd diff = y - vmu;
    
    // fprintf(out, "sigma= %lg  sqrt(detR)= %lg\n", sigma, sqrt(detR));
    coefficient = 1.0/(pow(2*PI,(double)titer/2.0)*pow(sigma,(double)titer/2.0)*sqrt(detR));
    // fprintf(out,"coefficient = %lg", coefficient);
    
    //ME: Not sure still if that is dot product?????
    //???????????
    //exponent = (diff*InvR*diff)/(2*sigma);
    exponent = (double)(diff.transpose()*InvR*diff)/(2*sigma);
    //  lik = coefficient*exp(-exponent);
    lik = coefficient*exp(-(double)titer/2.0);
    
    // fprintf(out,"exponent = %lg", exponent);
    // fprintf(out, "likelihood = %lg\n", lik);
    
    
    //ME: Destructors!
    //ME: Free the memory from Eigen. Check if it is automatically freed.
    //~y;
    //~InvR;
    //~one;
    
    return(-lik);
    
    
}


double universe::mu_hat(MatrixXd& InvR, VectorXd& y, int n)
{
    double numerator, denominator;
    VectorXd one(n);
    for(int i=0;i<n;i++)
        one(i)=1;
    //cout<<"Y"<<y;
//        cout<<"INVR:"<<InvR;
    numerator = one.transpose()*InvR*y;
    denominator = one.transpose()*InvR*one;
    return(numerator/denominator);
    
}


void universe::pr_sq_mat(MatrixXd& m, int dim)
{
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            cout << m(i,j) << " ";
        }
        cout << endl;
    }
}

void universe::pr_vec(VectorXd& v, int dim)
{
    for (int i=0; i<dim; i++)
        cout << v(i) << " ";
    cout << endl;
}

double universe::standard_distribution(double z)
{
    double zv;
    int idx;
    if(z<0.0)
    {
        z *= -1;
        if(z>=7.5)
            zv = gz[75];
        else
        {
            idx = (int)(z*10);
            zv = gz[idx]+((10*z)-idx)*(gz[idx+1]-gz[idx]);
        }
        zv = 1-zv;
    }
    else if(z==0.0)
        zv = 0.5;
    else
    {
        if(z>=7.5)
            zv = gz[75];
        else
        {
            idx = (int)(z*10);
            zv = gz[idx]+((10*z)-idx)*(gz[idx+1]-gz[idx]);
        }
    }
    return(zv);
}

void universe::init_gz()
{
    gz[0]=0.50000000000000;
    gz[1]=0.53982783727702;
    gz[2]=0.57925970943909;
    gz[3]=0.61791142218894;
    gz[4]=0.65542174161031;
    gz[5]=0.69146246127400;
    gz[6]=0.72574688224993;
    gz[7]=0.75803634777718;
    gz[8]=0.78814460141985;
    gz[9]=0.81593987468377;
    gz[10]=0.84134474629455;
    gz[11]=0.86433393905361;
    gz[12]=0.88493032977829;
    gz[13]=0.90319951541439;
    gz[14]=0.91924334076623;
    gz[15]=0.93319279873114;
    gz[16]=0.94520070830044;
    gz[17]=0.95543453724146;
    gz[18]=0.96406968088707;
    gz[19]=0.97128344018400;
    gz[20]=0.97724986805182;
    gz[21]=0.98213557943718;
    gz[22]=0.98609655248650;
    gz[23]=0.98927588997832;
    gz[24]=0.99180246407540;
    gz[25]=0.99379033467422;
    gz[26]=0.99533881197628;
    gz[27]=0.99653302619696;
    gz[28]=0.99744486966957;
    gz[29]=0.99813418669962;
    gz[30]=0.99865010196838;
    gz[31]=0.99903239678678;
    gz[32]=0.99931286206208;
    gz[33]=0.99951657585762;
    gz[34]=0.99966307073432;
    gz[35]=0.99976737092097;
    gz[36]=0.99984089140984;
    gz[37]=0.99989220026652;
    gz[38]=0.99992765195608;
    gz[39]=0.99995190365598;
    gz[40]=0.99996832875817;
    gz[41]=0.99997934249309;
    gz[42]=0.99998665425098;
    gz[43]=0.99999146009453;
    gz[44]=0.99999458745609;
    gz[45]=0.99999660232688;
    gz[46]=0.99999788754530;
    gz[47]=0.99999869919255;
    gz[48]=0.99999920667185;
    gz[49]=0.99999952081672;
    gz[50]=0.99999971332813;
    gz[51]=0.99999983016330;
    gz[52]=0.99999990035088;
    gz[53]=0.99999994209631;
    gz[54]=0.99999996667842;
    gz[55]=0.99999998100990;
    gz[56]=0.99999998928215;
    gz[57]=0.99999999400951;
    gz[58]=0.99999999668420;
    gz[59]=0.99999999818247;
    gz[60]=0.99999999901340;
    gz[61]=0.99999999946965;
    gz[62]=0.99999999971768;
    gz[63]=0.99999999985118;
    gz[64]=0.99999999992231;
    gz[65]=0.99999999995984;
    gz[66]=0.99999999997944;
    gz[67]=0.99999999998958;
    gz[68]=0.99999999999477;
    gz[69]=0.99999999999740;
    gz[70]=0.99999999999872;
    gz[71]=0.99999999999938;
    gz[72]=0.99999999999970;
    gz[73]=0.99999999999986;
    gz[74]=0.99999999999993;
    gz[75]=0.99999999999997;
    
}

void universe::snake_wv(int s, int k)
{
    // This funtion uses the method of generating reflected k-ary Gray codes
    // in order to generate every normalized weight vector of k dimensions and
    // s divisions, so that they `snake' in the space, i.e., each wv is near the
    // previous one. This is useful for MOO search using weight vectors.
    
    int i,j;
    int n;
    int m;
    int d=k-1;
    //int b;
    int sum;
    int count=0;
    
    m=s;
    
    
    for(i=0;i<s;i++)
        wv[i][d]=i;
    
    n=s;
    
    while(m<mypow(s,k))
    {
        reverse(k,m,s);
        d--;
        m=s*m;
        for(i=0;i<m;i++)
            wv[i][d]=i/(m/s);
        
        //      for(i=0;i<m;i++)
        //	print_vec(wv[i],5);
        
    }
    //  for(i=0;i<pow(s,k);i++)
    //  print_vec(wv[i],k);
    
    for(i=0;i<mypow(s,k);i++)
    {
        for(j=0;j<k;j++)
            dwv[i][j]=(double)wv[i][j]/(s-1.0);
    }
    
    count=0;
    for(i=0;i<mypow(s,k);i++)
    {
        sum=0;
        for(j=0;j<k;j++)
            sum+=wv[i][j];
        if(sum==s-1)
        {
            //  print_vec_double(dwv[i],k);
            for(j=0;j<k;j++)
                normwv[count][j]=dwv[i][j];
            count++;
        }
    }
    //  printf("\n\n");
    //  printf("%d weight vectors generated\n",count);
    nwvs=count;
}

void universe::reverse(int k, int n, int s)
{
    int h,i,j;
    for(i=0;i<n;i++)
    {
        for(h=0;h<s-1;h++)
        {
            for(j=0;j<k;j++)
            {
                if(h%2==0)
                    wv[h*n+n+i][j]=wv[n-1-i][j];
                else
                    wv[h*n+n+i][j]=wv[i][j];
            }
        }
    }
    
}


void mysort(int *idx, double *val, int num)
{
    vector< pair<double, int> > sorted;
   
    
    for(int i=1;i<=num;i++)
    {
        sorted.push_back(make_pair(val[i], i));
    }
    sort(sorted.begin(), sorted.end());

    for(int i=1;i<=num;i++)
        idx[i]=sorted[i-1].second;
    
}

