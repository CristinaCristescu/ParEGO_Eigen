//
//  ParEGOIteration4.cpp
//  ParEGOIteration4
//
//  Created by Bianca Cristina Cristescu on 30/10/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//


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
 maximum likelihood model-> However, I later found that using a
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
// Use the Eigen library as a first trial for comparison and profiling.
#include </Users/cristina/developer/eigen/eigen-eigen-1306d75b4a21/Eigen/Dense>
//#include "nrutil.h"

#include "SearchSpace.h"
#include "WeightVector.h"
#include "DACE.h"

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
    SearchSpace* space = NULL; // the model fo the search space->
    WeightVector* weights = NULL;
    DACE* model = NULL ;
        
    int improvements[300];

    bool Debug;
    
    double global_min;
        int best_ever;
    double niche;
    
    double gz[76]; // an array holding the z values for the gaussian distribution
    
    clock_t start, end;
    double cpu_time_used;
public:

    universe(int x);
    void init_ParEGO();
    void iterate_ParEGO();
    void setspace(const char* objectiveFunctionName);
    void setweights();
    void setDACE();

private:
    void init_gz();
    void cwr(int **target, int k, int n);
    void pr_sq_mat(MatrixXd& m, int dim);
    void pr_vec(VectorXd& v, int dim);
    double myabs(double v);
    double myfit(int i, int j);
    double standard_density(double z);
    double standard_distribution(double z);
    double expected_improvement(double yhat, double ymin, double s);
    double wrap_ei(double *x); // returns -(expected improvement), given the solution x
    void latin_hyp(double **ax, int iter);
    void compute_n(int n, int *dig, int base);
    double Tcheby(double *wv, double *vec, double *ideal);
    int tourn_select(double *xsel, double **ax, double *ay, int iter, int t_size);
    void mutate(double *x);
    void cross(double *child, double *par1, double *par2);
    
};

universe::universe(int x)
{
    Debug=false;
    best_ever=1;
    niche=1.0;
    MAX_ITERS = 100;
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
    
    for(int d=1; d<=space->fSearchSpaceDim; d++)
    {
        
        /*Selected Two Parents*/
        
        xl = space->fXMin[d];
        xu = space->fXMax[d];
        
        /* Check whether variable is selected or not*/
        double rn = RN;
        //printf("rn1 = %lg\n", rn);
        if(rn <= 0.5)
        {
            if(myabs(par1[d] - par2[d]) > 0.000001)
            {
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
                //printf("rn2 = %lg\n", rnd);
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
        //printf("idx[i] %d ay[idx[i]]=%lg\n", idx[i], ay[idx[i]]);

        if(ay[idx[i]]<besty)
        {
            besty=ay[idx[i]];
            bestidx=i;
            //printf("besty %lg bestidx %d\n", besty, bestidx);

        }
    }
    for(int d=1;d<=space->fSearchSpaceDim;d++)
    {
        xsel[d] = ax[idx[bestidx]][d];
    }
    //printf("res %d \n", idx[bestidx]);
    return(idx[bestidx]);
}

void universe::mutate(double *x)
{
    // this mutation always mutates at least one gene (even if the m_rate is set to zero)
    // this is to avoid duplicating points which would cause the R matrix to be singular
    double m_rate=1.0/space->fSearchSpaceDim;
    double shift;
    bool mut[space->fSearchSpaceDim+1];
    int nmutations=0;
    for(int i=1;i<=space->fSearchSpaceDim;i++)
    {
        mut[i]=false;
        double rn = RN;
        //printf("rn3 = %lg\n", rn);
        if(rn<m_rate)
        {
            mut[i]=true;
            nmutations++;
        }
    }
    
    if(nmutations==0)
    {
        double rn = RN;
        //printf("rn4 = %lg\n", rn);
        mut[1+(int)(rn*space->fSearchSpaceDim)]=true;
    }
    
    
    for(int d=1;d<=space->fSearchSpaceDim;d++)
    {
        if(mut[d])
        {
            double rn = RN;
            //printf("rn5 = %lg\n", rn);
            shift = (0.0001+rn)*(space->fXMax[d]-space->fXMin[d])/100.0;
            rn = RN;
            //printf("rn6 = %lg\n", rn);
            if(rn<0.5)
                shift*=-1;
            
            
            x[d]+=shift;
            
            //printf("space max %lg\n", space->fXMax[d]);
            if(x[d]>space->fXMax[d])
                x[d]=space->fXMax[d]-myabs(shift);
            else if(x[d]<space->fXMin[d])
                x[d]=space->fXMin[d]+myabs(shift);
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
    
    
    
    for(i=0;i<space->fNoObjectives;i++)
    {
        norm[i] = (vec[i]-space->fAbsMin[i])/(space->fAbsMax[i]-space->fAbsMin[i]);
        nideal[i] = ideal[i];
        diff = wv[i]*(norm[i]-nideal[i]);
        sum += diff;
        if(diff>d_max)
            d_max = diff;
    }
    
    
    // fprintf(out, "d_max= %.5lf + 0.5 * sum= %.5lf\n", d_max, sum);
    return(d_max + 0.05*sum);
}



void universe::compute_n(int n, int *dig, int base)
{
    // given the number n, convert it to base base, and put the digits into dig
    // we already know the number n will be less than pow(base,fSearchSpaceDim) where fSearchSpaceDim is global
    int d=space->fSearchSpaceDim-1;
    int div = (int)pow((double)base,(double)(space->fSearchSpaceDim-1));
    for(int i=0; i < space->fSearchSpaceDim; i++)
    {
        dig[i] = n/div;
        n-=dig[i]*div;
        d--;
        div = (int)pow(double(base),double(d));
    }
}

void universe::latin_hyp(double **ax, int iter)
{
    int v;
    
    double L[space->fSearchSpaceDim][iter];
    bool viable[space->fSearchSpaceDim][iter];
    
    
    for(int d=0; d<space->fSearchSpaceDim; d++)
    {
        for(int i=0;i<iter;i++)
        {
            viable[d][i]=true;
            
            L[d][i] = space->fXMin[d+1] + i*((space->fXMax[d+1]-space->fXMin[d+1])/double(iter));
        }
    }
    for(int i=0; i<iter; i++)
    {
        for(int d = 0; d < space->fSearchSpaceDim; d++)
        {
            
            do
            {
                double rn = RN;
                //printf("rn7 = %lg\n", rn);
                v = int(rn*iter);
            }
            while(!viable[d][v]);
            viable[d][v]=false;
            double rn = RN;
            //printf("rn8 = %lg\n", rn);
            ax[i+1][d+1] = L[d][v]+rn*((space->fXMax[d+1]-space->fXMin[d+1])/double(iter));
        }
        
    }
}

double universe::myabs(double v)
{
    if(v >= 0.0)
        return v;
    else
        return -v;
}

/* HERE 2
 Below are the current fitness function definitions. Add your functions here.
 The function should take an x and y array and compute the value of the y objectives
 from the x decision variables. Note that the first decision variable is x[1] and t
 he first objective is y[1], i.e. there is an offset to the array. */




double universe::myfit(int i, int j)
{
    /* HERE 3
     
     Here is where the fitness function is called. Just add in a call for your function here,
     following this format.
     
     */
//    printf("myfit\n");
//    for (int j=1; j < space->fSearchSpaceDim+1; j++)
//    {
//        printf( "%.5lf ", space->fXVectors[i][j]);
//        printf( "%.5lf ", space->fCostVectors[i][j]);
//    }
//    printf("\n");
    
    space->applyFit(i,j);
//    printf("after myfit/n");
//    for (int j=1; j < space->fSearchSpaceDim+1; j++)
//    {
//        printf( "%.5lf ", space->fXVectors[i][j]);
//        printf( "%.5lf ", space->fCostVectors[i][j]);
//    }
//    printf("\n");
//    printf("\n");
    return(Tcheby(space->fWeightVectors,&space->fCostVectors[j][1],space->fIdealObjective));
    
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
        //printf("rn9 = %lg\n", rn);
        j=(int)(rn*l_t);
        to[i]=from[j];
        from[j]=from[l_t-1];
        l_t--;
    }
}

double universe::standard_density(double z)
{
    double psi;
    
    psi = (1/sqrt(2*PI))*exp(-(z*z)/2.0);
    return (psi);
    
}


double universe::expected_improvement(double yhat, double ymin, double s)
{
    double E;
    double sdis;
    double sden;
    if(s<=0)
        return 0;
    if((ymin-yhat)/s < -7.0)
        sdis = 0.0;
    else if((ymin-yhat)/s > 7.0)
        sdis = 1.0;
    else
        sdis = standard_distribution( (ymin-yhat)/s );
    
    sden = standard_density((ymin-yhat)/s);
    
    E = (ymin - yhat)*sdis + s*sden;
    return E;
}


double universe::wrap_ei(double *x)
{
    //printf("WRAP_BEGIN\n");
    for(int d=1;d<=space->fSearchSpaceDim; d++)
        (*model->pax)[space->fCorrelationSize+1][d] = x[d];
    //fprintf(stdout,"X vectors:%.5lf %.5lf\n", x[1], (*pax)[space->fCorrelationSize+1][1]);
    double fit;
    // predict the fitness
    fit=model->predict_y(*model->pax);
    
    //fprintf(stdout,"predicted fitness in wrap_ei() = %lg\n", fit);
    
    
    // compute the error
    double ss;
    //ME: ss -least square error.
    ss=model->s2(*model->pax);
    //fprintf(stdout,"s^2 error in wrap_ei() = %lg\n", ss);
    //fprintf(stderr,"%.9lf %.9lf \n", x[1], ss);

    
    // compute the expected improvement
    double ei;
    ei=expected_improvement(fit, model->gymin, sqrt(ss));
     //fprintf(stdout,"-ei in wrap_ei() = %lg\n", -ei);
    
    
    for(int d=1; d <= space->fSearchSpaceDim; d++)
    {
        if((x[d]>space->fXMax[d])||(x[d]<space->fXMin[d]))
            ei=-1000;
    }
    
    //fprintf(stderr,"%.9lf\n", ei);
    
    //printf("WRAP_END\n");
    // return the expected improvement
    return(-ei);
}

void universe::setweights()
{
    weights = new WeightVector(space->fNoObjectives);
}

void universe::setspace(const char* objectiveFunctionName)
{
    space = new SearchSpace(objectiveFunctionName, MAX_ITERS);
}

void universe::setDACE()
{
    model = new DACE(space);
}

int main(int argc, char **argv)
{
    clock_t time1, time2;
    time1 = clock();
    //out = fopen("outputEigen.txt", "w");
    universe U(1);
    cerr << "errorororor!";
    unsigned int seed=47456536;
    //long seed = 3563563;
    //if(argc>1)
    //seed = atoi(argv[1]);
    srand(47456536);
    string function = "f_vlmop2";
    U.setspace(function.c_str());
    U.setweights();
    U.setDACE();
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
    
    double **param;
    param = (double **)calloc((space->fNoParamDACE+2),sizeof(double *));
    for(int i=0;i<=space->fNoParamDACE+1;i++)
        param[i]=(double *)calloc((space->fNoParamDACE+1),sizeof(double));
    
    double **bestparam;
     bestparam = (double **)calloc((space->fNoParamDACE+2),sizeof(double *));
     for(int i=0;i<=space->fNoParamDACE+1;i++)
     bestparam[i]=(double *)calloc((space->fNoParamDACE+1),sizeof(double));
    
    
    
    //  double recomp_prob=1.0;
    
    weights->changeWeights(iter, space->fWeightVectors);
    
          //fprintf(stdout, "%.2lf %.2lf weightvectors \n", space->fWeightVectors[0], space->fWeightVectors[1]);
    
    for(int i=1;i<=iter;i++)
    {
        space->fMeasuredFit[i] = Tcheby(space->fWeightVectors,&(space->fCostVectors[i][1]),space->fIdealObjective);
        
        if(space->fMeasuredFit[i]<model->ymin)
        {
            model->ymin=space->fMeasuredFit[i];
            best_ever=i;
        }
    }
    
    
    if(iter>11*space->fSearchSpaceDim+24)
    {
        space->fCorrelationSize =11*space->fSearchSpaceDim+24;
        
        int ind[iter+1];
        mysort(ind, space->fMeasuredFit, iter);
        for (int i=1; i<=space->fCorrelationSize/2; i++)
        {
            for(int d=1;d<=space->fSearchSpaceDim;d++)
                space->fSelectedXVectors[i][d] = space->fXVectors[ind[i]][d];
            space->fSelectedMeasuredFit[i] = space->fMeasuredFit[ind[i]];                   // fprintf(stderr, "evaluate\n");
            
            //find the best fitness out of the selected initial solutions
            //	  if(ay[parA]<best_imp)
            //  best_imp=ay[parA];
        }
        
        
        int *choose;
        cwr(&choose,iter-space->fCorrelationSize/2,iter-space->fCorrelationSize/2); // cwr(k,n) - choose without replacement k items from n
        
        for(int i=space->fCorrelationSize/2+1;i<=space->fCorrelationSize;i++)
        {
            assert(choose[i-space->fCorrelationSize/2]+space->fCorrelationSize/2 <= iter && 0 < choose[i-space->fCorrelationSize/2]+space->fCorrelationSize/2);
            int j= ind[choose[i-space->fCorrelationSize/2]+space->fCorrelationSize/2];
            for(int d=1;d <=space->fSearchSpaceDim;d++)
                space->fSelectedXVectors[i][d] = space->fXVectors[j][d];
            space->fSelectedMeasuredFit[i] = space->fMeasuredFit[j];
        }
        
        model->pax=&space->fSelectedXVectors;
        model->pay=&space->fSelectedMeasuredFit;
        free(choose);
    }
    else
    {
        space->fCorrelationSize=iter;
        model->pax=&space->fXVectors;
        model->pay=&space->fMeasuredFit;
    }
    
    
    
    double best_imp=INFTY;
    double best_x[space->fSearchSpaceDim+1];

    model->buildDACE(param, bestparam, weights->change, iter);
    
    model->gymin = INFTY;
    
    for(int i = 1;i < space->fCorrelationSize;i++)
        if((model->pgy)[i]<model->gymin)
            model->gymin=(model->pgy)[i];
    
    int gapopsize=20;
    
    double **popx;
    popx = (double **)calloc((gapopsize+1),sizeof(double *));  // GA population for searching over modelled landscape
    for(int i=0;i<gapopsize+1;i++)
        popx[i]=(double *)calloc((space->fSearchSpaceDim+1),sizeof(double));
    
    double popy[gapopsize+1];
    
    double mutx[space->fSearchSpaceDim+1];
    
    double muty;

    
    start = clock();
    
    // BEGIN GA code
    best_imp=INFTY;
    int parA;
    int parB;
    
    
    // initialize using the latin hypercube method
    latin_hyp(popx, gapopsize);
//    printf("popx ");
//    for(int i = 1; i < gapopsize + 1; i++)
//        for (int j = 1; j < space->fSearchSpaceDim+1; j++)
//        {
//            printf("%lg ", popx[i][j]);
//            printf("\n");
//        }
//
//    printf("popy ");
    for (int i=1; i<=gapopsize; i++)
    {
        popy[i] = wrap_ei(popx[i]);
        //printf("%lg ", popy[i]);

    }
    //printf("POPY_END\n");
    
    // initialize with mutants of nearby good solutions
    
    int ind[iter+1];
    mysort(ind, space->fMeasuredFit, iter);
    
    //   for(int i=1;i<=fCorrelationSize;i++)
    //	printf("ay = %.5lf ; %.5lf\n", ay[ind[i]], ay[i]);
    
    //printf("Begin_MUTATION");

    for (int i=1; i<=5; i++)
    {
        //	  printf("parent = %.5lf\n", ay[ind[i]]);
        parA=ind[i];
        for(int d=1;d<=space->fSearchSpaceDim;d++)
            popx[i][d] = space->fXVectors[parA][d];
        mutate(popx[i]);                                // fprintf(stderr, "mutate\n");
//        for(int j=1; j < space->fSearchSpaceDim+1 ;j++)
//            printf("%lg ", popx[i][j]);
        popy[i] = wrap_ei(popx[i]);                   // fprintf(stderr, "evaluate\n");
        
    }
    
    //printf("End_MUTATION");

    
    double p_cross=0.2;
    for (int i=1; i<=10000; i++)
    {
//        //printf("GA iter %d ", i);
//        //printf("popx ");
//        for(int i = 1; i < gapopsize + 1; i++)
//            for (int j = 1; j < space->fSearchSpaceDim+1; j++)
//            {
//                printf("%lg ", popx[i][j]);
//                printf("\n");
//            }
        

        double rn = RN;
        //printf("rn12 = %lg\n", rn);
        if(rn < p_cross)
        {
            if(0.127396<=rn && rn <= 0.127398){
                cerr << "bla\n";
                cerr << iter <<"\n";
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
            for(int d=1;d<=space->fSearchSpaceDim;d++)
            {
                popx[parA][d]=mutx[d];
                
            }
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
            for(int d=1; d<=space->fSearchSpaceDim; d++)
                best_x[d]=popx[i][d];
        }
    }
    printf("ei= %lf\n", best_imp);
    if(improved==false)
    {
        fprintf(stderr, "GA found no improvement\n");
        for(int d=1; d<=space->fSearchSpaceDim; d++)
        {
            best_x[d]=popx[1][d];
        }
        mutate(best_x);
    }
    // END GA code
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // timestr << iter << " " << cpu_time_used << endl;
    
    
    for(int d=1;d<=space->fSearchSpaceDim;d++)
        space->fXVectors[iter+1][d]=best_x[d];
    space->fMeasuredFit[iter+1]=myfit(iter+1, iter+1);
    
    
    printf("%d ", iter+1+prior_it);
    for(int d=1;d <=space->fSearchSpaceDim; d++)
        printf("%.5lg ", space->fXVectors[iter+1][d]);
    printf("decision\n");
    
    printf("%d ", iter+1);
    for(int i=1;i<=space->fNoObjectives;i++)
    {
        printf( "%.5lf ", space->fCostVectors[iter+1][i]);
        //fprintf(plotfile, "%.5lf ", ff[iter+1][i]);
    }
    //fprintf(plotfile, "\n");
    printf("objective\n");
    
    improvements[iter+1]=improvements[iter];
    if (space->fMeasuredFit[iter+1]>=model->ymin)
    {
        // fprintf(stderr,"No actual improver found\n");
        stopcounter++;
    }
    
    else
    {
        improvements[iter+1]++;
        model->ymin = space->fMeasuredFit[iter+1];
        stopcounter=0;
        best_ever=iter+1;
    }
    
    //ME: Free the memory from Eigen. Check if it is automatically freed.
    //~InvR;
    //~Rpred;
    //~fy;
    iter++;
    
    
    for(int i=0;i<gapopsize+1;i++)
        free(popx[i]);
    free(popx);
    
    for(int i=0;i<=space->fNoParamDACE+1;i++)
        free(param[i]);
    free(param);
    
    for(int i=0;i<=space->fNoParamDACE+1;i++)
        free(bestparam[i]);
    free(bestparam);

    
    
    
    
}


void universe::init_ParEGO()
{
    
    init_gz();
    
    int prior_it=0;
    
    iter=10*space->fSearchSpaceDim+(space->fSearchSpaceDim-1);
    do
    {
        for(int i=1; i<=iter;i++)
            improvements[i]=0;
        
        
        
        latin_hyp(space->fXVectors, iter);  // select the first solutions using the latin hypercube
        
//        printf("init_ParEGO\n");
//        for (int i = 1; i < 3; i++)
//        {    for (int j=1; j < space->fSearchSpaceDim+1; j++)
//             {
//                printf("%.5lf ", space->fXVectors[i][j]);
//                printf("%.5lg ", space->fCostVectors[i][j]);
//             }
//            printf("\n");
//        }
        
        for(int i=1;i<=iter;i++)
        {
            space->fMeasuredFit[i]=myfit(i,i);
            printf( "%d ", i+prior_it);
            for(int d=1;d<=space->fSearchSpaceDim;d++)
                printf( "%.5lf ", space->fXVectors[i][d]);
            printf("decision\n");
            printf("%d ", i+prior_it);
            for(int k=1;k<=space->fNoObjectives;k++)
                printf("%.5lg ", space->fCostVectors[i][k]);
            printf("objective\n");
        }
    }while(0);
    
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

