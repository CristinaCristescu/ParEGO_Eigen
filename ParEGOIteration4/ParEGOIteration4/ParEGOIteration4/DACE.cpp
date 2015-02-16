//
//  DACE.cpp
//  ParEGOIteration4
//
//  Created by Bianca Cristina Cristescu on 05/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#include <cstdlib>

#define Debug false
#define PI 3.141592653
#define INFTY 1.0e35;
#define RN rand()/(RAND_MAX+1.0)

#include <iostream>

using namespace std;

#include "DACE.h"

double static myabs(double v)
{
    if(v >= 0.0)
        return v;
    else
        return -v;
}

DACE::DACE(SearchSpace* space)
{
    daceSpace = space;
    ymin=INFTY;
    double gmu = 0;
    double gsigma = 0;
    double* gtheta = NULL;
    double* gp = NULL;
    double ymin = 0;
    double gymin = 0;
    MatrixXd pgR = MatrixXd();
    MatrixXd pInvR = MatrixXd();
    VectorXd pgy = VectorXd();
    
    double glik = 0;
    
    
    //take them outttttt
    double ***pax = NULL; // a pointer
    double **pay = NULL;
}

double DACE::correlation(double *xi, double *xj, double *theta, double *p, int dim)
{
    if(Debug)
        for(int d=1;d<=dim;d++)
            fprintf(stdout, "CORRELATION: %.5lf %.5lf %.5lf %.5lf\n", xi[d],xj[d],theta[d],p[d]);
    return exp(-weighted_distance(xi,xj));
}

double DACE::weighted_distance(double *xi, double *xj)
{
    double sum=0.0;
    
    double nxi[daceSpace->fSearchSpaceDim+1];
    double nxj[daceSpace->fSearchSpaceDim+1];
    
    
    
    for(int h=1;h<=daceSpace->fSearchSpaceDim;h++)
    {
        nxi[h] = (xi[h]-daceSpace->fXMin[h])/(daceSpace->fXMax[h]-daceSpace->fXMin[h]);
        nxj[h] = (xj[h]-daceSpace->fXMin[h])/(daceSpace->fXMax[h]-daceSpace->fXMin[h]);
        
        sum += gtheta[h]*pow(myabs(nxi[h]-nxj[h]),gp[h]);
        //      sum += 4.0*pow(myabs(xi[h]-xj[h]),2.0);     // theta=4 and p=2
        
    }
    if(Debug)
        fprintf(stdout, "sum: %.5lf", sum);
    return(sum);
}



void DACE::build_y(double *ay, VectorXd& y)
{
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
        y(i)=ay[i+1];
}

void DACE::build_R(double **ax, MatrixXd& R)
{
    // takes the array of x vectors, theta, and p, and returns the correlation matrix R.
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
    {
        for(int j=0;j<daceSpace->fCorrelationSize;j++)
        {
            if(Debug)
                fprintf(stdout,"%.5lf %.5lf %.5lf  %.5lf %d\n", ax[i][1],ax[j][1],gtheta[1], gp[1], daceSpace->fSearchSpaceDim);
            R(i,j)=correlation(ax[i+1],ax[j+1], gtheta, gp, daceSpace->fCorrelationSize);
            if(Debug)
                fprintf(stdout,"%.5lf\n", R(i,j));
        }
        if(Debug)
            fprintf(stdout,"\n");
    }
    
}

double DACE::s2(double **ax)
{
    if(Debug)
        printf("sigma: %f\n", gsigma);
    double s2;
    VectorXd one(daceSpace->fCorrelationSize);
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
        one(i)=1;
    
    VectorXd r(daceSpace->fCorrelationSize);
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
    {
        //fprintf(out,"theta=%.5lf p=%.5lf ax[n+1]=%.5lf, ax[i+1]=%.5lf\n", theta[1],p[1],ax[n+1][1],ax[i+1][1]);
        r(i) = correlation(ax[daceSpace->fCorrelationSize+1],ax[i+1],gtheta,gp,daceSpace->fSearchSpaceDim);
        //fprintf(out,"r[i]=%.5lf ",r(i));
    }
    
    s2 = gsigma * (1- r.transpose()*pInvR*r +
                  pow((1-one.transpose()*pInvR*r),2)/(one.transpose()*pInvR*one));
    return(s2);
}

double DACE::mu_hat(VectorXd& y)
{
    double numerator, denominator;
    VectorXd one(daceSpace->fCorrelationSize);
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
        one(i)=1;
    numerator = one.transpose()*pInvR*y;
    denominator = one.transpose()*pInvR*one;
    return(numerator/denominator);
    
}

double DACE::sigma_squared_hat(VectorXd& y)
{
    double numerator, denominator;
    
    VectorXd one(daceSpace->fCorrelationSize);
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
        one(i)=1;
    
    VectorXd vmu=one*gmu;
    VectorXd diff = y - vmu;
    
    numerator = diff.transpose()*pInvR*diff;
    denominator = daceSpace->fCorrelationSize;

    return (numerator/denominator);
}

double DACE::predict_y(double **ax)
{
    double y_hat;
    VectorXd one(daceSpace->fCorrelationSize);
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
        one(i)=1;
    
    VectorXd r(daceSpace->fCorrelationSize);
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
    {
        r(i) = correlation(ax[daceSpace->fCorrelationSize+1],ax[i+1],gtheta,gp,daceSpace->fSearchSpaceDim);
        //fprintf(out,"r[%di]=%.5lf ax[n+1][1]=%.5lf\n",i, r(i), ax[n+1][1]);
    }
    
    double intermidiate = ((r.transpose()*pInvR)*(pgy-(one*gmu)));
    y_hat = gmu + intermidiate;
    
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



long double DACE::posdet(MatrixXd& R, int n)
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

double DACE::likelihood(double *param, double** pax, double* pay)
{
    // uses global variable storing the size of R: iter
    // uses global variable storing the dimension of the search space: dim
    // uses global ax and ay values
    
    // return(approx_likelihood(param));
    
    
    double lik;
    
    // constraint handling
    double sum=0.0;
    for(int j=1;j<daceSpace->fNoParamDACE;j++)
    {
        if(param[j]<0.0)
            sum+=param[j];
        
    }
    if(sum<0)
        return(-sum);
    sum=0.0;
    bool outofrange=false;
    for(int j=1;j<=daceSpace->fSearchSpaceDim;j++)
    {
        if(param[daceSpace->fSearchSpaceDim+j]>=2.0)
        {
            sum+=param[daceSpace->fSearchSpaceDim+j]-2.0;
            outofrange=true;
        }
        else if(param[daceSpace->fSearchSpaceDim+j]<1.0)
        {
            sum+=-(param[daceSpace->fSearchSpaceDim+j]-1.0);
            outofrange=true;
        }
    }
    if(outofrange)
        return(sum);
    
    double coefficient;
    double exponent;
    
    double mu=param[2*daceSpace->fSearchSpaceDim+2];
    double sigma=param[2*daceSpace->fSearchSpaceDim+1];
    
    MatrixXd R = MatrixXd::Zero(daceSpace->fCorrelationSize,daceSpace->fCorrelationSize);
    build_R(pax, R);
    
    
    VectorXd y(daceSpace->fCorrelationSize);
    build_y(pay, y);
    
    double detR = posdet(R,daceSpace->fCorrelationSize);
    
    //  pr_sq_mat(R, fCorrelationSize);
    
    //  fprintf(out,"determinant=%lg\n", detR);
    MatrixXd InvR = R.inverse();
    //cout<<"MATRIX INV IN LIKELIHOOD" << InvR;
    // fprintf(out,"Inverse = \n");
    // pr_sq_mat(InvR, fCorrelationSize);
    
    VectorXd one(daceSpace->fCorrelationSize);
    for(int i=0;i<daceSpace->fCorrelationSize;i++)
        one(i)=1;
    
    VectorXd vmu=one*mu;
    VectorXd diff = y - vmu;
    
    // fprintf(out, "sigma= %lg  sqrt(detR)= %lg\n", sigma, sqrt(detR));
    coefficient = 1.0/(pow(2*PI,(double)daceSpace->fCorrelationSize/2.0)*pow(sigma,(double)daceSpace->fCorrelationSize/2.0)*sqrt(detR));
    // fprintf(out,"coefficient = %lg", coefficient);
    
    //ME: Not sure still if that is dot product?????
    //???????????
    //exponent = (diff*InvR*diff)/(2*sigma);
    exponent = (double)(diff.transpose()*InvR*diff)/(2*sigma);
    //  lik = coefficient*exp(-exponent);
    lik = coefficient*exp(-(double)daceSpace->fCorrelationSize/2.0);
    
    // fprintf(out,"exponent = %lg", exponent);
    // fprintf(out, "likelihood = %lg\n", lik);
    
    return(-lik);
    
    
}

void DACE::get_params(double** param)
{
    // uses fSearchSpaceDim value
    for (int i=1;i <= daceSpace->fSearchSpaceDim; i++)
    {
        gtheta[i]=param[1][i];
        gp[i]=param[1][i+daceSpace->fSearchSpaceDim];
    }
    gsigma=param[1][2*daceSpace->fSearchSpaceDim+1];
    gmu=param[1][2*daceSpace->fSearchSpaceDim+2];
    
}


void DACE::buildDACE(double** param, double** bestparam, bool change, int iter)
{
    
    gp=(double *)calloc((daceSpace->fSearchSpaceDim+1),sizeof(double));
    gtheta=(double *)calloc((daceSpace->fSearchSpaceDim+1),sizeof(double));
    
    VectorXd y(daceSpace->fCorrelationSize);
    build_y(*pay, y);
    
    double best_lik=INFTY;
    int besti;
    // ME: changed from change=1
    if(change=1)
    {
        
        for(int i=0;i<30;i++)
        {
            for(int d=1; d<=daceSpace->fSearchSpaceDim; d++)
            {
                double rn1 = RN;
                //printf("rn10 = %lg\n", rn1);
                double rn2 = RN;
                //printf("rn11 = %lg\n", rn2);
                gtheta[d]=1+rn1*2;
                gp[d]=1.01+rn2*0.98;
                //printf("rn1=%lg rn2=%lg gtheta[d]=%lg gp[d]=%lg\n", rn1, rn2, gtheta[d], gp[d]);
            }
            //ME: Carefull with the indexes!!! It starts from 1.
            MatrixXd R = MatrixXd::Zero(daceSpace->fCorrelationSize, daceSpace->fCorrelationSize);
            build_R(*pax, R);
            //  fprintf(out,"NEW STARTING R Matrix\n");
            // pr_sq_mat(R,iter);
            
            double detR=posdet(R,daceSpace->fCorrelationSize);
            
            //printf("det = %.5lf\n", R.determinant());
            // pr_sq_mat(R,fCorrelationSize);
            //cout<< "THe MATRIX:"<<R<<"\n";
            pInvR = R.inverse();
            //cout<<"MATRIX INV IN CHANGE" <<InvR<<"\n";
            gmu=mu_hat(y);
            //	  fprintf(out,"OK - mu calculated\n");
            gsigma=sigma_squared_hat(y);
            //  fprintf(out,"OK - sigma calculated\n");
            
            for(int j=1;j<=daceSpace->fSearchSpaceDim;j++)
            {
                param[1][j]=gtheta[j];
                param[1][j+daceSpace->fSearchSpaceDim]=gp[j];
            }
            param[1][2*daceSpace->fSearchSpaceDim+1]=gsigma;
            param[1][2*daceSpace->fSearchSpaceDim+2]=gmu;
            
//                        for (int i = 0; i < 2*daceSpace->fSearchSpaceDim+2+1; i++)
//                            fprintf(stdout, "%lg ", param[1][i]);
//                        fprintf(stdout, "\n");
            
            glik=likelihood(param[1], *pax, *pay);
            
            if(glik<best_lik)
            {
                besti = i;
                //  fprintf(stderr,"glik= %lg best_lik= %lg  BETTER....\n", glik, best_lik);
                best_lik=glik;
                for(int j=1;j<=daceSpace->fNoParamDACE;j++)
                {
                    bestparam[1][j]=param[1][j];
                }
            }
            
        }
    }
    
    get_params(bestparam);
    
    
//        	fprintf(stdout,"FINAL DACE parameters = \n");
//        	for(int d=1; d<=daceSpace->fSearchSpaceDim; d++)
//        	fprintf(stdout,"%lg ", gtheta[d]);
//        	for(int d=1; d<=daceSpace->fSearchSpaceDim; d++)
//        	fprintf(stdout,"%lg ", gp[d]);
//        	fprintf(stdout," %lg %lg\n", bestparam[1][2*daceSpace->fSearchSpaceDim+1], bestparam[1][2*daceSpace->fSearchSpaceDim+2]);
//    
//    
    /* Use the full R matrix */
    daceSpace->fCorrelationSize=iter;
    pax = &daceSpace->fXVectors;
    //ME: Carefull with the indexes!!! It starts from 1.
    pgR = MatrixXd::Zero(daceSpace->fCorrelationSize,daceSpace->fCorrelationSize);
    build_R(*pax, pgR);
    posdet(pgR, daceSpace->fCorrelationSize);
    pInvR = pgR.inverse();
    //cout<<"MATRIX INV" <<InvR<<"\n";
    pgy = VectorXd(daceSpace->fCorrelationSize);
    build_y(daceSpace->fMeasuredFit, pgy);
    /* ***************************** */
    
    //   fprintf(out,"predicted R matrix built OK:\n");
    //   pr_sq_mat(pgR,fCorrelationSize);
    
    
    // set the global variables equal to the local ones
    //printf("gsigma %f\n", gsigma);
//    printf("GP\n");
//    for(int index = 1; index < daceSpace->fSearchSpaceDim+1; index++)
//        printf("%lf \n", gp[index]);
//    printf("\n");
    
    

 
}




