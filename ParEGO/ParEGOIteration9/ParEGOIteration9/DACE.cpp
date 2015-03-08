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
#define NMAX 100

#include <iostream>
#include <math.h>

using namespace std;

#include "DACE.h"
#include "Utilities.cpp"



double static myabs(double v)
{
    if(v>=0.0)
        return v;
    else
        return -v;
}

DACE::DACE(SearchSpace* space)
{
    daceSpace = space;
    fNoParamDACE=daceSpace->fSearchSpaceDim*2+2;
    ymin=INFTY;
    gmu = 0;
    gsigma = 0;
    gtheta = NULL;
    gp = NULL;
    gymin = 0;
    pInvR = MyMatrix();
    pgy = MyVector();
    
    glik = 0;
    
    //take them outttttt
    pax = NULL; // a pointer
    pay = NULL;
    
    init_gz();
}

DACE::~DACE()
{
    free(gtheta);
    free(gp);
}

double DACE::correlation(double *xi, double *xj, double *theta, double *p, int dim)
{
    
        //for(int d=1;d<=dim;d++)
            //fprintf(stdout,, "CORRELATION: %.5lf %.5lf %.5lf %.5lf\n", xi[d],xj[d],theta[d],p[d]);
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
        
    }
    if(Debug)
        fprintf(stdout, "sum: %.5lf", sum);
    return(sum);
}


void DACE::build_R(double **ax, MyMatrix& R)
{
    // takes the array of x vectors, theta, and p, and returns the correlation matrix R.
    for(int i=0;i<fCorrelationSize;i++)
    {
        for(int j=0;j<fCorrelationSize;j++)
        {
            if(Debug)
                fprintf(stdout,"%.5lf %.5lf %.5lf  %.5lf %d\n", ax[i][1],ax[j][1],gtheta[1], gp[1], daceSpace->fSearchSpaceDim);
            R.insert(i,j,correlation(ax[i+1],ax[j+1], gtheta, gp, fCorrelationSize));
            if(Debug)
                fprintf(stdout,"%.5lf\n", R(i,j));
        }
        if(Debug)
            fprintf(stdout,"\n");
    }
    
}

double DACE::s2(double **ax)
{
    double s2;
    MyVector one(fCorrelationSize, 1);
    
    MyVector r(fCorrelationSize);
    for(int i=0;i<fCorrelationSize;i++)
    {
        //fprintf(stdout,"theta=%.5lf p=%.5lf ax[n+1]=%.5lf, ax[%d+1]=%.5lf\n", gtheta[i+1],gp[i+1],ax[fCorrelationSize+1][i+1],i, ax[i+1][i]);
        r.insert(i, correlation(ax[fCorrelationSize+1],ax[i+1],gtheta,gp,daceSpace->fSearchSpaceDim));
        //fprintf(stdout,"r[i]=%lg \n",r(i));
    }
    double intermediate = (1- (r.transpose()*pInvR*r)(0,0) +
                           pow((1-(one.transpose()*pInvR*r)(0,0)),2)/(one.transpose()*pInvR*one)(0,0));
    if (intermediate < 0)
        intermediate = myabs(intermediate);
    s2 = gsigma * intermediate;
    return(s2);
}

double DACE::mu_hat(MyVector& y, int iter)
{
    double numerator, denominator;
    MyVector one(fCorrelationSize, 1);
    
    MyMatrix interm(one*pInvR);
    numerator = (interm*y)(0,0);
    denominator = (one.transpose()*pInvR*one)(0,0);
    return(numerator/denominator);
    
}

double DACE::sigma_squared_hat(MyVector& y)
{
    double numerator, denominator;
    MyVector one(fCorrelationSize,1);
    
    MyVector vmu(one*gmu);
    MyVector diff = y - vmu;
    numerator = (diff.transpose()*pInvR*diff)(0,0);
    denominator = fCorrelationSize;
    
    return (numerator/denominator);
}

double DACE::predict_y(double **ax)
{
    double y_hat;
    MyVector one(fCorrelationSize, 1);
    
    MyVector r(fCorrelationSize);
    for(int i=0;i<fCorrelationSize;i++)
    {
        r.insert(i, correlation(ax[fCorrelationSize+1],ax[i+1],gtheta,gp,daceSpace->fSearchSpaceDim));
        //fprintf(stdout,"r[]=%.20lg\n", r(i));
    }
    //fprintf(stdout, "in: %.20lg\n", (r.transpose()*pInvR)(0,0));
    //fprintf(stdout, "pgy: %.20lg\n", pgy(0));
    double intermidiate = ((r.transpose()*pInvR)*(pgy-(one*gmu)))(0,0);
    //fprintf(stdout, "multip y: %.20lg\n", intermidiate);
    y_hat = gmu + intermidiate;
    
    //fprintf(stdout,"y_hat=%f mu_hat=%f\n",y_hat, mu_hat);
    
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




double DACE::likelihood(double *param, double** pax, double* pay)
{
    // uses global variable storing the size of R: iter
    // uses global variable storing the dimension of the search space: dim
    // uses global ax and ay values
    
    double lik;
    
    // constraint handling
    double sum=0.0;
    for(int j=1;j<fNoParamDACE;j++)
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

    double sigma=param[2*daceSpace->fSearchSpaceDim+1];
    
    MyMatrix R(fCorrelationSize,fCorrelationSize);
    build_R(pax, R);
    
    MyVector y = MyVector(fCorrelationSize, pay);
    
    double detR = R.posdet();
    
    // fprintf(stdout, "sigma= %lg  sqrt(detR)= %lg\n", sigma, sqrt(detR));
    coefficient = 1.0/(pow(2*PI,(double)fCorrelationSize/2.0)*pow(sigma,(double)fCorrelationSize/2.0)*sqrt(detR));
    // fprintf(stdout,"coefficient = %lg", coefficient);
    
    lik = coefficient*exp(-(double)fCorrelationSize/2.0);
    
    return(-lik);
    
    
}


void DACE::buildDACE(bool change, int iter)
{
    
    gp=(double *)calloc((daceSpace->fSearchSpaceDim+1),sizeof(double));
    gtheta=(double *)calloc((daceSpace->fSearchSpaceDim+1),sizeof(double));
    
    double **param;
    param = (double **)calloc((fNoParamDACE+2),sizeof(double *));
    for(int i=0;i<=fNoParamDACE+1;i++)
        param[i]=(double *)calloc((fNoParamDACE+1),sizeof(double));
    
    double **bestparam;
    bestparam = (double **)calloc((fNoParamDACE+2),sizeof(double *));
    for(int i=0;i<=fNoParamDACE+1;i++)
        bestparam[i]=(double *)calloc((fNoParamDACE+1),sizeof(double));
    
    MyVector y = MyVector(fCorrelationSize, *pay);
    
    double best_lik=INFTY;
    int besti;
    // ME: changed from change=1
    if((change=1))
    {
        
        for(int i=0;i<30;i++)
        {
            for(int d=1; d<=daceSpace->fSearchSpaceDim; d++)
            {
                double rn1 = RN;
                //fprintf(stdout,"rn10 = %lg\n", rn1);
                double rn2 = RN;
                //fprintf(stdout, "rn11 = %lg\n", rn2);
                gtheta[d]=1+rn1*2;
                gp[d]=1.01+rn2*0.98;
                //////fprintf(stdout,"rn1=%lg rn2=%lg gtheta[d]=%lg gp[d]=%lg\n", rn1, rn2, gtheta[d], gp[d]);
            }
            //ME: Carefull with the indexes!!! It starts from 1.
            MyMatrix R(fCorrelationSize, fCorrelationSize);
            build_R(*pax, R);
            //DO NOT EVER COMMENT THIS OUT
            double detR = R.posdet();
            assert (detR > 0);
            //cout<< "THe MATRIX:"<<R<<"\n";
            pInvR = R.inverse();
            //cout<<"MATRIX INV IN CHANGE" <<pInvR<<"END_INV\n";
            gmu=mu_hat(y, iter);
            //fprintf(stdout,,"OK - mu %lg calculated\n", gmu);
            gsigma=sigma_squared_hat(y);
            //fprintf(stdout,,"OK - sigma %lg calculated\n", gsigma);
            
            for(int j=1;j<=daceSpace->fSearchSpaceDim;j++)
            {
                param[1][j]=gtheta[j];
                param[1][j+daceSpace->fSearchSpaceDim]=gp[j];
            }
            param[1][2*daceSpace->fSearchSpaceDim+1]=gsigma;
            param[1][2*daceSpace->fSearchSpaceDim+2]=gmu;
            
                                    //for (int i = 1; i < 2*daceSpace->fSearchSpaceDim+2+1; i++)
                                        //fprintf(stdout,, "%lg ", param[1][i]);
                                    //fprintf(stdout,, "\n");
            
            glik=likelihood(param[1], *pax, *pay);
            //fprintf(stdout,"glik= %lg best_lik= %lg  BETTER....\n", glik, best_lik);
            
            if(glik<best_lik)
            {
                //printf(stdout,"bestparam:\n");
                besti = i;
                best_lik=glik;
                for(int j=1;j<=fNoParamDACE;j++)
                {
                    bestparam[1][j]=param[1][j];
                    //printf(stdout,"%lg ", bestparam[1][j]);
                }
                //printf(stdout,"\n");
            }
            
        }
    }
    
    //fprintf(stdout,," theta_last[d]=%lg %lg p_last[d]=%lg %lg\n", gtheta[1], gtheta[2], gp[1], gp[2]);

    
    for (int i=1;i <= daceSpace->fSearchSpaceDim; i++)
    {
        gtheta[i]=bestparam[1][i];
        gp[i]=bestparam[1][i+daceSpace->fSearchSpaceDim];
    }
    gsigma=bestparam[1][2*daceSpace->fSearchSpaceDim+1];
    gmu=bestparam[1][2*daceSpace->fSearchSpaceDim+2];
    
    //fprintf(stdout,," theta_last[d]=%lg %lg p_last[d]=%lg %lg\n", gtheta[1], gtheta[2], gp[1], gp[2]);

//    fprintf(stdout,"FINAL DACE parameters = \n");
//    for(int d=1; d<=daceSpace->fSearchSpaceDim; d++)
//        fprintf(stdout,"%lg ", gtheta[d]);
//    for(int d=1; d<=daceSpace->fSearchSpaceDim; d++)
//        fprintf(stdout,"%lg ", gp[d]);
//    fprintf(stdout," %lg %lg\n", bestparam[1][2*daceSpace->fSearchSpaceDim+1], bestparam[1][2*daceSpace->fSearchSpaceDim+2]);
//
    
    //Choose the NMAX sample points to use
    
//    
    /* Use the full R matrix */
    //Actually use NMAX of the sample points.
    fCorrelationSize=iter;

    if (fCorrelationSize > 21*daceSpace->fSearchSpaceDim+24)
    {
        int ind[iter+1];
        Utilities::mysort(ind, daceSpace->fMeasuredFit, fCorrelationSize);
        for (int i=1; i <= fCorrelationSize; i++)
        {
            for(int d=1; d<= daceSpace->fSearchSpaceDim;d++)
            {
                daceSpace->fSelectedXVectors[i][d] = daceSpace->fXVectors[ind[i]][d];
                //printf("%lg", fSelectedXVectors[i][d]);
            }
            
            daceSpace->fSelectedMeasuredFit[i] = daceSpace->fMeasuredFit[ind[i]];
            //fprintf(stdout," tmpay: %lg", daceSpace->fSelectedMeasuredFit[i]);
        }
        fCorrelationSize = (fCorrelationSize > NMAX)? NMAX : fCorrelationSize;
    }
    //fprintf(stdout, "\n");
    
    pax = &daceSpace->fXVectors;
    
    //ME: Carefull with the indexes!!! It starts from 1.
    MyMatrix pgR(fCorrelationSize,fCorrelationSize);
    build_R(*pax, pgR);
    pgR.posdet();
    //long double det =  pgR.determinant();
    //printf("det = %lg\n",det);
    pInvR = pgR.inverse();
    //cout<<"MATRIX INV" <<InvR<<"\n";
    pgy = MyVector(fCorrelationSize, daceSpace->fMeasuredFit);
    /* ***************************** */
    
    //   fprintf(out,"predicted R matrix built OK:\n");
    
    gymin = INFTY;
    
    for(int i = 1;i < fCorrelationSize;i++)
        if(pgy(i)<gymin)
            gymin = pgy(i);
    
    //printf("gymin %lg \n", gymin);
    
    for(int i=0;i<=fNoParamDACE+1;i++)
        free(param[i]);
    free(param);
    
    for(int i=0;i<=fNoParamDACE+1;i++)
        free(bestparam[i]);
    free(bestparam);
}

//Evaluation functions
double DACE::standard_distribution(double z)
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

void DACE::init_gz()
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


double DACE::standard_density(double z)
{
    double psi;
    //fprintf(stdout, "z: %.20lg", z);
    
    //fprintf(stdout, "-z*z/2.0: %.20lg\n", -(z/2.0)*z);

    //fprintf(stdout, "exp in sden: %.20lg\n", exp(-(z*z)/2.0));
    //fprintf(stdout, "sqrt: %.20lg\n", 1/sqrt(2*PI));
    psi = (1/sqrt(2*PI))*exp(-(z*z)/2.0);
    return (psi);
    
}

double  DACE::expected_improvement(double yhat, double ymin, double s)
{
    double E;
    double sdis;
    double sden;
    if(s<=0)
        return 0;
    //fprintf(stdout, "expression: %.20lg\n", (ymin-yhat)/s);
    if((ymin-yhat)/s < -7.0)
        sdis = 0.0;
    else if((ymin-yhat)/s > 7.0)
    {
        sdis = 1.0;
    }
    else
        sdis = standard_distribution( (ymin-yhat)/s );
    //fprintf(stdout, "sdis: %.20lg\n", sdis);

//    fprintf(stdout, "yhat: %.20lg\n",yhat);
//    fprintf(stdout, "ymin: %.20lg\n",ymin);
//    fprintf(stdout, "ymin: %.20lg\n",s);
//
    sden = standard_density((ymin-yhat)/s);
    //fprintf(stdout,"sden: %.20lg\n", sden);

    E = (ymin - yhat)*sdis + s*sden;
    //fprintf(stdout, "E: %.20lg\n", (ymin-yhat));

    return E;
}


// to move in a evaluation!?
double DACE::wrap_ei(double *x, int iter)
{
    debug_iter = iter;
    //fprintf(stdout,"ITER5 %d\n", iter);
    for(int d=1;d<=daceSpace->fSearchSpaceDim; d++)
    {    (*pax)[fCorrelationSize+1][d] = x[d];
        
        //fprintf(stdout,"pax: %.20lg %.20lg\n", x[d], (*pax)[fCorrelationSize+1][d]);
    }
    double fit;
    // predict the fitness
    fit=predict_y(*pax);
    
    //fprintf(stdout,"predicted fitness in wrap_ei() = %.20lg\n", fit);
    
    
    // compute the error
    double ss;
    //ME: ss -least square error.
    ss=s2(*pax);
    //fprintf(stdout,"s^2 error in wrap_ei() = %.4lg\n", ss);
    //fprintf(stdout,"s^2 error in wrap_ei() = %.20lf %.20lf \n", x[1], ss);
    
    
    // compute the expected improvement
    double ei;
    ei=expected_improvement(fit, gymin, sqrt(ss));
    //fprintf(stdout,"-ei in wrap_ei() = %.20lg\n", -ei);
    
    
    for(int d=1; d <= daceSpace->fSearchSpaceDim; d++)
    {
        if((x[d]>daceSpace->fXMax[d])||(x[d]<daceSpace->fXMin[d]))
            ei=-1000;
    }
    
    ////fprintf(stdout,,"%.9lf\n", ei);
    
    //printf("WRAP_END\n");
    // return the expected improvement
    return(-ei);
}





