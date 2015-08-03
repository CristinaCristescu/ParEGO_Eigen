/**
 * \class DACE
 *
 *
 * \brief The class represents the DACE model of an optimization function.
 *
 * This class builds the function model for the optimization function
 * and sets all the hyperparameters and provides the routines for evaluting a
 * sample points using the estimated function.
 *
 *
 * \note Copyright (c) 2006 Joshua Knowles. All rights reserved.
 *
 * \author (last to touch it) Bianca-Cristina Cristescu
 *
 * \version $Revision: 13
 *
 * \date $Date: 05/02/15.
 *
 */

#include <cstdlib>
#include <iostream>
#include <math.h>

#include "DACE.h"
#include "SearchSpace.h"

#define Debug false
#define PI 3.141592653
#define INFTY 1.0e35;
#define RN rand()/(RAND_MAX+1.0)

using namespace std;

/**
  * Creates the DACE model for the function characterized by the given search
  * space.
  *
  * @param[in] space - The search space for the function to be estimated.
  *
  */
DACE::DACE(SearchSpace* space):
gtheta(space->getSearchSpaceDimensions()),
gp(space->getSearchSpaceDimensions())
{
    daceSpace = space;
    fNoParamDACE=daceSpace->getSearchSpaceDimensions()*2+2;
    gmu = 0;
    gsigma = 0;
    gymin = INFTY;
    pInvR = MyMatrix();
    pgy = MyVector();
    glik = 0;
    ymin=INFTY;

    
    one_pInvR = MyMatrix();
    onetransp_pInvR = MyMatrix();
    onetransp_pInvR_one = 0;
    predict_y_constant = MyVector();
    
    init_gz();
}

double static myabs(double v)
{
    if(v>=0.0)
        return v;
    else
        return -v;
}

/**
  * Computes the weighted distance between to sample points 
  * in the correlation matrix.
  * 
  * @param[in] xi - Sample point.
  * @param[in] xj - Sample point.
  *
  */
double DACE::weighted_distance(const std::vector<double>& xi, const std::vector<double>& xj)
{
    double sum=0.0;
    
    double nxi[daceSpace->getSearchSpaceDimensions()+1];
    double nxj[daceSpace->getSearchSpaceDimensions()+1];
    
    for(int h=1;h<=daceSpace->getSearchSpaceDimensions();h++)
    {
        nxi[h] = (xi[h]-daceSpace->fXMin[h])/(daceSpace->fXMax[h]-daceSpace->fXMin[h]);
        nxj[h] = (xj[h]-daceSpace->fXMin[h])/(daceSpace->fXMax[h]-daceSpace->fXMin[h]);
        sum += gtheta[h]*pow(myabs(nxi[h]-nxj[h]),gp[h]);
    }
    if(Debug)
        fprintf(stderr, "sum: %.5lf", sum);
        return(sum);
}

/**
  * Computes the correlation between two sample points.
  *
  * @param[in] xi - Sample point.
  * @param[in] xj - Sample point.
  * @param[in] theta - The  activity parameter.
  * @param[in] p - The smoothness parameter.
  * @param[in] dim -The dimension of the correlation.
  *
  */
double DACE::correlation(const std::vector<double>& xi, const std::vector<double>& xj,
                         const std::vector<double>& theta, const std::vector<double>& p,
                         int dim)
{
    return exp(-weighted_distance(xi,xj));
}

/** 
  * Build a correlation matrix.
  *
  * @param[in] solutionVector - The solution vectors to build the matrix from.
  * @param[out] R - The correlation matrix.
  *
  */
void DACE::build_R(const std::vector<std::vector<double> >& solutionVector,
                   MyMatrix& R)
{
    // takes the array of x vectors, theta, and p, and returns the correlation matrix R.
    for(int i=0;i<fCorrelationSize;i++)
    {
        for(int j=0;j<fCorrelationSize;j++)
        {
            if(Debug)
                fprintf(stderr,"%.5lf %.5lf %.5lf  %.5lf %d\n", solutionVector[i][1],solutionVector[j][1],gtheta[1], gp[1], daceSpace->getSearchSpaceDimensions());
            R.insert(i,j,correlation(solutionVector[i+1], solutionVector[j+1], gtheta, gp, fCorrelationSize));
            if(Debug)
                fprintf(stderr,"%.5lf\n", R(i,j));
        }
        if(Debug)
            fprintf(stderr,"\n");
    }
    
}

/** 
  * Computes the unbiased predictor of y.
  *
  * @param[in] solutionVector - The vector of solutions.
  *
  */
double DACE::predict_y(const std::vector<std::vector<double> >& solutionVector)
{
    double y_hat;
    MyVector one(fCorrelationSize, 1);
    
    MyVector r(fCorrelationSize);
    for(int i=0;i<fCorrelationSize;i++)
    {
        r.insert(i, correlation(solutionVector[fCorrelationSize+1],solutionVector[i+1],gtheta,gp,daceSpace->getSearchSpaceDimensions()));
    }
    
    double intermidiate = ((r.transpose()*pInvR)*(predict_y_constant))(0,0);
    y_hat = gmu + intermidiate;
    
    return(y_hat);
    
}

/** 
  * Computes the mean squarred error of the predictor.
  *
  * @param[in] solutionVector - The solution vector.
  * 
  */
double DACE::s2(const std::vector<std::vector<double> >& solutionVector)
{
    double s2;
    
    MyVector r(fCorrelationSize);
    for(int i=0;i<fCorrelationSize;i++)
    {
        r.insert(i, correlation(solutionVector[fCorrelationSize+1],
                                solutionVector[i+1],gtheta,gp,
                                daceSpace->getSearchSpaceDimensions()));
    }
    double intermediate = (1- (r.transpose()*pInvR*r)(0,0) +
                           pow((1-(onetransp_pInvR*r)(0,0)),2)/
                           onetransp_pInvR_one);
    if (intermediate < 0)
        intermediate = myabs(intermediate);
    s2 = gsigma * intermediate;
    return(s2);
}

/** 
  * Computes teh mean of the stochastic process.
  *
  * @param[in] y - The y vector.
  * @param[in] iter - The current iteration.
  *
  */
double DACE::mu_hat(MyVector& y, int iter)
{
    double numerator, denominator;
    MyVector one(fCorrelationSize, 1);
    
    numerator = (one_pInvR*y)(0,0);
    denominator = (one.transpose()*pInvR*one)(0,0);
    return(numerator/denominator);
    
}

/**
 * Computes the standard deviation of the stochastic process.
 *
 * @param[in] y - The y vector.
 *
 */
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

/**
 * Computes the likelihood of the stochastic process.
 *
 * @param[in] param - The paramenters of teh model.
 * @param[in] solutionVectors - The solution vectors.
 * @param[in] measuredFit - The measured fit of the solutions.
 *
 */
double DACE::likelihood(const std::vector<double>& param,
                        const std::vector<std::vector<double> >& solutionVectors,
                        const std::vector<double>& measuredFit)
{
    double lik;
    
    // Constraint handling.
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
    for(int j=1;j<=daceSpace->getSearchSpaceDimensions();j++)
    {
        if(param[daceSpace->getSearchSpaceDimensions()+j]>=2.0)
        {
            sum+=param[daceSpace->getSearchSpaceDimensions()+j]-2.0;
            outofrange=true;
        }
        else if(param[daceSpace->getSearchSpaceDimensions()+j]<1.0)
        {
            sum+=-(param[daceSpace->getSearchSpaceDimensions()+j]-1.0);
            outofrange=true;
        }
    }
    if(outofrange)
        return(sum);
    
    double coefficient;

    double sigma=param[2*daceSpace->getSearchSpaceDimensions()+1];
    
    MyMatrix R(fCorrelationSize,fCorrelationSize);
    build_R(solutionVectors, R);
    
    MyVector y = MyVector(fCorrelationSize, measuredFit);
    
    double detR = R.posdet();
    
    coefficient = 1.0/(pow(2*PI,(double)fCorrelationSize/2.0)*pow(sigma,(double)fCorrelationSize/2.0)*sqrt(detR));
    
    lik = coefficient*exp(-(double)fCorrelationSize/2.0);
    
    return(-lik);
}

/**
  * Builds the entire DACE model for the given function, setting the parameters
  * computing the correlation matrix, the inverse of it and the y vector.
  *
  * @param[in] iter - The current iteration number.
  *
  */
void DACE::buildDACE(int iter)
{
    bool change;
    std::vector<std::vector<double> > param(fNoParamDACE+2, std::vector<double>(fNoParamDACE+1));
    std::vector<std::vector<double> > bestparam(fNoParamDACE+2, std::vector<double>(fNoParamDACE+1));
    double best_lik=INFTY;
    
    int besti;
    if((change=1))
    {
        
        for(int i=0;i<30;i++)
        {
            for(int d=1; d<=daceSpace->getSearchSpaceDimensions(); d++)
            {
                double rn1 = RN;
                double rn2 = RN;
                gtheta[d]=1+rn1*2;
                gp[d]=1.01+rn2*0.98;
            }
            MyVector y;
            MyMatrix R;
            if(iter>11*daceSpace->getSearchSpaceDimensions()+24)
            {
                fCorrelationSize = 11*daceSpace->getSearchSpaceDimensions()+24;
                
                daceSpace->chooseAndUpdateSolutions(iter, fCorrelationSize);
                
                R = MyMatrix(fCorrelationSize, fCorrelationSize);
                build_R(daceSpace->fSelectedXVectors, R);
                y = MyVector(fCorrelationSize, daceSpace->fSelectedMeasuredFit);
            }
            else
            {
                fCorrelationSize=iter;
                R = MyMatrix(fCorrelationSize, fCorrelationSize);
                build_R(daceSpace->fXVectors, R);
                y = MyVector(fCorrelationSize, daceSpace->fMeasuredFit);

            }
            
            double detR = R.posdet();
            assert (detR > 0);
            pInvR = R.inverse();
            MyVector one = MyVector(fCorrelationSize, 1);
            one_pInvR = one*pInvR;
            onetransp_pInvR_one = (one.transpose()*pInvR*one)(0,0);
            gmu=mu_hat(y, iter);
            //fprintf(stderr,"OK - mu %lg calculated\n", gmu);
            gsigma=sigma_squared_hat(y);
            //fprintf(stderr,"OK - sigma %lg calculated\n", gsigma);
            
            for(int j=1;j<=daceSpace->getSearchSpaceDimensions();j++)
            {
                param[1][j]=gtheta[j];
                param[1][j+daceSpace->getSearchSpaceDimensions()]=gp[j];
            }
            param[1][2*daceSpace->getSearchSpaceDimensions()+1]=gsigma;
            param[1][2*daceSpace->getSearchSpaceDimensions()+2]=gmu;
            
            if(iter>11*daceSpace->getSearchSpaceDimensions()+24)
            {
                glik=likelihood(param[1], daceSpace->fSelectedXVectors, daceSpace->fSelectedMeasuredFit);
            }
            else
            {
                glik=likelihood(param[1], daceSpace->fXVectors, daceSpace->fMeasuredFit);

            }
            
            if(glik<best_lik)
            {
                besti = i;
                best_lik=glik;
                for(int j=1;j<=fNoParamDACE;j++)
                {
                    bestparam[1][j]=param[1][j];
                }
            }
        }
    }
    
    for (int i=1;i <= daceSpace->getSearchSpaceDimensions(); i++)
    {
        gtheta[i]=bestparam[1][i];
        gp[i]=bestparam[1][i+daceSpace->getSearchSpaceDimensions()];
    }
    gsigma=bestparam[1][2*daceSpace->getSearchSpaceDimensions()+1];
    gmu=bestparam[1][2*daceSpace->getSearchSpaceDimensions()+2];
    
//    fprintf(stderr,"FINAL DACE parameters = \n");
//    for(int d=1; d<=daceSpace->getSearchSpaceDimensions(); d++)
//        fprintf(stderr,,"%lg ", gtheta[d]);
//    for(int d=1; d<=daceSpace->getSearchSpaceDimensions(); d++)
//        fprintf(stderr,,"%lg ", gp[d]);
//    fprintf(stderr,," %lg %lg\n", bestparam[1][2*daceSpace->getSearchSpaceDimensions()+1], bestparam[1][2*daceSpace->getSearchSpaceDimensions()+2]);
//    
    
    /* Use the full R matrix */
    fCorrelationSize=iter;
    MyMatrix pgR(fCorrelationSize,fCorrelationSize);
    build_R(daceSpace->fXVectors, pgR);
    pgR.posdet();
    pInvR = pgR.inverse();
    pgy = MyVector(fCorrelationSize, daceSpace->fMeasuredFit);
    MyVector one = MyVector(fCorrelationSize, 1);
    one_pInvR = one*pInvR;
    onetransp_pInvR = one.transpose()*pInvR;
    onetransp_pInvR_one = (one.transpose()*pInvR*one)(0,0);
    predict_y_constant = pgy-(one*gmu);
    
    gymin = INFTY;
    
    for(int i = 1;i < fCorrelationSize;i++)
        if(pgy(i)<gymin)
            gymin = pgy(i);
}

/**
  * Computes the standard distribution.
  *
  * @param[in] z - The point.
  *
  */
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

/** 
  * Computes the standard density.
  *
  * @param[in] z - The point.
  *
  */
double DACE::standard_density(double z)
{
    double psi;
    
    psi = (1/sqrt(2*PI))*exp(-(z*z)/2.0);
    return (psi);
    
}

/** 
  * Computes teh expected improvement of the current solution point.
  *
  * @param[in] yhat - The y prediction.
  * @param[in] ymin - The minimum y.
  * @param[in] s - Square of the errror.
  *
  */
double  DACE::expected_improvement(double yhat, double ymin, double s)
{
    double E;
    double sdis;
    double sden;
    if(s<=0)
        return 0;
    if((ymin-yhat)/s < -7.0)
        sdis = 0.0;
    else if((ymin-yhat)/s > 7.0)
    {
        sdis = 1.0;
    }
    else
        sdis = standard_distribution( (ymin-yhat)/s );

    sden = standard_density((ymin-yhat)/s);

    E = (ymin - yhat)*sdis + s*sden;
    return E;
}

/** 
  * Returns -(expected improvement), given the solution x.
  *
  * @param[in] x - The solution for which to compute the ei.
  * @param[in] iter - The current iteration.
  *
  */
double DACE::wrap_ei(double *x, int iter)
{
    
    for(int d=1;d<=daceSpace->getSearchSpaceDimensions(); d++)
    {
        (daceSpace->fXVectors)[fCorrelationSize+1][d] = x[d];
    }
    double fit;
    // predict the fitness
    fit=predict_y(daceSpace->fXVectors);
    
    // compute the error
    double ss;
    ss=s2(daceSpace->fXVectors);
    
    // compute the expected improvement
    double ei;
    ei=expected_improvement(fit, gymin, sqrt(ss));
    //fprintf(stderr,"-ei in wrap_ei() = %.4lg\n", -ei);
    
    for(int d=1; d <= daceSpace->getSearchSpaceDimensions(); d++)
    {
        if((x[d]>daceSpace->fXMax[d])||(x[d]<daceSpace->fXMin[d]))
            ei=-1000;
    }
    return(-ei);
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





