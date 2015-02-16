//
//  SearchSpace.cpp
//  ParEGOIteration4
//
//  Created by Bianca Cristina Cristescu on 03/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#define PI 3.141592653

#include "SearchSpace.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// The actual test functions.

void SearchSpace::f_dtlz1a(double *x, double *y)
{
    
    
    double g = 0.0;
    for(int i=2;i<=fSearchSpaceDim;i++)
        g+= (x[i]-0.5)*(x[i]-0.5) - cos(2*PI*(x[i]-0.5)); // Note this is 20*PI in Deb's dtlz1 func
    g += fSearchSpaceDim-1;
    g *= 100;
    
    
    y[1] = 0.5*x[1]*(1 + g);
    y[2] = 0.5*(1-x[1])*(1 + g);
}

void SearchSpace::f_dtlz2a(double *x, double *y)
{
    double g = 0.0;
    for(int i=3;i<=fSearchSpaceDim;i++)
        g+=(x[i]-0.5)*(x[i]-0.5);
    
    
    y[1] = (1 + g)*cos(pow(x[1],alph)*PI/2)*cos(pow(x[2],alph)*PI/2);
    y[2] = (1 + g)*cos(pow(x[1],alph)*PI/2)*sin(pow(x[2],alph)*PI/2);
    y[3] = (1 + g)*sin(pow(x[1],alph)*PI/2);
}

void SearchSpace::f_dtlz7a(double *x, double *y)
{
    double g,h,sum;
    y[1]=x[1];
    y[2]=x[2];
    
    g = 0.0;
    for(int i=3;i<=fSearchSpaceDim;i++)
    {
        g+=x[i];
    }
    g*=9.0/(fSearchSpaceDim-fNoObjectives+1);
    g+=1.0;
    
    sum=0.0;
    for(int i=1;i<=fNoObjectives-1;i++)
        sum += ( y[i]/(1.0+g) * (1.0+sin(3*PI*y[i])) );
    h = fNoObjectives - sum;
    
    y[1]=x[1];
    y[2]=x[2];
    y[3]=(1 + g)*h;
}

void SearchSpace::f_vlmop3(double *x, double *y)
{
    y[1] = 0.5*(x[1]*x[1]+x[2]*x[2]) + sin(x[1]*x[1]+x[2]*x[2]);
    y[2] = pow(3*x[1]-2*x[2]+4.0, 2.0)/8.0 + pow(x[1]-x[2]+1, 2.0)/27.0 + 15.0;
    y[3] = 1.0 / (x[1]*x[1]+x[2]*x[2]+1.0) - 1.1*exp(-(x[1]*x[1]) - (x[2]*x[2]));
}

double static myabs(double v)
{
    if(v >= 0.0)
        return v;
    else
        return -v;
}

void SearchSpace::f_oka1(double *x, double *y)
{
    double x1p = cos(PI/12.0)*x[1] - sin(PI/12.0)*x[2];
    double x2p = sin(PI/12.0)*x[1] + cos(PI/12.0)*x[2];
    
    y[1] = x1p;
    y[2] = sqrt(2*PI) - sqrt(myabs(x1p)) + 2 * pow(myabs(x2p-3*cos(x1p)-3) ,0.33333333);
}

void SearchSpace::f_oka2(double *x, double *y)
{
//    printf("oka_2/n");
//    for (int j=1; j < fSearchSpaceDim+1; j++)
//    {
//        printf( "%.5lf ", x[j]);
//        printf( "%.5lf ", y[j]);
//    }
//    
//    printf("\n");
    
    y[1]=x[1];
    y[2]=1 - (1/(4*PI*PI))*pow(x[1]+PI,2) + pow(myabs(x[2]-5*cos(x[1])),0.333333333) + pow(myabs(x[3] - 5*sin(x[1])),0.33333333);
    
//    printf("after oka_2/n");
//    for (int j=1; j < fSearchSpaceDim+1; j++)
//    {
//        printf( "%.5lf ", x[j]);
//        printf( "%.5lf ", y[j]);
//    }
//    
//    printf("\n");
}

void SearchSpace::f_vlmop2(double *x, double *y)
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

void SearchSpace::f_kno1(double *x, double *y)
{
    double f;
    double g;
    double c;
    
    
    c = x[1]+x[2];
    
    f = 20-( 11+3*sin((5*c)*(0.5*c)) + 3*sin(4*c) + 5 *sin(2*c+2));
    //  f = 20*(1-(myabs(c-3.0)/3.0));
    
    g = (PI/2.0)*(x[1]-x[2]+3.0)/6.0;
    
    y[1]= 20-(f*cos(g));
    y[2]= 20-(f*sin(g));
    
}

void SearchSpace::init_arrays(int n)
{
    // using a 1 offset to make things easier for use with NR routines and matrix routines
    fXVectors = (double **)calloc((n+1),sizeof(double *));
    for(int i=0;i<n+1;i++)
        (fXVectors)[i]=(double *)calloc((fSearchSpaceDim+1),sizeof(double));
    fMeasuredFit=(double *)calloc((n+1),sizeof(double));
    
    fSelectedXVectors = (double **)calloc((n+1),sizeof(double *));
    for(int i=0;i<n+1;i++)
        (fSelectedXVectors)[i]=(double *)calloc((fSearchSpaceDim+1),sizeof(double));
    fSelectedMeasuredFit=(double *)calloc((n+1),sizeof(double));
    
    fCostVectors = (double **)calloc((n+2),sizeof(double *));
    for(int i=0;i<=n+1;i++)
        fCostVectors[i]=(double *)calloc((fSearchSpaceDim+1),sizeof(double));
}

/*
 To define a new fitness function, first provide information about
 it in the same way as done below for f_vlmop2. Then search for "HERE" again.
 */
void SearchSpace::SetSearch()
{
    
    if(strcmp(fObjectiveFunctionName,"f_vlmop2")==0) // fitness function name, as it will be given on command line
    {
        fNoObjectives = 2;  // number of objectives in the fCostVectors
        fSearchSpaceDim = 2;    // number of decision variables
        fXMin = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMax = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        for(int d=1;d<=fSearchSpaceDim;d++)
        {
            fXMin[d]=-2.0;  // give the minimum and maximum value of every decision variable (notice the array offset)
            fXMax[d]=2.0;
        }
        
        // There is no offset in the following arrays
        fIdealObjective[0]=0.0;  // set all objective ideal points to zero
        fIdealObjective[1]=0.0;
        fWeightVectors[0]=0.9;     // set the weight vectors of each objective to anything, so long as it sums to 1
        fWeightVectors[1]=0.1;
        fAbsMax[0]=1.0;  // give the absolute values of the maximum and minimum of each objective space fSearchSpaceDimension. If this is not known, then make sure you overestimate the extent of the space.
        fAbsMin[0]=0.0;
        fAbsMax[1]=1.0;
        fAbsMin[1]=0.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_dtlz1a")==0)
    {
        fNoObjectives = 2;
        fSearchSpaceDim = 6;
        fXMin = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMax = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        for(int d=1;d<=fSearchSpaceDim;d++)
        {
            fXMin[d]=0;
            fXMax[d]=1;
        }
        fIdealObjective[0]=0.0;
        fIdealObjective[1]=0.0;
        fWeightVectors[0]=0.9;
        fWeightVectors[1]=0.1;
        fAbsMax[0]=500.0;
        fAbsMin[0]=0.0;
        fAbsMax[1]=500.0;
        fAbsMin[1]=0.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_dtlz7a")==0)
    {
        fNoObjectives=3;
        fSearchSpaceDim = 8;
        fXMin = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMax = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        for(int d=1;d<=fSearchSpaceDim;d++)
        {
            fXMin[d]=0;
            fXMax[d]=1;
        }
        fIdealObjective[0]=0.0;
        fIdealObjective[1]=0.0;
        fIdealObjective[2]=0.0;
        fWeightVectors[0]=0.4;
        fWeightVectors[1]=0.3;
        fWeightVectors[2]=0.3;
        fAbsMax[0]=1.0;
        fAbsMin[0]=0.0;
        fAbsMax[1]=1.0;
        fAbsMin[1]=0.0;
        fAbsMax[2]=30.0;
        fAbsMin[2]=0.0;
    }
    else if((strcmp(fObjectiveFunctionName,"f_dtlz2a")==0)||(strcmp(fObjectiveFunctionName,"f_dtlz4a")==0))
    {
        fNoObjectives=3;
        fSearchSpaceDim = 8;
        fXMin = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMax = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        for(int d=1;d<=fSearchSpaceDim;d++)
        {
            fXMin[d]=0;
            fXMax[d]=1;
        }
        fIdealObjective[0]=0.0;
        fIdealObjective[1]=0.0;
        fIdealObjective[2]=0.0;
        fWeightVectors[0]=0.4;
        fWeightVectors[1]=0.3;
        fWeightVectors[2]=0.3;
        fAbsMax[0]=2.5;
        fAbsMin[0]=0.0;
        fAbsMax[1]=2.5;
        fAbsMin[1]=0.0;
        fAbsMax[2]=2.5;
        fAbsMin[2]=0.0;
        if(strcmp(fObjectiveFunctionName,"f_dtlz4a")==0)
            alph=100.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_vlmop3")==0)
    {
        fNoObjectives=3;
        fSearchSpaceDim = 2;
        fXMin = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMax = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        for(int d=1;d<=fSearchSpaceDim;d++)
        {
            fXMin[d]=-3;
            fXMax[d]=3;
        }
        fIdealObjective[0]=0.0;
        fIdealObjective[1]=0.0;
        fIdealObjective[2]=0.0;
        fWeightVectors[0]=0.4;
        fWeightVectors[1]=0.3;
        fWeightVectors[2]=0.3;
        fAbsMax[0]=10.0;
        fAbsMin[0]=0.0;
        fAbsMax[1]=62.0;
        fAbsMin[1]=15.0;
        fAbsMax[2]=0.2;
        fAbsMin[2]=-0.15;
    }
    else if(strcmp(fObjectiveFunctionName,"f_oka1")==0)
    {
        fNoObjectives = 2;
        fSearchSpaceDim = 2;
        fXMin = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMax = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMin[1]=6*sin(PI/12.0);
        fXMax[1]=fXMin[1]+2*PI*cos(PI/12.0);
        fXMin[2]=-2*PI*sin(PI/12.0);
        fXMax[2]=6*cos(PI/12.0);
        fIdealObjective[0]=0.0;
        fIdealObjective[1]=0.0;
        fWeightVectors[0]=0.9;
        fWeightVectors[1]=0.1;
        fAbsMax[0]=8.0;
        fAbsMin[0]=0.0;
        fAbsMax[1]=5.0;
        fAbsMin[1]=0.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_oka2")==0)
    {
        fNoObjectives = 2;
        fSearchSpaceDim = 3;
        fXMin = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMax = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMin[1]=-PI;
        fXMax[1]=PI;
        fXMin[2]=-5;
        fXMax[2]=5;
        fXMin[3]=-5;
        fXMax[3]=5;
        fIdealObjective[0]=0.0;
        fIdealObjective[1]=0.0;
        fWeightVectors[0]=0.9;
        fWeightVectors[1]=0.1;
        fAbsMax[0]=PI;
        fAbsMin[0]=-PI;
        fAbsMax[1]=5.1;
        fAbsMin[1]=0.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_kno1")==0)
    {
        fNoObjectives = 2;
        fSearchSpaceDim = 2;
        fXMin = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        fXMax = (double *)calloc((fSearchSpaceDim+1),sizeof(double));
        for(int d=1;d<=fSearchSpaceDim;d++)
        {
            fXMin[d]=0.0;
            fXMax[d]=3.0;
        }
        fIdealObjective[0]=0.0;
        fIdealObjective[1]=0.0;
        fWeightVectors[0]=0.9;
        fWeightVectors[1]=0.1;
        fAbsMax[0]=20.0;
        fAbsMin[0]=0.0;
        fAbsMax[1]=20.0;
        fAbsMin[1]=0.0;
    }
    else
    {
        fprintf(stderr,"Didn't recognise that fitness function\n");
        exit(0);
    }
    
    
    fNoParamDACE=fSearchSpaceDim*2+2;
}

// Function which applies the proper fit.
void SearchSpace::applyFit(int i, int j)
{
//    printf("before_applyfit/n");
//        for (int index=1; index < fSearchSpaceDim+1; index++)
//        {
//            printf( "%.5lf ", fXVectors[i][index]);
//            printf( "%.5lf ", fCostVectors[j][index]);
//        }
//    
//    printf("\n");
    
    if(strcmp(fObjectiveFunctionName, "f_kno1")==0)
    {
        f_kno1(fXVectors[i], fCostVectors[j]);
        
    }
    else if(strcmp(fObjectiveFunctionName, "f_vlmop2")==0)
    {
        f_vlmop2(fXVectors[i], fCostVectors[j]);
    }
    else if(strcmp(fObjectiveFunctionName, "f_vlmop3")==0)
    {
        double T;
        f_vlmop3(fXVectors[i], fCostVectors[j]);
    }
    else if(strcmp(fObjectiveFunctionName, "f_dtlz1a")==0)
    {
        double T;
        f_dtlz1a(fXVectors[i], fCostVectors[j]);
        
    }
    else if(strcmp(fObjectiveFunctionName, "f_dtlz2a")==0)
    {
        double T;
        f_dtlz2a(fXVectors[i], fCostVectors[j]);
        
    }
    else if(strcmp(fObjectiveFunctionName, "f_dtlz4a")==0)
    {
        double T;
        f_dtlz2a(fXVectors[i], fCostVectors[j]); // this is called but with global variable alph set = 100.0
       
    }
    else if(strcmp(fObjectiveFunctionName, "f_dtlz7a")==0)
    {
        double T;
        f_dtlz7a(fXVectors[i], fCostVectors[j]);
        
    }
    
    else if(strcmp(fObjectiveFunctionName, "f_oka1")==0)
    {
        double T;
        f_oka1(fXVectors[i], fCostVectors[j]);
        
    }
    else if(strcmp(fObjectiveFunctionName, "f_oka2")==0)
    {
        f_oka2(fXVectors[i], fCostVectors[j]);
    }
    else
    {
        fprintf(stderr, "Didn't recognise that fitness function.\n");
        exit(0);
    }

//    printf("after_applyfit/n");
//    for (int index=1; index < fSearchSpaceDim+1; index++)
//    {
//        printf( "%.5lf ", fXVectors[i][index]);
//        printf( "%.5lf ", fCostVectors[j][index]);
//    }
//
//    
//    printf("\n");
}

// Default Constructor
SearchSpace::SearchSpace()
{
    fObjectiveFunctionName = NULL;
}

// Function Name Constructor
SearchSpace::SearchSpace(const char* name, int max_iter)
{
    fObjectiveFunctionName = name;
    SetSearch();
    init_arrays(max_iter);
}