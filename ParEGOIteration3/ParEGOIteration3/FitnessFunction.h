//
//  FitnessFunction.h
//  ParEGOIteration3
//
//  Created by Bianca Cristina Cristescu on 03/12/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __ParEGOIteration3__FitnessFunction__
#define __ParEGOIteration3__FitnessFunction__

#include <stdio.h>
#include <vector>
#include <math.h>

#define PI 3.141592653

class FitnessFunction
{
public:
   char* fName;
   unsigned fNobjs;          // number of objectives in the ff
   unsigned fDim;            // number of decision variables
   std::vector<double> fXmin; // minimum value for every decision variable
   std::vector<double> fXmax; // maximum value for every decisoon variable
   
   // There is no offset in the following arrays
   std::vector<double> fIdealObjective; // objective ideals
   std::vector<double> fWeightVector; // weight vector for each objective
   std::vector<double> fAbsoluteMax; // maximum of the absolute values of the
   // objective seach fDimension
   std::vector<double> fAbsoluteMin; // minimum of the absolute values of the
   // objective seach fDimension
   double alph = 1.0;
   
public:
   void SetParameters()
   {
      
      if(strcmp(fName,"f_vlmop2")==0) // fitness function name, as it will be given on command line
      {
         fNobjs = 2;  // number of objectives in the ff
         fDim = 2;    // number of decision variables
         fXmin = std::vector<double>(2, -2.0);
         fXmax = std::vector<double>(2,2.0);
         // set the weight vectors of each objective to anything, so long as it sums to 1
         fWeightVector = {0.9, 0.1};
         fAbsoluteMax = {1.0, 1.0};
         fAbsoluteMin = {0.0, 0.0};
         
      }
      else if(strcmp(fName,"f_dtlz1a")==0)
      {
         fNobjs = 2;
         fDim = 6;
         fXmin = std::vector<double>(fDim,0);
         fXmax = std::vector<double>(fDim,1);
         fIdealObjective = {0.0, 0.0};
         fWeightVector = {0.9, 0.1};
         fAbsoluteMax = {500.0, 500.0};
         fAbsoluteMin = {0.0, 0.0};
      }
      else if(strcmp(fName,"f_dtlz7a")==0)
      {
         fNobjs=3;
         fDim = 8;
         fXmin = std::vector<double>(fDim,0);
         fXmax = std::vector<double>(fDim,1);
         fIdealObjective = {0.0, 0.0};
         fWeightVector = {0.4,0.3,0.3};
         fAbsoluteMax = {1.0, 1.0, 30.0};
         fAbsoluteMin = {0.0, 0.0, 0.0};
      }
      else if((strcmp(fName,"f_dtlz2a")==0)||(strcmp(fName,"f_dtlz4a")==0))
      {
         fNobjs=3;
         fDim = 8;
         fXmin = std::vector<double>(fDim,0);
         fXmax = std::vector<double>(fDim,1);
         fIdealObjective = {0.0, 0.0, 0.0};
         fWeightVector = {0.4,0.3,0.3};
         fAbsoluteMax = {2.5, 2.5, 2.5};
         fAbsoluteMin = {0.0, 0.0, 0.0};
         if(strcmp(fName,"f_dtlz4a")==0)
            alph=100.0;
      }
      else if(strcmp(fName,"f_vlmop3")==0)
      {
         fNobjs=3;
         fDim = 2;
         fXmin = std::vector<double>(fDim, -3);
         fXmax = std::vector<double>(fDim, 3);
         fIdealObjective = {0.0, 0.0, 0.0};
         fWeightVector = {0.4,0.3,0.3};
         fAbsoluteMax= {10.0, 62.0, 0.2};
         fAbsoluteMin= {0.0, 15.0, -0.15};
      }
      else if(strcmp(fName,"f_oka1")==0)
      {
         fNobjs = 2;
         fDim = 2;
         fXmin = {6*sin(PI/12.0), 2*PI*sin(PI/12.0)};
         fXmax = {fXmin[1]+2*PI*cos(PI/12.0), 6*cos(PI/12.0)};
         fIdealObjective = {0.0, 0.0};
         fWeightVector = {0.9, 0.1};
         fAbsoluteMax= {8.0,5.0};
         fAbsoluteMin= {0.0,0.0};
      }
      else if(strcmp(fName,"f_oka2")==0)
      {
         fNobjs = 2;
         fDim = 3;
         fXmin = {-PI, -5, -5};
         fXmax = {PI, 5, 5};
         fIdealObjective = {0.0, 0.0};
         fWeightVector = {0.9, 0.1};
         fAbsoluteMax = {PI, 5.1};
         fAbsoluteMin = {-PI, 0.0};
      }
      else if(strcmp(fName,"f_kno1")==0)
      {
         fNobjs = 2;
         fDim = 2;
         fXmin = std::vector<double>(fDim, 0.0);
         fXmax = std::vector<double>(fDim, 3.0);
         fIdealObjective = {0.0, 0.0};
         fWeightVector = {0.9, 0.1};
         fAbsoluteMax = {20.0, 20.0};
         fAbsoluteMin = {0.0, 0.0};
      }
      else
      {
         fprintf(stderr,"Didn't recognise that fitness function\n");
         //sexit(0);
      }
      
      fDim=fDim*2+2;
   }
   
   /* HERE 2
    Below are the current fitness function definitions. Add your functions here.
    The function should take an x and y array and compute the value of the y objectives
    from the x decision variables. Note that the first decision variable is x[1] and t
    he first objective is y[1], i.e. there is an offset to the array. */
   
   void f_dtlz1a(double *x, double *y)
   {
      
      double g = 0.0;
      for(int i=2;i<=fDim;i++)
         g+= (x[i]-0.5)*(x[i]-0.5) - cos(2*PI*(x[i]-0.5)); // Note this is 20*PI in Deb's dtlz1 func
      g += fDim-1;
      g *= 100;
      
      
      y[1] = 0.5*x[1]*(1 + g);
      y[2] = 0.5*(1-x[1])*(1 + g);
   }
   
   void f_dtlz2a(double *x, double *y)
   {
      double g = 0.0;
      for(int i=3;i<=fDim;i++)
         g+=(x[i]-0.5)*(x[i]-0.5);
      
      
      y[1] = (1 + g)*cos(pow(x[1],alph)*PI/2)*cos(pow(x[2],alph)*PI/2);
      y[2] = (1 + g)*cos(pow(x[1],alph)*PI/2)*sin(pow(x[2],alph)*PI/2);
      y[3] = (1 + g)*sin(pow(x[1],alph)*PI/2);
   }
   
   void f_dtlz7a(double *x, double *y)
   {
      double g,h,sum;
      y[1]=x[1];
      y[2]=x[2];
      
      g = 0.0;
      for(int i=3;i<=fDim;i++)
      {
         g+=x[i];
      }
      g*=9.0/(fDim-fNobjs+1);
      g+=1.0;
      
      sum=0.0;
      for(int i=1;i<=fNobjs-1;i++)
         sum += ( y[i]/(1.0+g) * (1.0+sin(3*PI*y[i])) );
      h = fNobjs - sum;
      
      y[1]=x[1];
      y[2]=x[2];
      y[3]=(1 + g)*h;
   }
   
   void f_vlmop3(double *x, double *y)
   {
      y[1] = 0.5*(x[1]*x[1]+x[2]*x[2]) + sin(x[1]*x[1]+x[2]*x[2]);
      y[2] = pow(3*x[1]-2*x[2]+4.0, 2.0)/8.0 + pow(x[1]-x[2]+1, 2.0)/27.0 + 15.0;
      y[3] = 1.0 / (x[1]*x[1]+x[2]*x[2]+1.0) - 1.1*exp(-(x[1]*x[1]) - (x[2]*x[2]));
   }
   
   
   void f_oka1(double *x, double *y)
   {
      double x1p = cos(PI/12.0)*x[1] - sin(PI/12.0)*x[2];
      double x2p = sin(PI/12.0)*x[1] + cos(PI/12.0)*x[2];
      
      y[1] = x1p;
      y[2] = sqrt(2*PI) - sqrt(fabs(x1p)) + 2 * pow(fabs(x2p-3*cos(x1p)-3) ,0.33333333);
   }
   
   void f_oka2(double *x, double *y)
   {
      
      y[1]=x[1];
      y[2]=1 - (1/(4*PI*PI))*pow(x[1]+PI,2) + pow(fabs(x[2]-5*cos(x[1])),0.333333333) + pow(fabs(x[3] - 5*sin(x[1])),0.33333333);
   }
   
   void f_vlmop2(double *x, double *y)
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
   
   void f_kno1(double *x, double *y)
   {
      double f;
      double g;
      double c;
      
      
      c = x[1]+x[2];
      
      f = 20-( 11+3*sin((5*c)*(0.5*c)) + 3*sin(4*c) + 5 *sin(2*c+2));
      //  f = 20*(1-(fabs(c-3.0)/3.0));
      
      g = (PI/2.0)*(x[1]-x[2]+3.0)/6.0;
      
      y[1]= 20-(f*cos(g));
      y[2]= 20-(f*sin(g));
      
   }
   
   double Tcheby(double *vec)
   {
      // the augmented Tchebycheff measure with normalization
      int i;
      double sum;
      double diff;
      double d_max=-LARGE;
      double norm[MAX_K];
      double nfIdealObjective[MAX_K];
      
      sum=0.0;
      
      
      
      for(i=0;i<fNobjs;i++)
      {
         norm[i] = (vec[i]-fAbsoluteMin[i])/(fAbsoluteMax[i]-fAbsoluteMin[i]);
         nfIdealObjective[i] = fIdealObjective[i];
         diff = fWeightVector[i]*(norm[i]-nfIdealObjective[i]);
         sum += diff;
         if(diff>d_max)
            d_max = diff;
      }
      
      
      // fprintf(out, "d_max= %.5lf + 0.5 * sum= %.5lf\n", d_max, sum);
      return(d_max + 0.05*sum);
   }
   
   
   double Fit(double* x, double* ff)
   {
      
      if(strcmp(fName, "f_kno1")==0)
      {
         f_kno1(x, ff);
         return(Tcheby(&(ff[1])));
      }
      else if(strcmp(fName, "f_vlmop2")==0)
      {
         f_vlmop2(x, ff);
         return(Tcheby(&(ff[1])));
      }
      else if(strcmp(fName, "f_vlmop3")==0)
      {
         double T;
         f_vlmop3(x, ff);
         T=(Tcheby(&(ff[1])));
         return(T);
      }
      else if(strcmp(fName, "f_dtlz1a")==0)
      {
         double T;
         f_dtlz1a(x, ff);
         T=(Tcheby(&(ff[1])));
         return(T);
      }
      else if(strcmp(fName, "f_dtlz2a")==0)
      {
         double T;
         f_dtlz2a(x, ff);
         T=(Tcheby(&(ff[1])));
         return(T);
      }
      else if(strcmp(fName, "f_dtlz4a")==0)
      {
         double T;
         f_dtlz2a(x, ff); // this is called but with global variable alph set = 100.0
         T=(Tcheby(&(ff[1])));
         return(T);
      }
      else if(strcmp(fName, "f_dtlz7a")==0)
      {
         double T;
         f_dtlz7a(x, ff);
         T=(Tcheby(&(ff[1])));
         return(T);
      }
      else if(strcmp(fName, "f_oka1")==0)
      {
         double T;
         f_oka1(x, ff);
         T=(Tcheby(&(ff[1])));
         return(T);
      }
      else if(strcmp(fName, "f_oka2")==0)
      {
         f_oka2(x, ff);
         return(Tcheby(&(ff[1])));
      }
      else
      {
         fprintf(stderr, "Didn't recognise that fitness function.\n");
         exit(0);
      }
   }

   FitnessFunction(char* name) {
      fName = name;
      SetParameters();
   }
};



#endif /* defined(__ParEGOIteration3__FitnessFunction__) */
