//
//  GeneticAlgorithm.cpp
//  ParEGOIteration5
//
//  Created by Bianca Cristina Cristescu on 09/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#include "GeneticAlgorithm.h"

#include "Utilities.cpp"
#include "DACE.h"

#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <thread>
#include <future>
#include <mutex>

std::mutex g_lock;

double static myabs(double v)
{
    if(v>=0.0)
        return v;
    else
        return -v;
}


GeneticAlgorithm::GeneticAlgorithm(int pop_size, int dim)
{
    gapopsize = pop_size;
    chromosomedim = dim;
    popx = (double **)calloc((gapopsize+1),sizeof(double *));  // GA population for searching over modelled landscape
    for(int i=0;i<gapopsize+1;i++)
        popx[i]=(double *)calloc((chromosomedim+1),sizeof(double));
    
    popy = new double[gapopsize+1];
    
    mutx = new double[chromosomedim+1];
    muty = 0;
    parB = 0;
    parB = 0;
}

GeneticAlgorithm::~GeneticAlgorithm()
{
    for(int i=0;i<gapopsize+1;i++)
        free(popx[i]);
    free(popx);
    
    delete popy;
    delete mutx;
}


void GeneticAlgorithm::cross(double *child, double *par1, double *par2, SearchSpace* space)
{
    double xl, xu;
    double x1, x2;
    double alpha;
    double expp;
    double di=20;
    double beta;
    double betaq;
    double rnd;
    
    for(int d=1; d<=chromosomedim; d++)
    {
        
        /*Selected Two Parents*/
        
        xl = space->fXMin[d];
        xu = space->fXMax[d];
        
        /* Check whether variable is selected or not*/
        double rn = RN;
        //fprintf(stderr,"rn1 = %lg\n", rn);
        if(rn <= 0.5)
        {
            //printf("par1 %lg", par1[d]);
            //printf("par2 %lg", par2[d]);
            double a = myabs((double)par1[d] - par2[d]);
            //printf("abs %.9lg", a);
            if(a > 0.000001)
            {
                //printf("go to bed");
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


int GeneticAlgorithm::tourn_select(double *xsel, double **ax, double *ay, int iter, int t_size, SearchSpace* space)
{
    int *idx;
    double besty=INFTY;
    //ME: Uninitialised var!!!
    int bestidx = 0;
    Utilities::cwr(&idx, t_size, iter);                    // fprintf(stderr,"cwr\n");
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

void GeneticAlgorithm::mutate(double *x, SearchSpace* space)
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
        //fprintf(stderr,"rn3 = %lg\n", rn);
        if(rn<m_rate)
        {
            mut[i]=true;
            nmutations++;
        }
    }
    
    if(nmutations==0)
    {
        double rn = RN;
        //fprintf(stderr,"rn4 = %lg\n", rn);
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


void GeneticAlgorithm::run(SearchSpace* space, DACE* model, int iter, double* best_x, double* best_imp)
{
    // initialize using the latin hypercube method
    Utilities::latin_hyp(popx, gapopsize, chromosomedim, space->fXMin, space->fXMax);
    //printf("popx ");
//    for(int i = 1; i < gapopsize + 1; i++)
//        for (int j = 1; j < chromosomedim+1; j++)
//        {
//            printf("%lg ", popx[i][j]);
//            printf("\n");
//        }
//    
//   printf("popy ");
    for (int i=1; i<=gapopsize; i++)
    {
        popy[i] = model->wrap_ei(popx[i], iter);
        //printf("%lg ", popy[i]);

    }
    //fprintf(stderr,"POPY_END\n");

    
    // initialize with mutants of nearby good solutions
    int ind[iter+1];
    Utilities::mysort(ind, space->fMeasuredFit, iter);
    
    //   for(int i=1;i<=fCorrelationSize;i++)
    //	fprintf(stderr,"ay = %.5lf ; %.5lf\n", ay[ind[i]], ay[i]);
    
    //fprintf(stderr,"Begin_MUTATION");
    
    for (int i=1; i<=5; i++)
    {
        //	  fprintf(stderr,"parent = %.5lf\n", ay[ind[i]]);
        parA=ind[i];
        for(int d=1;d<=space->fSearchSpaceDim;d++)
            popx[i][d] = space->fXVectors[parA][d];
        mutate(popx[i], space);                                // fprintf(stderr,stderr, "mutate\n");
        //        for(int j=1; j < space->fSearchSpaceDim+1 ;j++)
        //            fprintf(stderr,"%lg ", popx[i][j]);
        popy[i] = model->wrap_ei(popx[i], iter);                   // fprintf(stderr,stderr, "evaluate\n");
        
    }
    double p_cross=0.2;
    //fprintf(stderr,"End_MUTATION");
    for (int i = 1; i <= 10000; i++)
    {
        
        double rn = RN;
        //fprintf(stderr, "rn12 = %lg\n", rn);
        if(rn < p_cross)
        {
            if(0.0235743<=rn && rn <= 0.0235744){
                assert(rn != 0.127397);
            }
            parA=tourn_select(mutx, popx, popy, gapopsize, 2, space);
            do
            {
                //                fprintf(stderr,"TOURN");
                //                for (int i=1; i<=gapopsize; i++)
                //                {
                //                    fprintf(stderr,"%lg ", popy[i]);
                //                }
                //                fprintf(stderr,"TOURN_END\n");
                
                parB=tourn_select(mutx, popx, popy, gapopsize, 2, space);
            }
            while(parB==parA);
            cross(mutx, popx[parA], popx[parB], space);
        }
        else
            parA=tourn_select(mutx, popx, popy, gapopsize, 2, space);//  fprintf(stderr, "parent selected\n");
        
        mutate(mutx, space);                             // fprintf(stderr, "mutate\n");
        muty = model->wrap_ei(mutx, iter);
        //  fprintf(stderr, "evaluate\n");
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
        if(popy[i]<*best_imp)
        {
            improved=true;
            *best_imp=popy[i];
            for(int d=1; d<=space->fSearchSpaceDim; d++)
                best_x[d]=popx[i][d];
        }
    }
    printf("ei= %lf\n", *best_imp);
    if(improved==false)
    {
        fprintf(stderr, "GA found no improvement\n");
        for(int d=1; d<=space->fSearchSpaceDim; d++)
        {
            best_x[d]=popx[1][d];
        }
        mutate(best_x, space);
    }
    
//    fprintf(stderr,"popx ");
//    for(int i = 1; i < gapopsize + 1; i++)
//        for (int j = 1; j < chromosomedim+1; j++)
//        {
//            fprintf(stderr,"%lg ", popx[i][j]);
//            fprintf(stderr,"\n");
//        }
//    
//    fprintf(stderr,"popy ");


}

