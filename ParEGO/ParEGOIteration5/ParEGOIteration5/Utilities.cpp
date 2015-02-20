//
//  Utilities.cpp
//  ParEGOIteration5
//
//  Created by Bianca Cristina Cristescu on 08/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//


#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <utility>
#include <iostream>

#include </home/cristina/eigen/Eigen/Dense>

#define RN rand()/(RAND_MAX+1.0)


namespace Utilities
{
    
    
    void static latin_hyp(double **ax, int iter, int dim, double* xmin,
                          double* xmax)
    {
        int v;
        
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
        //printf("latin:ax\n");
        for(int i=0; i<iter; i++)
        {
            for(int d = 0; d < dim; d++)
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
                ax[i+1][d+1] = L[d][v]+rn*((xmax[d+1]-xmin[d+1])/double(iter));
                //printf("%lg ", ax[i+1][d+1]);
            }
            
        }
        //printf("\n");

    }

    void static mysort(int *idx, double *val, int num)
    {
        std::vector< std::pair<double, int> > sorted;
        
        
        for(int i=1;i<=num;i++)
        {
            sorted.push_back(std::make_pair(val[i], i));
        }
        sort(sorted.begin(), sorted.end());
        
        for(int i=1;i<=num;i++)
            idx[i]=sorted[i-1].second;
        
    }
    
    // choose without replacement k items from n
    void static cwr(int **target, int k, int n)
    {
        int i,j;
        int l_t; //(length of the list at time t)
        
        if(k>n)
        {
            fprintf(stderr,"trying to choose k items without replacement from n but k > n!!\n");
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
    
    void static pr_sq_mat(Eigen::MatrixXd& m, int dim)
    {
        for (int i=0; i<dim; i++) {
            for (int j=0; j<dim; j++) {
                std::cout << m(i,j) << " ";
            }
            std::cout << "\n";
        }
    }
    
    void static pr_vec(Eigen::VectorXd& v, int dim)
    {
        for (int i=0; i<dim; i++)
            std::cout << v(i) << " ";
        std::cout << "\n";
    }
};
