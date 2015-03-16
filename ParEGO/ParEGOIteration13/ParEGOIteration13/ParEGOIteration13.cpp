/**
 * \class ParEGOIteration13
 *
 *
 * \brief The class run the ParEGO algorithm.
 *
 * \note Copyright (c) Joshua Knowles, October 2006. All rights reserved.
 *
 * \author (last to touch it) Bianca-Cristina Cristescu
 *
 * \version $Revision: 13
 *
 * \date $Date: 25/02/15.
 *
 */
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
 ******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <string>

#include "SearchSpace.h"
#include "WeightVector.h"
#include "DACE.h"
#include "GeneticAlgorithm.h"
#include "Utilities.cpp"

#define IMPROV_NO 300

using namespace std;
using namespace Eigen;

class universe{
    
private:
    SearchSpace* space = NULL; ///< The description of the function search space
    WeightVector* weights = NULL; ///< The weights for scalarizing the solution.
    DACE* model = NULL ; ///> The model that estimates the real function.
    
    int improvements[IMPROV_NO];
    int best_ever;
    
    //Debug and performance information.
    bool Debug;
    clock_t start, end;
    double cpu_time_used;
    
public:
    int MAX_ITERS; ///< Maximum iterations number.
    string fitfunc; ///< Function to be optimized.
    int iter; ///< Iteration counter.
    
public:
    universe();
    void init_ParEGO();
    void iterate_ParEGO();
    void setspace(const char* objectiveFunctionName);
    void setweights();
    void setDACE();
};

///  Creates the universe for the algorithm to run.
universe::universe()
{
    Debug=false;
    best_ever=1;
}

/// Initializes the weights for scalarization.
void universe::setweights()
{
    weights = new WeightVector(space->getNoObjectives());
}

/// Sets up the search space for the given function.
void universe::setspace(const char* objectiveFunctionName)
{
    space = new SearchSpace(objectiveFunctionName, MAX_ITERS);
}

/// Sets up the parameters for the DACE model of the given function.
void universe::setDACE()
{
    model = new DACE(space);
}

FILE* plotfile;

int main(int argc, char **argv)
{
    clock_t time1, time2;
    time1 = clock();

    universe U;
    
    unsigned int seed=47456536;
    srand(seed);
    U.MAX_ITERS = 250;

    // Arguments check.
    string function = "f_vlmop2";
    if(argc>2)
    {
        function = string(argv[2]);
        U.MAX_ITERS = atoi(argv[1]);
    }
    
    fprintf(stdout, "Optimizing function: %s\n", function.c_str());
    string filename = function+ "-obj-it7-" + to_string(U.MAX_ITERS) ;
    
    plotfile = fopen((filename + ".dat").c_str(), "w");
    
    U.setspace(function.c_str());
    U.setweights();
    U.setDACE();
    
    // Starts up ParEGO and does the latin hypercube, outputting these to a file
    U.init_ParEGO();
    
    int i = U.iter;
    while ( i < U.MAX_ITERS )
    {
        // Takes n solution/point pairs as input and gives 1 new solution
        // as output.
        U.iterate_ParEGO();
        i++;
    }
    
    time2 = clock();
    float diff1 ((float)time2-(float)time1);
    float seconds1 = diff1 / CLOCKS_PER_SEC;
    fprintf(stdout, "Execution time: %lg\n", seconds1);
}

/// Runs one iteration of the ParEGO algorithm.
void universe::iterate_ParEGO()
{
    int prior_it=0;
    int stopcounter=0;
    
    weights->changeWeights(iter, space->fWeightVectors);
    for(int i=1;i<=iter;i++)
    {
        space->fMeasuredFit[i] = space->tcheby(space->fCostVectors[i]);
        if(space->fMeasuredFit[i]<model->ymin)
        {
            model->ymin=space->fMeasuredFit[i];
            best_ever=i;
        }
    }

    model->buildDACE(iter);
    
    start = clock();
    
    // BEGIN GA code
    double best_imp=INFTY;
    std::vector<double> best_x(space->getSearchSpaceDimensions()+1);
    
    GeneticAlgorithm ga(20, space->getSearchSpaceDimensions());
    ga.run(space, model, iter, best_x, &best_imp);
    
    // END GA code
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    for(int d=1;d<=space->getSearchSpaceDimensions();d++)
    {
        space->fXVectors[iter+1][d] = best_x[d];
    }
    space->fMeasuredFit[iter+1]=space->fit(iter+1);
    
    
    printf("%d ", iter+1+prior_it);
    for(int d=1;d <=space->getSearchSpaceDimensions(); d++)
        printf("%lg ", space->fXVectors[iter+1][d]);
    printf("decision\n");

    printf("%d ", iter+1);
    for(int i=1;i<=space->getNoObjectives();i++)
    {
        printf( "%lg ", space->fCostVectors[iter+1][i]);
        fprintf(plotfile, "%lg ", space->fCostVectors[iter+1][i]);
    }
    fprintf(plotfile, "\n");
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
    
    iter++;
    
}

/// Initializes the latin hypercube with solutions.
void universe::init_ParEGO()
{
    int prior_it=0;
    
    iter=10*space->getSearchSpaceDimensions()+(space->getSearchSpaceDimensions()-1);
    do
    {
        for(int i=1; i<=iter;i++)
            improvements[i]=0;
        
        Utilities::latin_hyp(space->fXVectors, iter, space->getSearchSpaceDimensions(),
                             space->fXMin, space->fXMax);  // select the first solutions using the latin hypercube
        
        for(int i=1;i<=iter;i++)
        {
            space->fMeasuredFit[i]=space->fit(i);
            printf( "%d ", i+prior_it);
            for(int d=1;d<=space->getSearchSpaceDimensions();d++)
                printf( "%lg ", space->fXVectors[i][d]);
            printf("decision\n");
            //printf("\n");
            printf("%d ", i+prior_it);
            for(int k=1;k<=space->getNoObjectives();k++)
            {
                printf("%lg ", space->fCostVectors[i][k]);
                fprintf(plotfile, "%.9lg ", space->fCostVectors[i][k]);

            }
            fprintf(plotfile, "\n");
            printf("objective\n");
        }
    }while(0);
}




