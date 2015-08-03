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
 ******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include <string>

#include "SearchSpace.h"
#include "WeightVector.h"
#include "DACE.h"
#include "GeneticAlgorithm.h"
#include "Utilities.cpp"

#define IMPROV_NO 300
#define INFTY 1.0e35;

using namespace std;
using namespace Eigen;

class ParEGO
{
private:
    SearchSpace* space; ///< The description of the function search space
    WeightVector* weights; ///< The weights for scalarizing the solution.
    DACE* model; ///> The model that estimates the real function.
    
    int improvements[IMPROV_NO];
    int best_ever; ///< Best solution throughout the iterations.
    
    //Debug and performance information.
    bool Debug;
    clock_t start, end;
    double cpu_time_used;
    FILE* plotfile;
    
public:
    int MAX_ITERS; ///< Maximum iterations number.
    int iter; ///< Iteration counter.
    
public:
    ParEGO();
    void init_ParEGO();
    void iterate_ParEGO();
    void setspace(const char* objectiveFunctionName);
    void setweights();
    void setDACE();
    void setPlotFile(string filename);
    
};

///  Creates the universe for the algorithm to run.
ParEGO::ParEGO()
{
    Debug=false;
    best_ever=1;
    MAX_ITERS = -1;
}

/// Initializes the weights for scalarization.
void ParEGO::setweights()
{
    weights = new WeightVector(space->getNoObjectives());
}

/// Sets up the search space for the given function.
void ParEGO::setspace(const char* objectiveFunctionName)
{
    space = new SearchSpace(objectiveFunctionName, MAX_ITERS);
}

/// Sets up the parameters for the DACE model of the given function.
void ParEGO::setDACE()
{
    model = new DACE(space);
}

/// Set up the file to plot the objectives.
void ParEGO::setPlotFile(string filename)
{
    plotfile = fopen((filename + ".dat").c_str(), "w");
}

/// Runs one iteration of the ParEGO algorithm.
void ParEGO::iterate_ParEGO()
{
    int prior_it = 0;
    int stopcounter = 0;
    
    // Update the weights in case it is needed.
    weights->changeWeights(iter, space->fWeightVectors);
    // Find best solution so far.
    best_ever = model->best_solution(iter);
    // Build the estimate function.
    model->buildDACE(iter);
    
    // BEGIN GA code
    std::vector<double> best_x(space->getSearchSpaceDimensions()+1);
    
    GeneticAlgorithm ga(20, space->getSearchSpaceDimensions());
    ga.run(space, model, iter, best_x);
    // END GA code
    
    for(int d = 1; d <= space->getSearchSpaceDimensions(); d++)
    {
        space->fXVectors[iter+1][d] = best_x[d];
    }
    space->fMeasuredFit[iter+1] = space->fit(iter+1);
    
    // Print decison variables and objectives for current iteration.
    printf("%d ", iter+1+prior_it);
    for(int d = 1; d <= space->getSearchSpaceDimensions(); d++)
        printf("%lg ", space->fXVectors.at(iter+1).at(d));
    printf("decision\n");
    
    printf("%d ", iter+1);
    for(int i = 1; i <= space->getNoObjectives(); i++)
    {
        printf( "%lg ", space->fCostVectors.at(iter+1).at(i));
        fprintf(plotfile, "%lg ", space->fCostVectors[iter+1][i]);
    }
    fprintf(plotfile, "\n");
    printf("objective\n");
    
    improvements[iter+1] = improvements[iter];
    if (space->fMeasuredFit[iter+1] >= model->gymin)
    {
        stopcounter++;
    }
    else
    {
        improvements[iter+1]++;
        model->gymin = space->fMeasuredFit[iter+1];
        stopcounter = 0;
        best_ever = iter+1;
    }
    
    iter++;
    
}

/// Initializes the latin hypercube with solutions.
void ParEGO::init_ParEGO()
{
    int prior_it = 0;
    
    iter = 10*space->getSearchSpaceDimensions()+
    (space->getSearchSpaceDimensions()-1);
    
    for(int i = 1; i <= iter; i++)
        improvements[i] = 0;
    
    Utilities::latin_hyp(space->fXVectors, iter,
                         space->getSearchSpaceDimensions(),
                         space->fXMin, space->fXMax);
    
    for(int i = 1; i <= iter; i++)
    {
        space->fMeasuredFit[i] = space->fit(i);
        printf( "%d ", i+prior_it);
        for(int d = 1; d <= space->getSearchSpaceDimensions(); d++)
            printf("%lg ", space->fXVectors[i][d]);
        printf("decision\n");
        
        printf("%d ", i+prior_it);
        for(int k = 1; k <= space->getNoObjectives(); k++)
        {
            printf("%lg ", space->fCostVectors[i][k]);
            fprintf(plotfile, "%.9lg ", space->fCostVectors[i][k]);
            
        }
        fprintf(plotfile, "\n");
        printf("objective\n");
    }
    
}

int main(int argc, char **argv)
{
    clock_t time1, time2;
    time1 = clock();
    
    // Create an instance of ParEGO.
    ParEGO U;
    
    // Get function to run and seed.
    string function;
    unsigned int seed = -1;
    
    // Arguments check.
    if(argc > 7)
    {
        fprintf(stderr, "Too many arguments.");
        fprintf(stderr, "Expected: -n <number of iterations> -f <function name> - s <seed>.");
        return 0;
    }
    else if (argc > 1)
    {
        int i = 1;
        while(i < argc)
        {
            string arg = argv[i++];
            if(arg == "-n")
            {
                assert(i < argc);
                arg = argv[i++];
                U.MAX_ITERS = atoi(arg.c_str());
            }
            else if(arg == "-f")
            {
                assert(i < argc);
                function = argv[i++];
            }
            else if(arg == "-s")
            {
                assert(i < argc);
                arg = argv[i++];
                seed = atoi(arg.c_str());
            }
            else
            {
                fprintf(stderr, "Unrecognised arguments.");
                fprintf(stderr, "Expected: -n <number of iterations> -f <function name> - s <seed>.");
            }
        }
    }
    //Default values for arguments not set.
    if (U.MAX_ITERS == -1)
    {
        U.MAX_ITERS = 250;
    }
    if (function.empty())
    {
        function = "f_vlmop2";
    }
    if (seed == -1)
    {
        seed = 47456536;
    }
    
    fprintf(stdout, "Number of iterations: %d\n", U.MAX_ITERS);
    fprintf(stdout, "Optimizing function: %s\n", function.c_str());
    fprintf(stdout, "Seedset to: %d\n", seed);
    
    string filename = function+ "-obj-it13-" + to_string(U.MAX_ITERS) ;
    U.setPlotFile(filename);
    
    srand(seed);
    
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
    float cpu_time_used = diff1 / CLOCKS_PER_SEC;
    fprintf(stdout, "Execution time: %lg\n", cpu_time_used);
}
