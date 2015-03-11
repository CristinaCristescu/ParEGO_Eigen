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


#include <vector>
#include <ctime>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>


#define INFTY 1.0e35;
// Use the Eigen library as a first trial for comparison and profiling.

#include "SearchSpace.h"
#include "WeightVector.h"
#include "DACE.h"
#include "GeneticAlgorithm.h"
#include "Utilities.cpp"

using namespace std;
using namespace Eigen;

class universe{
    
public:
    int MAX_ITERS;
    char fitfunc[100];
    int iter; // global iteration counter
    
private:
    SearchSpace* space ; // the model fo the search space->
    WeightVector* weights ;
    DACE* model ;
    
    int improvements[300];
    int best_ever;
    
    //Debug and performance information.
    bool Debug;
    clock_t start, end;
    double cpu_time_used;
    
public:
    universe(int x);
    void init_ParEGO();
    void iterate_ParEGO();
    void setspace(const char* objectiveFunctionName);
    void setweights();
    void setDACE();
};

universe::universe(int x)
{
    Debug=false;
    best_ever=1;
    MAX_ITERS = 250;
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
FILE* plotfile;

int main(int argc, char **argv)
{
    clock_t time1, time2;
    time1 = clock();
    
    universe U(1);
    
    unsigned int seed=47456536;
    srand(seed);
    
    U.MAX_ITERS = 250;
    
    if(argc>2)
    {
        function = string(argv[2]);
        U.MAX_ITERS = atoi(argv[1]);
    }

    string filename = "kno1-obj-it10-250";
    
    plotfile = fopen((filename + ".dat").c_str(), "w");
    
    string function = "f_kno1";
    U.setspace(function.c_str());
    U.setweights();
    U.setDACE();
    //if(argc>2)
    // sprintf(U.fitfunc, argv[2]);
    
    U.init_ParEGO(); // starts up ParEGO and does the latin hypercube, outputting these to a file
    
    int i = U.iter;
    while ( i < U.MAX_ITERS )
    {
        U.iterate_ParEGO(); // takes n solution/point pairs as input and gives 1 new solution as output
        i++;
    }
    
    time2 = clock();
    double diff1 ((double)time2-(double)time1);
    double seconds1 = diff1 / CLOCKS_PER_SEC;
    cout<<seconds1<<endl;
    
    fclose(plotfile);
    
}

void universe::iterate_ParEGO()
{
    
    int prior_it=0;
    int stopcounter=0;
    
    weights->changeWeights(iter, space->fWeightVectors);
    //fprintf(stdout, "%.2lf %.2lf weightvectors \n", space->fWeightVectors[0], space->fWeightVectors[1]);
    //fprintf(stdout, "fMEasureFIt\n");
    for(int i=1;i<=iter;i++)
    {
        space->fMeasuredFit[i] = space->Tcheby(&space->fCostVectors[i][1]);
        //fprintf(stdout,"%lg ", space->fMeasuredFit[i]);
        if(space->fMeasuredFit[i]<model->ymin)
        {
            model->ymin=space->fMeasuredFit[i];
            best_ever=i;
        }
    }
    //fprintf(stdout,"\n ymin: %lg\n", model->ymin);
    //fprintf(stdout,"best_ever: %d\n", best_ever);
    //chose the solutions to use to update the DACE model
    if(iter>11*space->fSearchSpaceDim+24)
    {
        model->fCorrelationSize = 11*space->fSearchSpaceDim+24;

        space->chooseUpdateSolutions(iter, model->fCorrelationSize);
        model->pax=&space->fSelectedXVectors;
        model->pay=&space->fSelectedMeasuredFit;
    }
    else
    {
        model->fCorrelationSize=iter;
        model->pax=&space->fXVectors;
        model->pay=&space->fMeasuredFit;
    }
    
    model->buildDACE(weights->change, iter);
    
    start = clock();
    
    // BEGIN GA code
    double best_imp=INFTY;
    double* best_x;
    best_x = (double*)calloc(space->fSearchSpaceDim+1, sizeof(double));
    
    //could change the GA not to be an object. have to think about adv and disadv
    // pop size inti 20
    GeneticAlgorithm ga(20, space->fSearchSpaceDim);
    ga.run(space, model, iter, best_x, &best_imp);
    
       // END GA code
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // timestr << iter << " " << cpu_time_used << endl;
    
    //fprintf(stdout, "ax\n");
    for(int d=1;d<=space->fSearchSpaceDim;d++)
    {
        space->fXVectors[iter+1][d]=best_x[d];
        //fprintf(stdout, "%lg ", space->fXVectors[iter+1][d]);
    }
    //fprintf(stdout, "\n");
    space->fMeasuredFit[iter+1]=space->myfit(iter+1, iter+1);
    
    
    fprintf(stdout, "%d ", iter+1+prior_it);
    for(int d=1; d <=space->fSearchSpaceDim; d++)
        fprintf(stdout, "%lg ", space->fXVectors[iter+1][d]);
    fprintf(stdout, "decision\n");
    fprintf(stdout,"%d ", iter+1);
    for(int i=1;i<=space->fNoObjectives;i++)
    {
        fprintf(stdout, "%lg ", space->fCostVectors[iter+1][i]);
        fprintf(plotfile, "%lg ", space->fCostVectors[iter+1][i]);

        //fprintf(plotfile, "%.5lf ", ff[iter+1][i]);
    }
    fprintf(plotfile, "\n");
    fprintf(stdout, "objective\n");
    
    //cout<<"ymin"<<model->ymin<<"\n";

    improvements[iter+1]=improvements[iter];
    if (space->fMeasuredFit[iter+1]>=model->ymin)
    {
        // fprintf(stdout,"No actual improver found\n");
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

void universe::init_ParEGO()
{
    int prior_it=0;
    
    iter=10*space->fSearchSpaceDim+(space->fSearchSpaceDim-1);
    do
    {
        for(int i=1; i<=iter;i++)
            improvements[i]=0;
        
        Utilities::latin_hyp(space->fXVectors, iter, space->fSearchSpaceDim,
                             space->fXMin, space->fXMax);  // select the first solutions using the latin hypercube
        
        for(int i=1;i<=iter;i++)
        {
            space->fMeasuredFit[i]=space->myfit(i,i);
            fprintf(stdout, "%d ", i+prior_it);
            for(int d=1;d<=space->fSearchSpaceDim;d++)
                fprintf(stdout, "%.9lg ", space->fXVectors[i][d]);
            fprintf(stdout, "decision\n");
            //printf("\n");
            fprintf(stdout, "%d ", i+prior_it);
            for(int k=1;k<=space->fNoObjectives;k++)
            {
                fprintf(stdout, "%.9lg ", space->fCostVectors[i][k]);
                fprintf(plotfile, "%.9lg ", space->fCostVectors[i][k]);

            }
            fprintf(plotfile, "\n");
            fprintf(stdout, "objective\n");
        }
    }while(0);
}




