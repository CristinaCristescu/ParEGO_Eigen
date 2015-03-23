/**
 * \class SearchSpace
 *
 * \brief Class that describes the search space of the function to be optimized.
 *
 * This class contains all the necessary information about the function to be
 * optimised such as solutions, costs, minimum and maximum values of the
 * function and the fitness evaluation.
 *
 * \note Copyright (c) 2006 Joshua Knowles. All rights reserved.
 *
 * \author (last to touch it) Bianca-Cristina Cristescu
 *
 * \version $Revision: 13
 *
 * \date $Date: 03/02/15.
 *
 */

#define PI 3.141592653

#include "SearchSpace.h"
#include "Utilities.cpp"
#include "DACE.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>

#define LARGE 2147000000
#define MAX_K 5

/**
 * Creates the search space and initializes it.
 * This function sets all the environment for the function that is optimized.
 *
 * @param[in] name - The name of the function to optimize.
 * @param[in] maxIter - The number of iterations to be run.
 */
SearchSpace::SearchSpace(const char* name, int maxIter):
fAbsMax(MAX_K), fAbsMin(MAX_K), fIdealObjective(MAX_K), fWeightVectors(MAX_K),
fMeasuredFit(maxIter+1), fSelectedMeasuredFit(maxIter+1)
{
    fObjectiveFunctionName = name;
    setSearch();
    fXVectors.resize(maxIter+1, std::vector<double>(fSearchSpaceDim+1));
    fSelectedXVectors.resize(maxIter+1, std::vector<double>(fSearchSpaceDim+1));
    fCostVectors.resize(maxIter+2, std::vector<double>(fSearchSpaceDim+1));
    alph = 1.0;
}

/** 
 * Sets all the information about the function that is optimised.
 * :
 * - number of objectives
 * - dimensionality of the function.
 * - minimum/maximum function values.
 * - ideal objective values.
 * - weight vector values.
 * - absolute minimum/maximum of the objective space.
 *
 */
void SearchSpace::setSearch()
{
    // Fitness function name, as it will be given on command line
    if(strcmp(fObjectiveFunctionName,"f_vlmop2") == 0)
    {
        fNoObjectives = 2;  // number of objectives in the fCostVectors
        fSearchSpaceDim = 2;    // number of decision variables
        fXMin.resize(fSearchSpaceDim+1);
        fXMax.resize(fSearchSpaceDim+1);
        // Give the minimum and maximum value of every decision variable
        // (notice the array offset).
        for(int d = 1; d <= fSearchSpaceDim; d++)
        {
            fXMin[d] = -2.0;
            fXMax[d] = 2.0;
        }
        
        // There is no offset in the following arrays
        // Set all objective ideal points to zero
        fIdealObjective[0] = 0.0;
        fIdealObjective[1] = 0.0;
        // Set the weight vectors of each objective to anything,
        // so long as it sums to 1.
        fWeightVectors[0] = 0.9;
        fWeightVectors[1] = 0.1;
        // Give the absolute values of the maximum and minimum of each
        // objective space in fSearchSpaceDimension.
        // If this is not known, then make sure you overestimate the extent
        // of the space.
        fAbsMax[0] = 1.0;
        fAbsMin[0] = 0.0;
        fAbsMax[1] = 1.0;
        fAbsMin[1] = 0.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_dtlz1a") == 0)
    {
        fNoObjectives = 2;
        fSearchSpaceDim = 6;
        fXMin.resize(fSearchSpaceDim+1);
        fXMax.resize(fSearchSpaceDim+1);
        for(int d = 1; d <= fSearchSpaceDim; d++)
        {
            fXMin[d] = 0;
            fXMax[d] = 1;
        }
        fIdealObjective[0] = 0.0;
        fIdealObjective[1] = 0.0;
        fWeightVectors[0] = 0.9;
        fWeightVectors[1] = 0.1;
        fAbsMax[0] = 500.0;
        fAbsMin[0] = 0.0;
        fAbsMax[1] = 500.0;
        fAbsMin[1] = 0.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_dtlz7a") == 0)
    {
        fNoObjectives = 3;
        fSearchSpaceDim = 8;
        fXMin.resize(fSearchSpaceDim+1);
        fXMax.resize(fSearchSpaceDim+1);
        for(int d = 1; d <= fSearchSpaceDim; d++)
        {
            fXMin[d] = 0;
            fXMax[d] = 1;
        }
        fIdealObjective[0] = 0.0;
        fIdealObjective[1] = 0.0;
        fIdealObjective[2] = 0.0;
        fWeightVectors[0] = 0.4;
        fWeightVectors[1] = 0.3;
        fWeightVectors[2] = 0.3;
        fAbsMax[0] = 1.0;
        fAbsMin[0] = 0.0;
        fAbsMax[1] = 1.0;
        fAbsMin[1] = 0.0;
        fAbsMax[2] = 30.0;
        fAbsMin[2] = 0.0;
    }
    else if((strcmp(fObjectiveFunctionName,"f_dtlz2a") == 0)
             ||(strcmp(fObjectiveFunctionName,"f_dtlz4a") == 0))
    {
        fNoObjectives = 3;
        fSearchSpaceDim = 8;
        fXMin.resize(fSearchSpaceDim+1);
        fXMax.resize(fSearchSpaceDim+1);
        for(int d = 1; d <= fSearchSpaceDim; d++)
        {
            fXMin[d] = 0;
            fXMax[d] = 1;
        }
        fIdealObjective[0] = 0.0;
        fIdealObjective[1] = 0.0;
        fIdealObjective[2] = 0.0;
        fWeightVectors[0] = 0.4;
        fWeightVectors[1] = 0.3;
        fWeightVectors[2] = 0.3;
        fAbsMax[0] = 2.5;
        fAbsMin[0] = 0.0;
        fAbsMax[1] = 2.5;
        fAbsMin[1] = 0.0;
        fAbsMax[2] = 2.5;
        fAbsMin[2] = 0.0;
        if(strcmp(fObjectiveFunctionName,"f_dtlz4a") == 0)
            alph = 100.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_vlmop3") == 0)
    {
        fNoObjectives = 3;
        fSearchSpaceDim = 2;
        fXMin.resize(fSearchSpaceDim+1);
        fXMax.resize(fSearchSpaceDim+1);
        for(int d = 1; d <= fSearchSpaceDim; d++)
        {
            fXMin[d] = -3;
            fXMax[d] = 3;
        }
        fIdealObjective[0] = 0.0;
        fIdealObjective[1] = 0.0;
        fIdealObjective[2] = 0.0;
        fWeightVectors[0] = 0.4;
        fWeightVectors[1] = 0.3;
        fWeightVectors[2] = 0.3;
        fAbsMax[0] = 10.0;
        fAbsMin[0] = 0.0;
        fAbsMax[1] = 62.0;
        fAbsMin[1] = 15.0;
        fAbsMax[2] = 0.2;
        fAbsMin[2] = -0.15;
    }
    else if(strcmp(fObjectiveFunctionName,"f_oka1") == 0)
    {
        fNoObjectives = 2;
        fSearchSpaceDim = 2;
        fXMin.resize(fSearchSpaceDim+1);
        fXMax.resize(fSearchSpaceDim+1);
        fXMin[1] = 6*sin(PI/12.0);
        fXMax[1] = fXMin[1]+2*PI*cos(PI/12.0);
        fXMin[2] = -2*PI*sin(PI/12.0);
        fXMax[2] = 6*cos(PI/12.0);
        fIdealObjective[0] = 0.0;
        fIdealObjective[1] = 0.0;
        fWeightVectors[0] = 0.9;
        fWeightVectors[1] = 0.1;
        fAbsMax[0] = 8.0;
        fAbsMin[0] = 0.0;
        fAbsMax[1] = 5.0;
        fAbsMin[1] = 0.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_oka2") == 0)
    {
        fNoObjectives = 2;
        fSearchSpaceDim = 3;
        fXMin.resize(fSearchSpaceDim+1);
        fXMax.resize(fSearchSpaceDim+1);
        fXMin[1] = -PI;
        fXMax[1] = PI;
        fXMin[2] = -5;
        fXMax[2] = 5;
        fXMin[3] = -5;
        fXMax[3] = 5;
        fIdealObjective[0] = 0.0;
        fIdealObjective[1] = 0.0;
        fWeightVectors[0] = 0.9;
        fWeightVectors[1] = 0.1;
        fAbsMax[0] = PI;
        fAbsMin[0] = -PI;
        fAbsMax[1] = 5.1;
        fAbsMin[1] = 0.0;
    }
    else if(strcmp(fObjectiveFunctionName,"f_kno1") == 0)
    {
        fNoObjectives = 2;
        fSearchSpaceDim = 2;
        fXMin.resize(fSearchSpaceDim+1);
        fXMax.resize(fSearchSpaceDim+1);
        for(int d = 1; d <= fSearchSpaceDim; d++)
        {
            fXMin[d] = 0.0;
            fXMax[d] = 3.0;
        }
        fIdealObjective[0] = 0.0;
        fIdealObjective[1] = 0.0;
        fWeightVectors[0] = 0.9;
        fWeightVectors[1] = 0.1;
        fAbsMax[0] = 20.0;
        fAbsMin[0] = 0.0;
        fAbsMax[1] = 20.0;
        fAbsMin[1] = 0.0;
    }
    else
    {
        fprintf(stderr,"Didn't recognise that fitness function\n");
        exit(0);
    }
}

/**
  * Function selects the best solutions according to their fitness to be used
  * for the model function estimation.
  *
  * @param[in] iterationNo - The iteration number to sort for.
  * @param[in] correlationSize - The current size fo the correlation matrix 
  *
  */
void SearchSpace::chooseAndUpdateSolutions(int iterationNo, int correlationSize)
{
    // First half solutions are the fitest solutions.
    std::vector<int> ind(iterationNo+1);
    Utilities::mysort(ind, fMeasuredFit, iterationNo);
    for (int i = 1; i <= correlationSize/2; i++)
    {
        for(int d = 1; d <= fSearchSpaceDim; d++)
        {
            fSelectedXVectors[i][d] = fXVectors[ind[i]][d];
        }
        fSelectedMeasuredFit[i] = fMeasuredFit[ind[i]];
    }
    
    //Second half are chose at random.
    std::vector<int> choose;
    // cwr(k,n) - choose without replacement k items from n
    Utilities::cwr(choose,iterationNo-correlationSize/2,
                   iterationNo-correlationSize/2);
    for(int i = correlationSize/2+1; i <= correlationSize; i++)
    {
        // Asert for bounds from cwr.
        assert(choose[i-correlationSize/2]+correlationSize/2 <= iterationNo
               && 0 < choose[i-correlationSize/2]+correlationSize/2);
        int j = ind[choose[i-correlationSize/2]+correlationSize/2];
        for(int d = 1; d <= fSearchSpaceDim; d++){
            fSelectedXVectors[i][d] = fXVectors[j][d];
        }
        fSelectedMeasuredFit[i] = fMeasuredFit[j];
    }
}

/**
  * Returns the augmented Tchebycheff measure with normalization
  * used to scalarize the solution vectors of the function.
  *
  * @param[in] vec - The solution vector for which to apply the scalarization.
  *
  */
double SearchSpace::tcheby(const std::vector<double>& vec)
{
    // the augmented Tchebycheff measure with normalization
    int i;
    double sum;
    double diff;
    double d_max = -LARGE;
    double norm[MAX_K];
    double nideal[MAX_K];
    
    sum=0.0;
    
    for(i = 0; i < fNoObjectives; i++)
    {
        norm[i] = (vec[i+1]-fAbsMin[i])/(fAbsMax[i]-fAbsMin[i]);
        nideal[i] = fIdealObjective[i];
        diff = fWeightVectors[i]*(norm[i]-nideal[i]);
        sum += diff;
        if(diff > d_max)
            d_max = diff;
    }
    // fprintf(out, "d_max= %.5lf + 0.5 * sum= %.5lf\n", d_max, sum);
    return(d_max + 0.05*sum);
}

/**
 * Applies the corresponding fitness function evaluation.
 *
 * @param[in] iterationNo - The iteration for which to calculate the fitness.
 */
double SearchSpace::fit(int iterationNo)
{
    
    if(strcmp(fObjectiveFunctionName, "f_kno1") == 0)
    {
        f_kno1(fXVectors[iterationNo], fCostVectors[iterationNo]);
        
    }
    else if(strcmp(fObjectiveFunctionName, "f_vlmop2") == 0)
    {
        f_vlmop2(fXVectors[iterationNo], fCostVectors[iterationNo]);
    }
    else if(strcmp(fObjectiveFunctionName, "f_vlmop3") == 0)
    {
        f_vlmop3(fXVectors[iterationNo], fCostVectors[iterationNo]);
    }
    else if(strcmp(fObjectiveFunctionName, "f_dtlz1a") == 0)
    {
        f_dtlz1a(fXVectors[iterationNo], fCostVectors[iterationNo]);
        
    }
    else if(strcmp(fObjectiveFunctionName, "f_dtlz2a") == 0)
    {
        f_dtlz2a(fXVectors[iterationNo], fCostVectors[iterationNo]);
        
    }
    // this is called but with global variable alph set = 100.0
    else if(strcmp(fObjectiveFunctionName, "f_dtlz4a") == 0)
    {
        f_dtlz2a(fXVectors[iterationNo], fCostVectors[iterationNo]);
    }
    else if(strcmp(fObjectiveFunctionName, "f_dtlz7a") == 0)
    {
        f_dtlz7a(fXVectors[iterationNo], fCostVectors[iterationNo]);
        
    }
    else if(strcmp(fObjectiveFunctionName, "f_oka1") == 0)
    {
        f_oka1(fXVectors[iterationNo], fCostVectors[iterationNo]);
        
    }
    else if(strcmp(fObjectiveFunctionName, "f_oka2") == 0)
    {
        f_oka2(fXVectors[iterationNo], fCostVectors[iterationNo]);
    }
    else
    {
        fprintf(stderr, "Didn't recognise that fitness function.\n");
        exit(0);
    }
    
    return(tcheby(fCostVectors[iterationNo]));
}

// The actual test functions.
/* HERE 2
 Below are the current fitness function definitions. Add your functions here.
 The function should take an x and y array and compute the value of the y objectives
 from the x decision variables. Note that the first decision variable is x[1] and t
 he first objective is y[1], i.e. there is an offset to the array. */

void SearchSpace::f_dtlz1a(const std::vector<double>& x, std::vector<double>& y)
{
    
    
    double g = 0.0;
    for(int i=2;i<=fSearchSpaceDim;i++)
        // Note this is 20*PI in Deb's dtlz1 func.
        g+= (x[i]-0.5)*(x[i]-0.5) - cos(2*PI*(x[i]-0.5));
    g += fSearchSpaceDim-1;
    g *= 100;
    
    
    y[1] = 0.5*x[1]*(1 + g);
    y[2] = 0.5*(1-x[1])*(1 + g);
}

void SearchSpace::f_dtlz2a(const std::vector<double>& x, std::vector<double>& y)
{
    double g = 0.0;
    for(int i = 3; i <= fSearchSpaceDim; i++)
        g += (x.at(i)-0.5)*(x.at(i)-0.5);
    assert(y.size() >= 4);
    
    y[1] = (1 + g)*cos(pow(x.at(1),alph)*PI/2)*cos(pow(x.at(2),alph)*PI/2);
    y[2] = (1 + g)*cos(pow(x.at(1),alph)*PI/2)*sin(pow(x.at(2),alph)*PI/2);
    y[3] = (1 + g)*sin(pow(x.at(1),alph)*PI/2);
}

void SearchSpace::f_dtlz7a(const std::vector<double>& x, std::vector<double>& y)
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

void SearchSpace::f_vlmop3(const std::vector<double>& x, std::vector<double>& y)
{
    y[1] = 0.5*(x[1]*x[1]+x[2]*x[2]) + sin(x[1]*x[1]+x[2]*x[2]);
    y[2] = pow(3*x[1]-2*x[2]+4.0, 2.0)/8.0 + pow(x[1]-x[2]+1, 2.0)/27.0 + 15.0;
    y[3] = 1.0 / (x[1]*x[1]+x[2]*x[2]+1.0) - 1.1*exp(-(x[1]*x[1]) - (x[2]*x[2]));
}

void SearchSpace::f_oka1(const std::vector<double>& x, std::vector<double>& y)
{
    double x1p = cos(PI/12.0)*x[1] - sin(PI/12.0)*x[2];
    double x2p = sin(PI/12.0)*x[1] + cos(PI/12.0)*x[2];
    
    y[1] = x1p;
    y[2] = sqrt(2*PI) - sqrt(abs(x1p)) + 2 *
           pow(abs(x2p-3*cos(x1p)-3) ,0.33333333);
}


void SearchSpace::f_oka2(const std::vector<double>& x, std::vector<double>& y)
{
    y[1] = x[1];
    y[2] = 1 - (1/(4*PI*PI))*pow(x[1]+PI,2) +
         pow(abs(x[2]-5*cos(x[1])),0.333333333) +
         pow(abs(x[3] - 5*sin(x[1])),0.33333333);
}

void SearchSpace::f_vlmop2(const std::vector<double>& x, std::vector<double>& y)
{
    double sum1 = 0;
    double sum2 = 0;
    
    for(int i = 1; i <= 2; i++)
    {
        sum1 += pow(x[i]-(1/sqrt(2.0)),2);
        sum2 += pow(x[i]+(1/sqrt(2.0)),2);
    }
    
    y[1] = 1 - exp(-sum1);
    y[2] = 1 - exp(-sum2);
}

void SearchSpace::f_kno1(const std::vector<double>& x, std::vector<double>& y)
{
    double f;
    double g;
    double c;
    
    c = x[1]+x[2];
    
    f = 20-( 11+3*sin((5*c)*(0.5*c)) + 3*sin(4*c) + 5 *sin(2*c+2));

    g = (PI/2.0)*(x[1]-x[2]+3.0)/6.0;
    
    y[1]= 20-(f*cos(g));
    y[2]= 20-(f*sin(g));
}
