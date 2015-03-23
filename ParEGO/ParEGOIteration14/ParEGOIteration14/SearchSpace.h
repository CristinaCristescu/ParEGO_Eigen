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


#ifndef __ParEGOIteration13__SearchSpace__
#define __ParEGOIteration13__SearchSpace__

#define MAX_K 5

#include <vector>

class DACE;

class SearchSpace
{
//change to private and make setters and getters
private:
    const char* fObjectiveFunctionName; ///< Name of the objective function.
    int fNoObjectives; ///< The number of objectivesS.
    int fSearchSpaceDim; ///< Dimension of the actual search space.
    std::vector<double> fAbsMax;
    ///<absolute values of the maximum and of each objective space dimension
    std::vector<double> fAbsMin;
    ///<absolute values of the minimum and of each objective space dimension
    std::vector<double> fIdealObjective; ///< Objective ideal points.
    double alph; ///< Power in the DTLZ2 and DTLZ4 test functions.

public:
    std::vector<double> fXMin; ///< Upper constraints on the search space.
    std::vector<double> fXMax; ///< Lower constraints on the search space.
    std::vector<std::vector<double> > fXVectors;
    ///< Two-dimensional vector of all solution vectors.
    std::vector<std::vector<double> > fSelectedXVectors;
    ///< Two dimensional vector of selected solutions vectors for hyperparameter
    std::vector<std::vector<double> > fCostVectors;
    ///< Two-dimensional vector of all multiobjective cost vectors.
    std::vector<double> fSelectedMeasuredFit; ///< Vector selected y fitnesses.
    std::vector<double> fMeasuredFit; ///< Vector of true/measured y fitnesses.
    std::vector<double> fWeightVectors; ///< Weight vectors.

public:
    SearchSpace(const char* function_name, int max_iter);
    void chooseAndUpdateSolutions(int iter, int correlation_size);
    double fit(int iterationNo);
    int getNoObjectives() { return fNoObjectives; }
    int getSearchSpaceDimensions() { return fSearchSpaceDim; }
    double tcheby(const std::vector<double>& vec);
    
private:
    void setSearch();
    void f_dtlz1a(const std::vector<double>& x, std::vector<double>& y);
    void f_dtlz2a(const std::vector<double>& x, std::vector<double>& y);
    void f_dtlz7a(const std::vector<double>& x, std::vector<double>& y);
    void f_vlmop3(const std::vector<double>& x, std::vector<double>& y);
    void f_oka1(const std::vector<double>& x, std::vector<double>& y);
    void f_oka2(const std::vector<double>& x, std::vector<double>& y);
    void f_vlmop2(const std::vector<double>& x, std::vector<double>& y);
    void f_kno1(const std::vector<double>& x, std::vector<double>& y);
};
#endif /* defined(__ParEGOIteration13__SearchSpace__) */
