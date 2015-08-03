/**
 * \class WeightVector
 *
 *
 * \brief Representation of weight for scalarizinf multiple dimension solutions.
 *
 * This class models the weighting system that enables use to scalarize 
 * a multi-dimensional solution to be used in the optimization problem to create
 * a surrogate model function for the real expensive function.
 *
 * \note Copyright (c) 2006 Joshua Knowles. All rights reserved.
 *
 * \author (last to touch it) Bianca-Cristina Cristescu
 *
 * \version $Revision: 13
 *
 * \date $Date: 04/02/15.
 *
 */

#ifndef __ParEGOIteration13__WeightVector__
#define __ParEGOIteration13__WeightVector__

#include <vector>

class WeightVector
{
private:
    int N; ///< parameter - number of divisions
    std::vector<std::vector<int> > wv; ///< Weights vector. 
    std::vector<std::vector<double> > normwv; ///< Normalized weights.
    int wvSize; ///< Size of weight vectors.
    int fobjectives; ///< Number of objectives for function.
    int next; ///< Next iteration.
    int add; ///< Add/decrease the weights.
    bool change; ///< Change weights or not.
    
public:
    WeightVector(int objectivesNumber);
    void changeWeights(int iter, std::vector<double>& newvector);
    
private:
    void snake_wv();
    void reverse(int n);


};

#endif /* defined(__ParEGOIteration13__WeightVector__) */
