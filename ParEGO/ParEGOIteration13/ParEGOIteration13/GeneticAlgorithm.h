/**
 * \class Genetic Algorithm
 *
 *
 * \brief Evolve a solution using a Genetic Algorithm.
 *
 * This class is a genetic algorithm implementation which uses the following 
 * parameters:
 * Population size: 20 solutions.
 * Population update: steady state (one offspring produced per generation, from 
 * either a crossover or cloning event, followed by a mutation)
 * Generations/evaluations: 10,000 evaluations
 * Reproductive selection: binary tournament without replacement
 * Crossover: simulated binary crossover with probability 0.2,
 * producing one offspring
 * Mutation: decision value shifted by +/- 1/100 m.p, where m is drawn uniformly
 * at random from (0.0001, 1) , p is the range of the decision variable, 
 * and the per-gene mutation probability, is 1/d.
 * Replacement: offspring replaces (first) parent if it is better, else it is 
 * discarded
 * Initialization: 5 solutions are mutants of the 5 best solutions evaluated on 
 * the real fitness function under the prevailing lambda vector; the remaining
 * 15 solutions are generated in a latin hypercube in decision space.
 * Is the dominant time complexity part of the algorithm.
 *
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


#ifndef __ParEGOIteration13__GeneticAlgorithm__
#define __ParEGOIteration13__GeneticAlgorithm__

#include <vector>

#include "SearchSpace.h"

#define INFTY 1.0e35;

class DACE;

typedef std::vector<std::vector<double> > solutionType;

class GeneticAlgorithm
{
private:
    int gapopsize; ///< Genetic algorithm population size.
    int chromosomedim; ///< Dimnesion of the solution vector.
    solutionType popSolutions; ///< Decision variables.
    std::vector<double> popCost; ///< Cost vectors.
    std::vector<double> mutatedSolution; ///< Mutated solution.
    double mutatedCost; ///< Cost of mutated solution.
        
public:
    GeneticAlgorithm(int popSize, int dim);

    /** Run the genetic algorithm to evolve the solutions.
     * @param[in] space - The search space of the algorithm.
     * @param[in] dace - The DACE model created for the function.
     * @param[in] iter - The current iteration number.
     * @param[out] best_x - The best solution found.
     * @param[out] best_imp - The best cost found.
     */
    void run(SearchSpace* space, DACE* model, int iter,
             std::vector<double>& best_x, double* best_imp);

private:
    
    void cross(const std::vector<double>& parent1,
               const std::vector<double>& parent2,
               SearchSpace* space, std::vector<double>& child);
    int tourn_select(int t_size, SearchSpace* space);
    void mutate(std::vector<double>& mutatedSolution, SearchSpace* space);
    
};

#endif /* defined(__ParEGOIteration4__GeneticAlgorithm__) */

