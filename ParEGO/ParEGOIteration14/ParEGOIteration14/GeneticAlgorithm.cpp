/*
 * \class GeneticAlgorithm
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


#include "GeneticAlgorithm.h"

#include "Utilities.cpp"
#include "DACE.h"

#include <cmath>
#include <algorithm>


using namespace std;

/** Creates an instance of the genetic algorithm.
 * @param popSize - The size of the population.
 * @param dim - The dimension of the solution vector.
 */
GeneticAlgorithm::GeneticAlgorithm(int pop_size, int dim):
popSolutions(pop_size+1, vector<double>(dim+1)), popCost(pop_size+1),
mutatedSolution(dim+1), mutatedCost(0)
{
    gapopsize = pop_size;
    chromosomedim = dim;
}

/**
 * Crossover operation.
 * @param[in] - The first parent to cross.
 * @param[in] - The second param to cross.
 * @param[in] - The search space parameters description.
 * @param[out] - The resulting child of the crossover operation.
 */
void GeneticAlgorithm::cross(const std::vector<double>& parent1,
                             const std::vector<double>& parent2,
                             SearchSpace* space, std::vector<double>& child)
{
    // Crossover operation paramenters.
    double xl, xu;
    double x1, x2;
    double alpha;
    double expp;
    double di=20;
    double beta;
    double betaq;
    double rnd;
    
    for(int d = 1; d <= chromosomedim; d++)
    {
        
        // Selected Two Parents.
        xl = space->fXMin[d];
        xu = space->fXMax[d];
        
        // Check whether variable is selected or not.
        double rn = RN;
        if(rn <= 0.5)
        {
            double a = abs((double)parent1[d] - parent2[d]);
            if(a > 0.000001)
            {
                if(parent2[d] > parent1[d])
                {
                    x2 = parent2[d];
                    x1 = parent1[d];
                }
                else
                {
                    x2 = parent1[d];
                    x1 = parent2[d];
                }
                
                // Find beta value
                if((x1 - xl) > (xu - x2))
                {
                    beta = 1 + (2*(xu - x2)/(x2 - x1));
                }
                else
                {
                    beta = 1 + (2*(x1-xl)/(x2-x1));
                    
                }
                
                // Find alpha
                expp = di + 1.0;
                
                beta = 1.0/beta;
                
                alpha = 2.0 - pow(beta,expp);
                
                if (alpha < 0.0)
                {
                    printf("ERRRROR %f %f %f\n",alpha,parent1[d],parent2[d]);
                    exit(-1);
                }
                
                rnd = RN;
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
                
                // Generating one child
                child[d] = 0.5*((x1+x2) - betaq*(x2-x1));
            }
            else
            {
                
                betaq = 1.0;
                x1 = parent1[d]; x2 = parent2[d];
                
                // Generating one child
                child[d] = 0.5*((x1+x2) - betaq*(x2-x1));
                
            }
            
            if (child[d] < xl) child[d] = xl;
            if (child[d] > xu) child[d] = xu;
        }
    }
}

/**
 * Tournament selection operation.
 * @param[in] t_size - Tournament size
 * @param[in] space - The search space parameters description.
 */
int GeneticAlgorithm::tourn_select(int t_size, SearchSpace* space)
{
    std::vector<int> idx;
    double besty = INFTY;
    int bestidx = 0;
    Utilities::cwr(idx, t_size, gapopsize);
    for(int i = 1; i <= t_size; i++)
    {
        if(popCost[idx[i]]<besty)
        {
            besty = popCost[idx[i]];
            bestidx = i;
        }
    }
    for(int d = 1; d <= space->getSearchSpaceDimensions(); d++)
    {
        mutatedSolution[d] = popSolutions[idx[bestidx]][d];
    }
    return(idx[bestidx]);
}

/**
 * Mutation operation.
 * This mutation always mutates at least one gene
 * (even if the m_rate is set to zero) this is to avoid duplicating points
 * which would cause the R matrix to be singular.
 *
 * @param[in] space - The search space parameters description.
 * @param[out] mutatedSolution - The resulting solution from the mutation.
 */
void GeneticAlgorithm::mutate(std::vector<double>& mutatedSolution,
                              SearchSpace* space)
{
    // Compute the mutation rate.
    double m_rate = 1.0/space->getSearchSpaceDimensions();
    double shift;
    bool mut[space->getSearchSpaceDimensions()+1];
    int nmutations = 0;
    for(int i = 1; i <= space->getSearchSpaceDimensions(); i++)
    {
        mut[i] = false;
        double rn = RN;
        if(rn < m_rate)
        {
            mut[i] = true;
            nmutations++;
        }
    }
    
    if(nmutations == 0)
    {
        double rn = RN;
        mut[1+(int)(rn*space->getSearchSpaceDimensions())] = true;
    }
    
    
    for(int d = 1; d <= space->getSearchSpaceDimensions(); d++)
    {
        if(mut[d])
        {
            double rn = RN;
            shift = (0.0001+rn)*(space->fXMax[d]-space->fXMin[d])/100.0;
            rn = RN;
            if(rn < 0.5)
                shift *= -1;
            
            
            mutatedSolution[d] += shift;
            
            if(mutatedSolution[d] > space->fXMax[d])
                mutatedSolution[d] = space->fXMax[d]-abs(shift);
            else if(mutatedSolution[d]<space->fXMin[d])
                mutatedSolution[d] = space->fXMin[d]+abs(shift);
        }
    }
}

/** Run the genetic algorithm to evolve the solutions.
 * @param[in] space - The search space of the algorithm.
 * @param[in] dace - The DACE model created for the function.
 * @param[in] iter - The current iteration number.
 * @param[out] best_x - The best solution found.
 */
void GeneticAlgorithm::run(SearchSpace* space, DACE* model, int iter,
                           std::vector<double>& best_x)
{
    // Parents
    int parentA;
    int parentB;

    // Initialize using the latin hypercube method
    Utilities::latin_hyp(popSolutions, gapopsize, chromosomedim, space->fXMin,
                         space->fXMax);
    
    // Compute the cost fo the solutions in the population.
    for (int i = 1; i <= gapopsize; i++)
    {
        popCost[i] = model->wrap_ei(popSolutions[i], iter);
    }
    
    // Initialize with mutants of nearby good solutions.
    std::vector<int> ind(iter+1);
    Utilities::mysort(ind, space->fMeasuredFit, iter);

    for (int i = 1; i <= 5; i++)
    {
        parentA = ind[i];
        for(int d = 1; d <= space->getSearchSpaceDimensions(); d++)
            popSolutions[i][d] = space->fXVectors[parentA][d];
        mutate(popSolutions[i], space);
        popCost[i] = model->wrap_ei(popSolutions[i], iter);
        
    }
    double p_cross = 0.2;
    for (int i = 1; i <= 10000; i++)
    {
        //Select Parent.
        double rn = RN;
        if(rn < p_cross)
        {
            parentA = tourn_select(2, space);
            do
            {
                parentB = tourn_select(2, space);
            }
            while(parentB == parentA);
            cross(popSolutions[parentA], popSolutions[parentB], space,
                  mutatedSolution);
        }
        else
            parentA = tourn_select(2, space);
        
        mutate(mutatedSolution, space);
        mutatedCost = model->wrap_ei(mutatedSolution, iter);
        if(mutatedCost<popCost[parentA])
        {
            for(int d = 1 ; d <= space->getSearchSpaceDimensions(); d++)
            {
                popSolutions[parentA][d] = mutatedSolution[d];
                
            }
            popCost[parentA] = mutatedCost;
        }
    }

    bool improved=false;
    double best_imp = INFTY;
    for(int i = 1; i <= gapopsize; i++)
    {
        if(popCost[i]<best_imp)
        {
            improved = true;
            best_imp = popCost[i];
            for(int d = 1; d <= space->getSearchSpaceDimensions(); d++)
                best_x[d] = popSolutions[i][d];
        }
    }
    fprintf(stdout, "ei= %lg\n", best_imp);
    if(improved == false)
    {
        fprintf(stderr, "GA found no improvement\n");
        for(int d = 1; d <= space->getSearchSpaceDimensions(); d++)
        {
            best_x[d] = popSolutions[1][d];
        }
        mutate(best_x, space);
    }
}

