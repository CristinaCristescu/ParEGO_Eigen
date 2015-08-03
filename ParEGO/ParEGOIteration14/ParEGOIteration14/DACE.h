/**
 * \class DACE
 *
 *
 * \brief The class represents the DACE model of an optimization function.
 *
 * This class builds the function model for the optimization function
 * and sets all the hyperparameters and provides the routines for evaluting a 
 * sample points using the estimated function.
 *
 *
 * \note Copyright (c) 2006 Joshua Knowles. All rights reserved.
 *
 * \author (last to touch it) Bianca-Cristina Cristescu
 *
 * \version $Revision: 13
 *
 * \date $Date: 05/02/15.
 *
 */

#ifndef __ParEGOIteration13__DACE__
#define __ParEGOIteration13__DACE__

#include <vector>
#include "Matrix.h"
#include "Vector.h"

class SearchSpace;
class MyMatrix;
class MyVector;

class DACE
{
private:
    SearchSpace* daceSpace; ///< The function space characteristics.
    int fCorrelationSize; ///< Size of the current correlation matrix.
    int fNoParamDACE; ///< Number of parameters in the DACE model
    double gmu; ///< Accroding to literature, global mean.
    double gsigma; ///< According to literature, global standard deviation.
    std::vector<double> gtheta; ///< According to literature, activity parameter.
    std::vector<double> gp; ///< According to literature, smoothness parameter.
    MyMatrix pInvR; ///< Inverse of the correlation matrix.
    MyVector pgy; ///< Vector fo the predicted y values.
    double glik; ///< Global likelyhood.
    double gz[76]; ///< Array holding the z values for the gaussian distribution
    
    // Constants to spped up the linear algebra operations for the function
    // repeated function evaluations.
    MyMatrix one_pInvR;
    MyMatrix onetransp_pInvR;
    double onetransp_pInvR_one;
    MyVector predict_y_constant;
    
public:
    double gymin; ///< Global minimum predicted estimation y.

    DACE(SearchSpace* space);
    void buildDACE(int iter);
    double wrap_ei(const std::vector<double>& x, int iter);
    int best_solution(int iter);
    
private:
    double weighted_distance(const std::vector<double>& xi,
                             const std::vector<double>& xj);
    double correlation(const std::vector<double>& xi,
                       const std::vector<double>& xj,
                       const std::vector<double>& theta,
                       const std::vector<double>& p,
                       int dim);
    double mu_hat(MyVector& y, int iter);
    double sigma_squared_hat(MyVector& y);
    
    double predict_y(const std::vector<std::vector<double> >& solutionVector);
    double s2(const std::vector<std::vector<double> >& solutionVector);
    void build_R(const std::vector<std::vector<double> >& solutionVector,
                 MyMatrix& R);
    double posdet(MyMatrix& R, int n);
    void init_gz();
    double standard_density(double z);
    double standard_distribution(double z);
    double expected_improvement(double yhat, double ymin, double s);
    double likelihood(const std::vector<double>& param,
                      const std::vector<std::vector<double> >& solutionVectors,
                      const std::vector<double>& measuredFit);
    
};

#endif /* defined(__ParEGOIteration13__DACE__) */
