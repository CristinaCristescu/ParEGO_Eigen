//
//  UtilDonnowhere.cpp
//  ParEGOIteration3
//
//  Created by Bianca Cristina Cristescu on 01/12/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//


#include <stdlib.h>
#include <vector>
#include <utility>
#include <math.h>

#include </Users/cristina/developer/eigen/eigen-eigen-1306d75b4a21/Eigen/Dense>

#define RN rand()/(RAND_MAX+1.0)
#define PI 3.141592653
#define Debug false

using namespace Eigen;

void cwr(int **target, int k, int n)  // choose without replacement k items from n
{
    int i,j;
    int l_t; //(length of the list at time t)
    
    if(k>n)
    {
        printf("trying to choose k items without replacement from n but k > n!!\n");
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
        j=(int)(RN*l_t);
        to[i]=from[j];
        from[j]=from[l_t-1];
        l_t--;
    }
}

// Why do we have a function for abs...?
double myabs(double v)
{
    if(v >= 0.0)
        return v;
    else
        return -v;
}

extern void mysort(int *idx, double *val, int num)
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

extern double standard_density(double z)
{
    double psi;
    
    psi = (1/sqrt(2*PI))*exp(-(z*z)/2.0);
    return (psi);
    
}

extern double standard_distribution(double z, double gz[76])
{
    double zv;
    int idx;
    if(z<0.0)
    {
        z *= -1;
        if(z>=7.5)
            zv = gz[75];
        else
        {
            idx = (int)(z*10);
            zv = gz[idx]+((10*z)-idx)*(gz[idx+1]-gz[idx]);
        }
        zv = 1-zv;
    }
    else if(z==0.0)
        zv = 0.5;
    else
    {
        if(z>=7.5)
            zv = gz[75];
        else
        {
            idx = (int)(z*10);
            zv = gz[idx]+((10*z)-idx)*(gz[idx+1]-gz[idx]);
        }
    }
    return(zv);
}


extern double expected_improvement(double yhat, double ymin, double s, double gz[76])
{
    double E;
    double sdis;
    double sden;
    if(s<=0)
        return 0;
    if((ymin-yhat)/s < -7.0)
        sdis = 0.0;
    else if((ymin-yhat)/s > 7.0)
        sdis = 1.0;
    else
        sdis = standard_distribution( (ymin-yhat)/s, gz);
    
    sden = standard_density((ymin-yhat)/s);
    
    E = (ymin - yhat)*sdis + s*sden;
    return E;
}

extern double weighted_distance(double *xi, double *xj, double *theta, double *p, int dim,
                                double* xmin, double* xmax)
{
    double sum=0.0;
    
    double nxi[dim+1];
    double nxj[dim+1];
    
    for(int h=1;h<=dim;h++)
    {
        nxi[h] = (xi[h]-xmin[h])/(xmax[h]-xmin[h]);
        nxj[h] = (xj[h]-xmin[h])/(xmax[h]-xmin[h]);
        
        sum += theta[h]*pow(myabs(nxi[h]-nxj[h]),p[h]);
        //      sum += 4.0*pow(myabs(xi[h]-xj[h]),2.0);     // theta=4 and p=2
        
    }
    if(Debug)
        printf("sum: %.5lf", sum);
    return(sum);
}


extern double correlation(double *xi, double *xj, double *theta, double *p, int dim, double* xmin, double* xmax)
{
    if(Debug)
        for(int d=1;d<=dim;d++)
            printf("CORRELATION: %.5lf %.5lf %.5lf %.5lf\n", xi[d],xj[d],theta[d],p[d]);
    return exp(-weighted_distance(xi,xj,theta,p,dim, xmin, xmax));
}

extern double predict_y(double **ax, MatrixXd& InvR, VectorXd& y, double mu_hat,
                        double *theta, double *p, int n, unsigned dim, double* xmin, double* xmax)
{
    double y_hat;
    //  Matrix InvR = R.Inverse();
    VectorXd one(n);
    for(int i=0;i<n;i++)
        one(i)=1;
    
    VectorXd r(n);
    for(int i=0;i<n;i++)
    {
        r(i) = correlation(ax[n+1],ax[i+1],theta,p,dim, xmin, xmax);
        //fprintf(out,"r[%di]=%.5lf ax[n+1][1]=%.5lf\n",i, r(i), ax[n+1][1]);
    }
    // fprintf(out,"\n");
    
    
    //ME: unsupported operation of Eigen to add a scalar coefficient-wise????
    //????????????????????
    //y_hat = mu_hat + r * InvR * (y - one*mu_hat);
    /*cout<<"r transp"<< r.transpose()<<"\n";
     cout<<"InvR"<<InvR<<"\n";
     cout <<"last" << y-(one*mu_hat) << "\n";*/
    double intermidiate = ((r.transpose()*InvR)*(y-(one*mu_hat)));
    //cout<< "intermidiate"<< intermidiate<<"\n";(double)((r.transpose()*InvR)*(y-(one*mu_hat)));
    y_hat = mu_hat + intermidiate;
    
    //fprintf(stderr,"y_hat=%f mu_hat=%f\n",y_hat, mu_hat);
    
    /*
     if((y_hat>100)||(y_hat<-100))
     {
     //      fprintf(out,"mu_hat=%.5lf theta=%.5lf p=%.5lf\n", mu_hat, theta[1], p[1]);
     for(int i=1;i<=n;i++)
     fprintf(out,"%.2f-%.2f log(r[i])=%le ", ax[n+1][1],ax[i][1] , log(r[i]));
     }
     */
    
    //ME: Free the memory from Eigen. Check if it is automatically freed.
    //~one;
    //~r;
    return(y_hat);
    
}

extern double s2(double **ax, double *theta, double *p, double sigma, int dim, int n, MatrixXd& InvR, double* xmin, double* xmax)
{
    if(Debug)
        printf("sigma: %f\n", sigma);
    double s2;
    //  Matrix InvR = R.Inverse();
    VectorXd one(n);
    for(int i=0;i<n;i++)
        one(i)=1;
    
    VectorXd r(n);
    for(int i=0;i<n;i++)
    {
        //fprintf(out,"theta=%.5lf p=%.5lf ax[n+1]=%.5lf, ax[i+1]=%.5lf\n", theta[1],p[1],ax[n+1][1],ax[i+1][1]);
        r(i) = correlation(ax[n+1],ax[i+1],theta,p,dim, xmin, xmax);
        //fprintf(out,"r[i]=%.5lf ",r(i));
    }
    
    //ME: Not sure still if that is dot product?????
    //???????????
    //s2 = sigma * (1 - r*InvR*r + pow((1-one*InvR*r),2)/(one*InvR*one) );
    //s2 = sigma * (1 - r*InvR*r + pow((1-one*InvR*r),2)/(one*InvR*one) );
    
    double f1 = (r.transpose()*InvR*r);
    double f2 = (one.transpose()*InvR*r);
    double f4  = 1.0000000000000 - f2;
    double f3 = (one.transpose()*InvR*one);
    double f5 = pow(f4,2)/f3;
    double f6 = myabs(1.0000-f1);
    double f7 = f6 + f5;
    //printf("%f %f %f %f %f %f %f\n", f1, f2, f4, f3, f5, f6, f7);
    s2 = sigma * f7;
    
    //ME: Free the memory from Eigen. Check if it is automatically freed.
    //~one;
    //~r;
    return(s2);
}

extern double wrap_ei(double *x, unsigned dim, int titer, double** pax, MatrixXd* pInvR,
                      double* gtheta, double* gp, double gsigma, double gymin,
                      double* xmin, double* xmax, VectorXd* pgy, double globalmu, double gz[76])
{
    for(int d=1; d <= dim; d++)
        (pax)[titer+1][d] = x[d];
    //fprintf(out,"%.5lf %.5lf\n", x[1], (*pax)[titer+1][1]);
    double fit;
    // predict the fitness
    fit=predict_y(pax, *pInvR, *pgy, globalmu, gtheta, gp, titer, dim, xmin, xmax);
    
    //fprintf(out,"predicted fitness in wrap_ei() = %lg\n", fit);
    
    // compute the error
    double ss;
    //ME: ss -least square error.
    ss=s2(pax, gtheta, gp, gsigma, dim, titer, *pInvR, xmin, xmax);
    // fprintf(out,"s^2 error in wrap_ei() = %lg\n", ss);
    //  fprintf(stderr,"%.9lf %.9lf ", x[1], ss);
    
    
    // compute the expected improvement
    double ei;
    ei=expected_improvement(fit, gymin, sqrt(ss), gz);
    // fprintf(out,"-ei in wrap_ei() = %lg\n", -ei);
    
    
    for(int d=1; d <= dim; d++)
    {
        if((x[d]>xmax[d])||(x[d]<xmin[d]))
            ei=-1000;
    }
    
    // fprintf(stderr,"%.9lf\n", ei);
    
    // return the expected improvement
    return(-ei);
}




static void cross(double *child, double *par1, double *par2, const unsigned dim,
                  const double* xmin, const double* xmax)
{
    double xl, xu;
    double x1, x2;
    double alpha;
    double expp;
    double di=20;
    double beta;
    double betaq;
    double rnd;
    
    for(int d=1; d<=dim; d++)
    {
        
        /*Selected Two Parents*/
        
        xl = xmin[d];
        xu = xmax[d];
        
        /* Check whether variable is selected or not*/
        if(RN <= 0.5)
        {
            if(myabs(par1[d] - par2[d]) > 0.000001)
            {
                if(par2[d] > par1[d])
                {
                    x2 = par2[d];
                    x1 = par1[d];
                }
                else
                {
                    x2 = par1[d];
                    x1 = par2[d];
                }
                
                /*Find beta value*/
                
                if((x1 - xl) > (xu - x2))
                {
                    beta = 1 + (2*(xu - x2)/(x2 - x1));
                }
                else
                {
                    beta = 1 + (2*(x1-xl)/(x2-x1));
                    
                }
                
                /*Find alpha*/
                
                expp = di + 1.0;
                
                beta = 1.0/beta;
                
                alpha = 2.0 - pow(beta,expp);
                
                if (alpha < 0.0)
                {
                    printf("ERRRROR %f %f %f\n",alpha,par1[d],par2[d]);
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
                
                /*Generating one child */
                child[d] = 0.5*((x1+x2) - betaq*(x2-x1));
            }
            else
            {
                
                betaq = 1.0;
                x1 = par1[d]; x2 = par2[d];
                
                /*Generating one child*/
                child[d] = 0.5*((x1+x2) - betaq*(x2-x1));
                
            }
            
            if (child[d] < xl) child[d] = xl;
            if (child[d] > xu) child[d] = xu;
        }
    }
}


static int tourn_select(double *xsel, double **ax, double *ay, int iter, int t_size,
                        unsigned dim)
{
    int *idx;
    double besty=INFINITY;
    //ME: Uninitialised var!!!
    int bestidx = 0;
    cwr(&idx, t_size, iter);                    // fprintf(stderr,"cwr\n");
    for(int i=1;i<=t_size;i++)
    {
        if(ay[idx[i]]<besty)
        {
            besty=ay[idx[i]];
            bestidx=i;
        }
    }
    for(int d=1;d<=dim;d++)
    {
        xsel[d] = ax[idx[bestidx]][d];
    }
    return(idx[bestidx]);
}


static void mutate(double *x, unsigned dim, const double* xmin, const double* xmax)
{
    // this mutation always mutates at least one gene (even if the m_rate is set to zero)
    // this is to avoid duplicating points which would cause the R matrix to be singular
    double m_rate=1.0/dim;
    double shift;
    bool mut[dim+1];
    int nmutations=0;
    for(int i=1;i<=dim;i++)
    {
        mut[i]=false;
        if(RN<m_rate)
        {
            mut[i]=true;
            nmutations++;
        }
    }
    if(nmutations==0)
        mut[1+(int)(RN*dim)]=true;
    
    
    for(int d=1;d<=dim;d++)
    {
        if(mut[d])
        {
            shift = (0.0001+RN)*(xmax[d]-xmin[d])/100.0;
            if(RN<0.5)
                shift*=-1;
            
            
            x[d]+=shift;
            
            if(x[d]>xmax[d])
                x[d]=xmax[d]-myabs(shift);
            else if(x[d]<xmin[d])
                x[d]=xmin[d]+myabs(shift);
        }
    }
}

extern void run_genetic(double best_imp, double* xmin, double* xmax,
                 double** popx, const int gapopsize, const unsigned dim,
                 double* popy, int iter, double* ay, double** ax,
                 double* mutx, double muty, double* best_x, MatrixXd* pInvR,
                 double* gtheta, double* gp, double gsigma, double gymin,
                 VectorXd* pgy, double globalmu, double gz[76])
{
    best_imp=INFINITY;
    int parA;
    int parB;
    
    
    // initialize with mutants of nearby good solutions
    
    int ind[iter+1];
    mysort(ind, ay, iter);
    
    //   for(int i=1;i<=titer;i++)
    //	printf("ay = %.5lf ; %.5lf\n", ay[ind[i]], ay[i]);
    
    
    for (int i=1; i<=5; i++)
    {
        //	  printf("parent = %.5lf\n", ay[ind[i]]);
        parA=ind[i];
        for(int d=1;d<=dim;d++)
            popx[i][d] = ax[parA][d];
        mutate(popx[i], dim, xmin, xmax);                              // fprintf(stderr, "mutate\n");
        popy[i] = wrap_ei(popx[i], dim, iter, ax, pInvR, gtheta, gp, gsigma,
                          gymin, xmin, xmax, pgy,globalmu, gz);                   // fprintf(stderr, "evaluate\n");
        
    }
    
    double p_cross=0.2;
    for (int i=1; i<=10000; i++)
    {
        if(RN < p_cross)
        {
            parA=tourn_select(mutx, popx, popy, gapopsize, 2, dim);
            do
                parB=tourn_select(mutx, popx, popy, gapopsize, 2, dim);
            while(parB==parA);
            cross(mutx, popx[parA], popx[parB], dim, xmin, xmax);
        }
        else
            parA=tourn_select(mutx, popx, popy, gapopsize, 2, dim);//  fprintf(stderr, "parent selected\n");
        mutate(mutx, dim, xmin, xmax);                             // fprintf(stderr, "mutate\n");
        muty = wrap_ei(mutx, dim, iter, ax, pInvR, gtheta, gp, gsigma,
                       gymin, xmin, xmax, pgy,globalmu, gz);                     //  fprintf(stderr, "evaluate\n");
        if(muty<popy[parA])
        {
            for(int d=1;d<=dim;d++)
                popx[parA][d]=mutx[d];
            popy[parA]=muty;
        }
    }
    
    bool improved=false;
    for(int i=1;i<=gapopsize; i++)
    {
        if(popy[i]<best_imp)
        {
            improved=true;
            best_imp=popy[i];
            for(int d=1; d<=dim; d++)
                best_x[d]=popx[i][d];
        }
    }
    printf("ei= %lf\n", best_imp);
    if(improved==false)
    {
        fprintf(stderr, "GA found no improvement\n");
        for(int d=1; d<=dim; d++)
        {
            best_x[d]=popx[1][d];
        }
        mutate(best_x, dim, xmin, xmax);
    }
    
}


