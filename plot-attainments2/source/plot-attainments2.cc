/*********************************************************************
plot-attainments2.cc

   Copyright (C) 2005, Joshua Knowles   j.knowles@manchester.ac.uk

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version. 

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details. 

   The GNU General Public License is available at:
      http://www.gnu.org/copyleft/gpl.html
   or by writing to: 
        The Free Software Foundation, Inc., 
        675 Mass Ave, Cambridge, MA 02139, USA.  


This program takes R approximation sets as input and
 outputs a number of uniformly distributed 
 points on the Jth (J<=R) attainment surface.
 Its purpose is to allow straightforward plotting 
 of (especially 3d) attainment surfaces in gnuplot.

The method of computing the points is inspired by the article:
 Smith, Everson and Fieldsend (2004) "Dominance measures for multiobjective 
 simulated annealing", Proc. CEC 2004, IEEE Press, pp. 23-30.

For each point constructed on a lattice, the algorithm
 checks how many surfaces strictly 
 dominate it, and how many just dominate it 
 (including weakly and strictly). If the
 desired attainment surface is the Jth one, 
 then if #strict<J and #dom>=J then the
 point is plotted, else it is "discarded".

-----------------------------------------------------------------------
 
USAGE :

To Compile:
$ g++ plot-attainments2.cc -o plot-att -lm -O3

To Run:
$ ./plot-att <input-file> [-b <boundfile>] [-r <resolution>] [-k <#objectives> [-mm <{-1|1}> <{-1|1}> ...<{-1|1}>] [-a <attainment_surface>]

  where:
   <input-file> is the name of a file of approximation sets.
     The format for this file is one point (vector) per line, and each point described
     by space-separated real numbers. Each approximation set must be separated by a blank line.
     E.g.:
      1.0 3.0 4.0
      0.5 2.2 6.1

      3.0 2.0 1.0
      1.3 4.3 0.7
      0.2 1.5 4.5

      is a valid input file for a 3-objective problem, consisting of two appoximation sets.
   
   <boundfile>
      is the name of a file that specifies the ideal and nadir points. Without this parameter, these will
      be computed from the collection of approximation sets that are input. However, to allow plotting of 
      different attainment surfaces on the same axes (with all points uniformly distributed),
      it is best to use this boundfile option.
      The format for the file is:
      <min1> <min2> <min3> ... <minK>
      <max1> <max2> <max3> ... <maxK>

   <resolution>
      is an integer, specifying how many points in each objective will make up the lattice.
      E.g. if resolution R=30, and the number of objectives K is 3, then 30^(K-1) = 900 points
      will be projected onto each of the three objectives in turn, so the total number of 
      points that will potentially land on the attainment surface is K*(R^(K-1)) = 2700 points.
      If resolution is not specified, a default value of 24 is used.
     
   <#objectives>
      is an integer specifying the number of objectives K in the input file. If this is not set, the
      default value of 3 is used.

   [-mm <{-1|1}> ... <{-1|1}>]
      specifies whether each objective is being minimized (-1) or maximized (1). The -mm switch
      MUST come AFTER the number of objectives parameter on the command line and must be followed 
      by exactly K parameters.

   <attainment_surface>
      is an integer specifying which of the attainment surfaces to plot. The range of this parameter
      is [1,...,N] where N is the number of approximation sets in the input file.
 
Output:
   The output to stdout is a set of points on the Jth attainment surface. If there are 3 objectives,
   then these can be plot is gnuplot using:
   gnuplot> splot "outfile.dat"

   To make nice postscript figures, I recommend using: 
   gnuplot> set ls 1 pt 10 ps 0.5
   gnuplot> set terminal postscript 18
   gnuplot> set view 70 100
   gnuplot> set output "figure.ps"
   gnuplot> splot "outfile.dat"

------------------------------------------------------------------------------------

IMPORTANT NOTE:

Each approximation set must contain internally nondominated points only ! 
 Please run a filter on the sets first to remove the dominated points. 

------------------------------------------------------------------------------------

Example:

If you downloaded this file as part of an archive then you should have
the data file testminmax.dat which consists of 21 approximation sets of
a 3-objective mixed minimization/maximization problem.

Do:
$ ./plot-att testminmax.dat -k 3 -mm 1 -1 1 -r 60 -a 1 > att_1.dat

to obtain the first (best) attainment surface. Change the -a option
for other attainment surfaces.

To plot the output file in gnuplot, do:

gnuplot> set term x11
gnuplot> set xlabel "maximize f1"
gnuplot> set ylabel "minimize f2"
gnuplot> set zlabel "maximize f3"
gnuplot> set view 70, 80
gnuplot> splot "att_1.dat", ..., "att_21.dat"


*************************************************************************/

#include <ctime>
#include <iostream>       
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define MAX_LINE_LENGTH 1024
#define MAX_STR_LENGTH 200
#define RN rand()/(RAND_MAX+1.0)
#define LARGE 2147483647
#define VLARGE 1.0e99
#define mydalloc(X, n) if((X=(double*)malloc(n *sizeof(double)))==NULL){fprintf(stderr,"malloc error\n");exit(1);}

typedef struct dnode
{
  double *o;
  int num_strict; // point is strictly dominated by this many surfaces
  int num_dom; // point is dominated (including weakly and strictly) by this many surfaces
}P;

P *po;

FILE *fp;

int nobjs;
int d=1; // the dimension on which to sort
int compare(const void *i, const void *j);
int compdoub(const void *i, const void *j);
int dominates(double *a, double *b, int *minmax, int n);
void  check_file(FILE  *fp, int  *no_runsp, int  *tot_pointsp);
void  read_file(FILE  *fp, int  *no_pointsp, P *po);

int main(int argc, char *argv[])
{
  int resol=24; // default resolution
  int att_surf=1;  // default attainment surface is the best one
  int dim=3; // default number of objectives is 3
  nobjs=dim;
  int nr=0; // number of runs (i.e. approximation sets)
  int num;  // number of input points in total
  int nf; // number of input points in one file i.e approximation set

  double *q; // array of size nobjs
  double *r; // array of size nobjs
  int *minmax; // array of size nobjs
  double *min;// array of size nobjs
  double *max;// array of size nobjs
  double *inter; // array of size number of runs +1
  double *v;// array of size nobjs

  char paramfile[100];


  if(argc==1)
    {
      fprintf(stderr,"Usage: $./sample <file> [-b <boundfile>] [-r <resolution>] [-k <#objectives> [-mm <{-1|1} > <{-1|1}> .. <{-1|1}>]]  [-a <attainment_surface>]\n");
      exit(1);
    }

  bool paramflag=false;
  int nextarg=2;
  while((argc > 2)&&(nextarg<argc))
    {
      if(strcmp(argv[nextarg],"-b")==0)
	{
	  sprintf(paramfile, argv[++nextarg]);
	  paramflag=true;
	}
      else if(strcmp(argv[nextarg],"-r")==0)
	{
	  resol = atoi(argv[++nextarg]);
	}
      else if(strcmp(argv[nextarg],"-k")==0)
	{
	  nobjs = dim = atoi(argv[++nextarg]);
	  if((minmax=(int *)malloc(nobjs*sizeof(int)))==NULL){fprintf(stderr,"malloc error.\n");exit(1);}
	  for(int j=0;j<dim;j++)
	    minmax[j]=-1;
	}
     else if(strcmp(argv[nextarg],"-a")==0)
	{
	  att_surf = atoi(argv[++nextarg]);
	}
      else if(strcmp(argv[nextarg],"-mm")==0)	
	{
	  for(int i=0;i<nobjs;i++)
	    {
	      minmax[i]=atoi(argv[++nextarg]);
	    }
	}
      else
	{
	  fprintf(stderr,"Unrecognised command line parameter.\n");
	  fprintf(stderr,"Usage: $./sample <file> [-b <boundfile>] [-r <resolution>] [-k <#objectives>] [-a <attainment_surface>]\n");
	  exit(1);
	}
      

      nextarg++;
    }

  if(minmax==NULL)
    {
      if((minmax=(int *)malloc(nobjs*sizeof(int)))==NULL){fprintf(stderr,"malloc error.\n");exit(1);}
      for(int j=0;j<dim;j++)
	minmax[j]=-1;
    }


  fprintf(stderr,"Sampling attainment surface number %d\n", att_surf);
  fprintf(stderr,"Number of objectives being used is %d\n", nobjs);
  fprintf(stderr,"Resolution being used is %d\n", resol);
  fprintf(stderr,"Total number of test points is thus: %g\n", nobjs*pow(resol, nobjs-1));

  // MALLOC arrays
  mydalloc(q, nobjs)
  mydalloc(r, nobjs)

  mydalloc(min, nobjs)
  mydalloc(max, nobjs)
  mydalloc(v, nobjs)

  if(paramflag)
    {
      char line[MAX_LINE_LENGTH];
      double number;
      if((fp=fopen(paramfile,"rb")))
	{
	  for (int li=0;li<2;li++)
	    {
	      if(fgets(line, MAX_LINE_LENGTH, fp)==NULL)
		{
		  fprintf(stderr,"Error in param file\n");
		  exit(1);
		}
	      int k=0;
	      sscanf(line, "%lf", &number);
	      if(li==0)
		min[k++] = number;
	      else if(li==1)
		max[k++] = number;
	      int i = 0;
	      for (int j = 1; j < nobjs; j++) 
		{
		  while (line[i] != ' ' && line[i] != '\n' && line[i] != '\0')
		    i++;
		  if((sscanf(&(line[i]), "%lf", &number)) <= 0)
		    {
		      fprintf(stderr,"Error in param file\n");
		      exit(0);
		    }
		  if(li==0)
		    min[k++] = number;
		  else if(li==1)
		    max[k++] = number;
		  while (line[i] == ' ' && line[i] != '\0')
		    i++;
		}
	    }
	  fclose(fp);
	}      
      else
	{
	  fprintf(stderr,"Coundn't open %s for reading. Exiting.\n", paramfile);
	  exit(1);
	}
    }



  /* read in each of the approximation sets */
  int *start;
  if((fp=fopen(argv[1], "rb")))
    {
      check_file(fp, &nr, &num);
      rewind(fp);

      if((po = (dnode *)malloc(num*sizeof(dnode)))==NULL){fprintf(stderr,"malloc error\n"); exit(1);}
      for(int i=0;i<num;i++)
	{
	  if((po[i].o = (double *)malloc(nobjs*sizeof(double)))==NULL){fprintf(stderr,"malloc error\n"); exit(1);}
	}
      
      int j=0;

      if((start = (int *)malloc( (nr+1)*sizeof(int)))==NULL){fprintf(stderr,"malloc error\n"); exit(1);}
      int ntot=0;
      start[0]=ntot;
      while(j<nr)
	{
	  read_file(fp, &nf, &(po[start[j]]));
	  ntot+=nf;
	  start[j+1]=ntot;
	  j++;
	}
    }
  else
    {
      fprintf(stderr,"Couldn't open %s", argv[1]);
      exit(0);
    }

  if(att_surf>nr)
    {
      fprintf(stderr,"Only %d input approximation sets. But you have chosen to find attainment surface #%d. Exiting.\n", nr, att_surf);
      exit(1);
    }
  
  mydalloc(inter, (nr+1))
    //  if((inter=(double *)malloc( (nr+1)*sizeof(double)))==NULL){fprintf(stderr,"malloc error.\n");exit(1);}

  // find the ranges of each objective
  if(!paramflag)
    {
      for(int k=0;k<nobjs;k++)
	{
	  min[k]=VLARGE;
	  max[k]=-VLARGE;
	  for(int j=0;j<nr;j++)
	    {
	      int i=0;
	      int m;
	      do
		{
		  m=start[j]+i;
		  if(po[m].o[k]<min[k])
		    min[k]=po[m].o[k];
		  if(po[m].o[k]>max[k])
		    max[k]=po[m].o[k];	     
		  i++;
		}while(m+1<start[j+1]);
	    }
	  fprintf(stderr, "min = %g, max = %g\n", min[k],max[k]);
	}
    }
  

  for(int k=0;k<nobjs;k++)
    {      
      for(int j=0;j<nr;j++)
	{	  
	  int m = start[j];
	  int n = start[j+1]-start[j];
	  d=k;

  	  qsort(&(po[m]), n, sizeof(P), compare);
	}
      
      bool flag;
      int count=0;

      while(count<pow(resol,nobjs-1))
	{
	  int jj=0;
	  for(int j=0;j<dim;j++)
	    {
	      if(j<k)
		jj=j;
	      else if(j>k)
		jj=j-1;
		  
	      if(j!=k)
		{
		  v[j]=(count%((int)pow((double)resol,jj+1)))/((int)pow((double)resol,jj));
		  //    fprintf(stderr,"%g ", v[j]);
		  v[j]/=resol;		 
		}
	      //   else
	      // 	fprintf(stderr,"* ");
	    }
	  //	  fprintf(stderr, "k=%d\n", k);
	  for(int j=0;j<dim;j++)
	    {
	      if(j!=k)
		q[j]=min[j]-0.05*(max[j]-min[j])+v[j]*1.1*(max[j]-min[j]);
	    }

	  
	  int next=0;
	  for(int a=0;a<nr;a++)
	    {
	      flag=false;
	      for(int c=0;c<start[a+1]-start[a];c++)
		{
		  int cc;
		  if(minmax[k]==-1)
		    cc=c+start[a];
		  else if(minmax[k]==1)
		    cc=start[a+1]-1-c;
		     
		  q[d]=po[cc].o[d];
		  if (dominates(po[cc].o,q,minmax,dim)==1)
		    {
		      flag=true;
		      break;
		    }
		}
	      if(flag)
		{
		  if(minmax[k]==-1)
		    inter[next]=q[k];
		  else if(minmax[k]==1)
		    inter[next]=-q[k];
		  next++;
		}
	      
	      // fprintf(stderr,"\n");

	    }

	  qsort(inter, next, sizeof(double), compdoub);


	  if(next>att_surf-1)
	    {
	      if(minmax[k]==-1)
		q[k]=inter[att_surf-1];
	      else if(minmax[k]==1)
		q[k]=-inter[att_surf-1];
	      
	      for(int j=0;j<dim;j++)
		printf("%g ", q[j]);
	      printf("\n");
	    }
	  count++;
	}
    }
  exit(0);

}

int compare(const void *i, const void *j)
{
  // comparing function used by qsort()

  double x;
  x = ( ((P *)i)->o[d]-((P *)j)->o[d]);
  if(x<0)
    return -1;
  else if(x==0)
    return 0;
  else
    return 1;
}

int compdoub(const void *i, const void *j)
{
  // comparing function used by qsort() for sorting an array of doubles

  double x;
  x = *((double *)i) - *((double *)j);
  // cerr << (double*)i << " " << (double *)j << " " << x << " " << *(double*)i << " " << *(double*)j << endl;
  if(x<0)
    return -1;
  else if(x==0)
    return 0;
  else
    return 1;
}



int dominates(double *a, double *b, int *minmax, int n)
{
  // Returns 1 if a dominates b
  // Returns -1 if b dominates a
  // Returns 0 if a==b or a and b are mutually nondominating (a~b)

  // This dominance relationship is measured using the dimensions 
  // defined in the array minmax[]. If minmax[i]=1 then objective i
  // is compared assuming maximization of i. If minmax[i]=-1 then
  // objective i is compared assuming minimization of i. If
  // minmax[i]=0 then no comparison is made on objective i. The 
  // total number of dimensions which could potentially be compared
  // is given by the argument n.

  double diff;
  int abb=0; // counts number of dimensions where a beats b
  int bba=0;
  
  for(int i=0;i<n;i++)
    {
      if(!minmax[i]==0)
	{
	  diff=a[i]-b[i];
	  if(diff>0)
	    {
	      if(minmax[i]==1)
		abb++;
	      else if (minmax[i]==-1)
		bba++;
	      else
		{
		  fprintf(stderr, "minmax out of range\n");
		  exit(0);
		}
	    }
	  else if(diff<0)
	    {
	      if(minmax[i]==1)
		bba++;
	      else if (minmax[i]==-1)
		abb++;
	      else
		{
		  fprintf(stderr, "minmax out of range\n");
		  exit(0);
		}
	    }
	}
      if((bba>0)&&(abb>0))
	return(0);
    }
  if(abb>0)
    return(1);
  else if(bba>0)
    return(-1);
  else
    return(0);

}


void  check_file(FILE  *fp, int  *no_runsp, int  *tot_pointsp)
    /* check_file() function
       determines the total number of points and the number of runs
    */
{
  char  line[MAX_STR_LENGTH];
  int  i, j;
  int  new_run;
  int  no_points;
  double  number;
  
  no_points = 0;
  *tot_pointsp = 0;
  *no_runsp = 0;
  new_run = 1;
  while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) 
    {      
      if (sscanf(line, "%lf", &number) != 1)
	new_run = 1;
      else 
	{
	  if (new_run == 1)
	    {
	      (*no_runsp)++;
	      *tot_pointsp += no_points;
	      no_points = 0;
	    }
	  new_run = 0;
	  i = 0;
	  for (j = 1; j < nobjs; j++) 
	    {
	      while (line[i] != ' ' && line[i] != '\n' && line[i] != '\0')
		i++;
	      if((sscanf(&(line[i]), "%lf", &number)) <= 0)
		{
		  fprintf(stderr,"Error in data set file\n");
		  exit(0);
		}
	      
	      while (line[i] == ' ' && line[i] != '\0')
		i++;
	    }
	  no_points++;
	}
    }
  *tot_pointsp += no_points;
}

void  read_file(FILE  *fp, int  *no_pointsp, P *po)
{
  char  line[MAX_STR_LENGTH];
  int  i, j, k;
  int  reading;
  double  number;
  double *vector;

  //  if((vector=(double *)malloc(nobjs*sizeof(double)))==NULL){fprintf(stderr,"malloc error\n");exit(1);}
  mydalloc(vector,nobjs)
  
  reading = 0;
  *no_pointsp = 0;
  //  fprintf(stderr, "read_file\n");
  while (fgets(line, MAX_LINE_LENGTH, fp) != NULL) 
    {
      k=0;
      if (sscanf(line, "%lf", &number) != 1)
	{
	  //  fprintf(stderr,"reading=%d\n", reading);
	  if (reading) break;
	}
      else 
	{
	  reading = 1;
	  vector[k++] = number;
	  i = 0;
	  for (j = 1; j < nobjs; j++) 
	    {
	      while (line[i] != ' ' && line[i] != '\n' && line[i] != '\0')
		i++;
	      if((sscanf(&(line[i]), "%lf", &number)) <= 0)
		{
		    fprintf(stderr,"Error in data file.\n");
		    exit(0);
		}
	      
	      vector[k++] = number;
	      while (line[i] == ' ' && line[i] != '\0')
		i++;
	    }
	  (*no_pointsp)++;
	  for(int k=0;k<nobjs;k++)
	    po[*no_pointsp-1].o[k]=vector[k];
	}
    } 
}
