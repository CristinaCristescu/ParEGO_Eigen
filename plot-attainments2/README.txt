
plot_attainments2.cc takes R approximation sets as input, and outputs a
 number of uniformly distributed  points on the *Jth attainment
 surface (where J is in the range 1,...,R).  Its purpose is to allow
 straightforward plotting of (especially 3d) attainment surfaces in
 gnuplot.

For more information on usage, please see the header of the source
code.

An example of the output of the program is given in the three
postscript figures, att_1.ps, att_11.ps and att_21.ps.  These plot the
1st, 11th and 21st attainment surfaces, respectively, of the data file
supplied, testminmax.dat, which contains 21 approximation sets of an
optimizer running on a mixed minimization/maximization problem of 
3 objectives.

The shell script supplied, example_run.sh ,  was used to make these 
figures from the source files. Please read and execute this file.

--
NOTES:

* The Jth attainment surface is the boundary of the
set of goals that have been attained (independently)
in J or more of the runs of an optimizer (i.e. approximation sets in the input file).
In other words, every point on the Jth attainment surface
is weakly dominated (but NOT strictly dominated) 
by a goal (point) that has been attained in at least J of 
the approximation sets given as input.

For more information on attainment surfaces, the reader is
referred to:

@inproceedings{Fonseca96a,
   author = {Carlos M. Fonseca and Peter J. Fleming},
   address = {Berlin, Germany},
   booktitle = {Parallel Problem Solving from Nature---PPSN IV},
   editor = {Hans-Michael Voigt and Werner Ebeling and Ingo Rechenberg and Hans-Paul Schwefel},
   key = {Fonseca and Fleming, 1996a},
   OPTmonth = {September},
   pages = {584--593},
   publisher = {Springer-Verlag},
   series = {Lecture Notes in Computer Science},
   title = {On the {P}erformance {A}ssessment and {C}omparison of {S}tochastic {M}ultiobjective {O}ptimizers},
   year = {1996}
}

@incollection{Fonseca01,
   author = {Viviane Grunert da Fonseca and Carlos M. Fonseca
   and Andreia O. Hall},
   key = {Fonseca et al., 2001},
   booktitle = {First International Conference on 
	Evolutionary Multi-Criterion Optimization},
   title = {Inferential {P}erformance {A}ssessment of
   {S}tochastic {O}ptimisers and the {A}ttainment
   {F}unction},
   editor = {Eckart Zitzler and Kalyanmoy Deb and Lothar Thiele
	and Carlos A. Coello Coello and David Corne},
   publisher = {Springer-Verlag. Lecture Notes in Computer
	Science No. 1993},
   pages = {213--225},
   year = {2001}
}


Joshua Knowles, 2005. j.knowles@manchester.ac.uk
