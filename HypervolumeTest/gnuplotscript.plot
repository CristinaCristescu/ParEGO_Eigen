set key below
set ylabel "Number of variables set to true"
set xlabel "Number of satisfied clauses out of 180 from one algorithm run"
set xlabel "Number of satisfied clauses out of 180"
set title "Pareto front of test instance xxxxxxxxx from one algorithm run"
plot "test-positive.dat" title "Multiobjective EA: Nondominated Solutions" w points ps 3, "test-positive2.dat" title "Multiobjective EA: All Solutions"
set output "moo.gif"
set terminal gif
replot
set output "moo.ps"
set terminal post landscape font 18
replot
