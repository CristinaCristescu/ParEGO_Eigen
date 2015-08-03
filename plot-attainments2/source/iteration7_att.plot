set term post color 18
set ls 1 pt 7 ps 0.5
set ls 2 pt 8 ps 0.5
set ls 3 pt 10 ps 0.5
set xlabel "maximize f1"
set ylabel "minimize f2"
set zlabel "maximize f3"
set view 70, 80
set output "../figures/dtlz2_it7_250.ps"
splot "att_it7_dtlz2_250.dat" w p ls 1
