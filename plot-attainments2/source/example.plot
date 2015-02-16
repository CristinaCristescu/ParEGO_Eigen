set term post color 18
set ls 1 pt 7 ps 0.5
set ls 2 pt 8 ps 0.5
set ls 3 pt 10 ps 0.5
set xlabel "maximize f1"
set ylabel "minimize f2"
set zlabel "maximize f3"
set view 70, 80
set output "../figures/example_1.ps"
splot "att_1.dat" w p ls 1
set output "../figures/example_11.ps"
splot "att_11.dat" w p ls 2 
set output "../figures/example_21.ps"
splot "att_21.dat" w p ls 3