g++ plot-attainments2.cc -o plot-att -lm -O3
./plot-att testminmax.dat -k 3 -mm 1 -1 1 -r 60 -a 1 > att_1.dat
./plot-att testminmax.dat -k 3 -mm 1 -1 1 -r 60 -a 11 > att_11.dat
./plot-att testminmax.dat -k 3 -mm 1 -1 1 -r 60 -a 21 > att_21.dat
gnuplot "example.plot"
exit
