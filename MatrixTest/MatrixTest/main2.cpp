//
//  main2.cpp
//  MatrixTest
//
//  Created by Bianca Cristina Cristescu on 28/10/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//
#include <iostream>
#include </Users/cristina/eigen/eigen-eigen-1306d75b4a21/Eigen/Dense>
#include "Matrix.h"

#define MATRIX_SIZE 500
using namespace std;

using namespace Eigen;

int main()
{
   //double sum = 0;
   clock_t t1,t2,t3,t4;
   
   srand(1);
   MyMatrix MyA(MATRIX_SIZE, MATRIX_SIZE);
   MyMatrix MyB(MATRIX_SIZE, MATRIX_SIZE);
   MyMatrix MyC(MATRIX_SIZE, MATRIX_SIZE);
   MyMatrix MyD(MATRIX_SIZE, MATRIX_SIZE);
   MatrixXd EigenA(MATRIX_SIZE, MATRIX_SIZE);
   MatrixXd EigenB(MATRIX_SIZE, MATRIX_SIZE);
   MatrixXd EigenC(MATRIX_SIZE, MATRIX_SIZE);
   MatrixXd EigenD(MATRIX_SIZE, MATRIX_SIZE);
   
   for (int i = 0; i < MATRIX_SIZE; i++)
      for (int j = 0; j < MATRIX_SIZE; j++)
      {
         int a = rand();
         int b = rand();
         int c = rand();
         int d = rand();
         MyA.insert(i, j, a);
         MyB.insert(i, j, b);
         MyC.insert(i, j, c);
         MyD.insert(i, j, d);
         
      }
   
   for (int i = 0; i < MATRIX_SIZE; i++)
      for (int j = 0; j < MATRIX_SIZE; j++)
      {
         int a = rand();
         int b = rand();
         int c = rand();
         int d = rand();
         EigenA(i,j) = a;
         EigenB(i,j) = b;
         EigenC(i,j) = c;
         EigenD(i,j) = d;
      }
   
   MyMatrix resultMy(MATRIX_SIZE, MATRIX_SIZE);
   MatrixXd resultEigen(MATRIX_SIZE, MATRIX_SIZE);
   
   t1 = clock();
   for (int run = 0; run < 200; run++)
      resultMy = MyA*MyB;
   
   t2 = clock();
   
   t3 = clock();
   for (int run =0; run < 200; run++)
      resultEigen = EigenA*EigenB;
   t4 = clock();
   
   /* Matrix multiplication
    clock_t t1,t2,t3,t4;
    int A[MATRIX_SIZE][MATRIX_SIZE];
    
    for (int i = 0; i < MATRIX_SIZE; i++)
    for (int j = 0; j < MATRIX_SIZE; j++)
    A[i][j] = rand();
    
    MyMatrix mat(MATRIX_SIZE,MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; i++)
    for (int j = 0; j < MATRIX_SIZE; j++)
    mat.insert(i,j,A[i][j]);
    
    t1=clock();
    mat= mat*mat;
    t2=clock();
    
    MatrixXd eigen(MATRIX_SIZE,MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; i++)
    for (int j = 0; j < MATRIX_SIZE; j++)
    eigen(i,j) = A[i][j];
    t3 = clock();
    eigen = eigen*eigen;
    t4 = clock();
    //code goes here
    //float diff3 ((float)t6-(float)t5);
    //float seconds3 = diff3 / CLOCKS_PER_SEC;
    //cout<<seconds3<<endl;
    */
   
   float diff1 ((float)t2-(float)t1);
   float seconds1 = diff1 / CLOCKS_PER_SEC;
   cout<<seconds1<<endl;
   float diff2 ((float)t4-(float)t3);
   float seconds2 = diff2 / CLOCKS_PER_SEC;
   cout<<seconds2<<endl;
   
   return 0;
}