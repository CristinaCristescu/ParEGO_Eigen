//
//  main.cpp
//  MatrixTest
//
//  Created by Bianca Cristina Cristescu on 28/10/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//

#include <iostream>
#include </home/cristina/eigen/Eigen/Dense>

using namespace std;

using namespace Eigen;

int main()
{
    clock_t t1,t2;
    //matrix[1000][1000]
    MatrixXd matA(3,3);
    MatrixXd matB(3,3);
    MatrixXd mat(3,3);

    
    matA << 3, 3, 3, 3,3,3,3,3,3;
    matB << 5, 5, 5, 5,5,5,5,5,5;

    t1=clock();
    mat= matA*matB;
    t2=clock();
    //Vector2d u(-1,1), v(2,0);
    std::cout << "Here is mat*mat:\n" << mat << std::endl;
    //std::cout << "Here is mat*u:\n" << mat*u << std::endl;
    //std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
    //std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
    //std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
    //std::cout << "Let's multiply mat by itself" << std::endl;
    //mat = mat*mat;
    //std::cout << "Now mat is mat:\n" << mat << std::endl;
    //code goes here
    
    float diff ((float)t2-(float)t1);
    float seconds = diff / CLOCKS_PER_SEC;
    cout<<seconds<<endl;
    return 0;
}