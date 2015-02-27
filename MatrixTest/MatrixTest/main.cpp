//
//  main.cpp
//  MatrixTest
//
//  Created by Bianca Cristina Cristescu on 28/10/14.
//  Copyright (c) 2014 Bianca Cristina Cristescu. All rights reserved.
//

#include <iostream>
#include </Users/cristina/eigen/eigen-eigen-1306d75b4a21/Dense>
#define RN rand()/(RAND_MAX+1.0)
#define HA 250
#define WA 250
#define WB 250
#define HB WA 
#define WC WB   
#define HC HA
#define index(i,j,ld) (((j)*(ld))+(i))


using namespace std;

using namespace Eigen;

int main()
{
    clock_t t1,t2;
      srand(1);

    t1=clock();
    //matrix[1000][1000]
    MatrixXf matA(HA,WA);
    MatrixXf matB(HB,WB);
    MatrixXf mat(HC,WC);

    for(int i=0;i<HA;i++)
        for(int j=0;j<WA;j++)
            matA(index(i,j,HA))=(float)index(i,j,HA);
    for(int i=0;i<HB;i++)
        for(int j=0;j<WB;j++)
            matB(index(i,j,HB))=index(i,j,HB);    
    //cout<<"A:"<<matA<<"\n";
    //cout<<"B:"<<matB<<"\nwuu";

    
    mat= matA*matB;
    
    //Vector2d u(-1,1), v(2,0);
    //std::cout << "Here is mat*mat:\n" << mat << std::endl;
    //std::cout << "Here is mat*u:\n" << mat*u << std::endl;
    //std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
    //std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
    //std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
    //std::cout << "Let's multiply mat by itself" << std::endl;
    //mat = mat*mat;
    //std::cout << "Now mat is mat:\n" << mat << std::endl;
    //code goes here
    t2=clock();
    float diff ((float)t2-(float)t1);
    float seconds = diff / CLOCKS_PER_SEC;
    cout<<seconds<<endl;
    return 0;
}