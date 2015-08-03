#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	ifstream file;
	string filename = string(argv[1]);
	if(argc > 3)
	file.open(filename.c_str());

	int dim = atoi(argv[2]);

	int file_size = atoi(argv[3]);

    double min[dim], max[dim];

    for (int d= 0; d < dim; d++)
    {
    	file >> min[d];
  		max[d] = min[d];
    }
    double currentNumber[dim];
	for (int index = 1; index < file_size; index++)
	{
		for (int d = 0; d < dim; d++)
		{
			file >> currentNumber[d];
			if (currentNumber[d] < min[d])
			{
				min[d] = currentNumber[d];
				cout <<"min " << min[d] << "\n";
 			}
			if (currentNumber[d] > max[d])
			{
				max[d] = currentNumber[d];
				cout <<"max " << max[d] << "\n";

			}
		}
	}

	for (int d = 0; d < dim; d++)
	{
		cout <<"Min : " << min[d] << "\n";
		cout <<"Max : " << max[d] << "\n";
	}

	double point[dim];
	double delta = 0.01;
	cout << "B ( ";
	for (int d = 0; d < dim; d++)
	{
		point[d] = max[d]+delta*(max[d]-min[d]);
		cout << point[d] << " ";
	}
	cout << ")\n";

}