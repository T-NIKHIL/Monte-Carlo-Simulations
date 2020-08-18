// Solving 2-D unsteady heat diffusion equation using Finite difference method and Explicit method

// Libraries used
#include <iostream>
#include <math.h>
#include <stdlib.h>
// For writing to files
#include <fstream>
using std::ifstream;
using std::ofstream;
// For exit function 
#include <cstdlib>
// To prevent rounding off when printing to file
#include <iomanip>

using namespace std;

int main()
{

    // index values
	int i = 1;
	int j = 1;

    // Setting x limits 
	double x_min = 0;
	double x_max = 1;
	// Setting number of grid points along X axis
	int x_num = 21;
	double x_delta;

    // Setting y limits
	double y_min = 0;
	double y_max = 1;
	// Setting number of grid points along Y axis
	int y_num = 21;
	double y_delta;

	double t_max;
	int t_num;
	double t_delta;
	// time steps
	double t = 0;

	// Thermal diffusivity
	double alpha = 0.0001;

	// Stores the x-cood values
	double x[x_num];
	// Stores the y-cood values
	double y[y_num];
	// Matrix to store the temperature values
	double T[x_num][y_num]; 
	// calculated value
	float cal=0.0;
	double stability;

	ifstream inFile;
	inFile.open("input.txt");

	if(!inFile){
		cerr<<"Unable to locate input file.";
	}

	inFile>>t_max;
	inFile>>t_delta;

	cout<<"Entered end time : "<<t_max<<endl;

	cout<<"\n";

	cout<<"Entered delta t : "<<t_delta<<endl;

	cout<<"\n";

	x_delta = (x_max - x_min)/(x_num - 1);
	cout << "x_delta: "<< x_delta << endl;

	y_delta = (y_max - y_min)/(y_num - 1);
	cout << "y_delta: "<< y_delta << endl;

	cout<<"\n";

	stability = (alpha*t_delta)/pow(x_delta,2);
	cout << "Stability criterion value : " << stability << endl;

	cout<<"\n";

	t_num = 1 + (t_max - t)/t_delta;
	cout << "Number of iterations : " << t_num << endl;

	cout<<"\n";

	// Setting up x co-od values
	x[0] = 0;
	T[0][0] = 20;
	for(i=1;i<=x_num-1;i++)
	{
		x[i] = x[i-1] + x_delta;
		// Setting up boundary conditions at x = 0 and x = 1
		T[0][i] = 20;
		T[x_num-1][i] = 20;
	}

	// Setting up y co-od values
	y[0] = 0;
	for(j=1;j<=y_num-1;j++)
	{
		y[j] = y[j-1] + y_delta;
		// Setting up boundary conditions at y = 0 and y = 1
		T[j][0] = 20;
		T[j][y_num-1] = 20;
	}

	// Adding in the initial conditions
	for(i=1;i<=x_num-2;i++)
	{
		for(j=1;j<=y_num-2;j++)
		{
			cal=pow((x[i]-0.5),2) + pow((y[j]-0.5),2);
			if(cal < 0.2)
			{
				T[i][j] = 40;
			}
			else
			{
				T[i][j] = 20;
			}
		}
	}

	// Printing Temperature matrix at t=0 to .txt for plotting
	std::ofstream Tdist_int;
	Tdist_int.open("Tdist_0.txt",std::ofstream::out | std::ofstream::trunc);

	if(!Tdist_int){
		cerr<<"Error : file could not be opened"<<endl;
		exit(1);
	}

	for(i=0;i<=x_num-1;i++)
	{
		for(j=0;j<=y_num-1;j++)
		{
			Tdist_int<<" "<<std::setprecision(3)<<std::fixed<<T[i][j]<<" ";
		}
		Tdist_int<<"\n";
	}

	Tdist_int.close();

	int counter = 0;
	// Explicit from of 2-D unsteady heat diffusion equation
	while(t<=t_max)
	{
		for(i=1;i<=x_num-2;i++)
		{
			for(j=1;j<=y_num-2;j++)
			{
				T[i][j] = T[i][j] + stability*(T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1] - 4*T[i][j]);

			}
		}

		// Increasing time step
		t = t + t_delta;
		// Tracking number of iterations
		counter++;

		if(remainder(counter,100) == 0)
		{
			cout<<"Iteration "<<counter<<endl;
		}

	}

	// Printing Temperature matrix at the t=tmax to .txt for plotting.
	std::ofstream Tdist_final;
	Tdist_final.open("Tdist_7200.txt", std::ofstream::out | std::ofstream::trunc);

	if(!Tdist_final){
		cerr<<"Error : file could not be opened"<<endl;
		exit(1);
	}

	for(i=0;i<=x_num-1;i++)
	{
		for(j=0;j<=y_num-1;j++)
		{
			Tdist_final<<" "<<std::setprecision(3)<<std::fixed<<T[i][j]<<" ";
		}
		Tdist_final<<"\n";
	}

	Tdist_final.close();

	return 0;
}