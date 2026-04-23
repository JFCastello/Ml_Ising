#include "../include/Samples_Gen.h"
#include <cmath>
#include <iostream>

double treeshold = .6;
double Tc_seed0 = 2.;
int Num_Samples = 5000; // Número de muestras Monte Carlo
int L = 10;

int main(){
    std::cout << "Starting simulation with the following parameters: " << std::endl;
    std::cout << "- Treeshold = " << treeshold << std::endl;
    std::cout << "- Tc_seed0 = " << Tc_seed0 << std::endl;
    std::cout << "- Num_Samples = " << Num_Samples << std::endl;
    std::cout << "- L = " << L << std::endl;

    double tr = 2 * treeshold;
    int iter = 1;
    while (tr >= treeshold){
        std::cout << "Iteration #" << iter << std::endl;

    }

}