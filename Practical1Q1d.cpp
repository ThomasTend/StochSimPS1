#include <iostream>
#include <math.h>
#include <Eigen>
#include<chrono>
#include <random>
#include <boost/math/special_functions/erf.hpp>

using namespace Eigen;
using namespace std;

double ICDFnormal(double p){
    return -sqrt(2) * boost::math::erfc_inv(2*p);
}

int main()
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    int n=1;

    Matrix2d A;
    A << 4,1,1,4;
    cout << "The matrix A is" << endl << A << endl;

    LLT<Matrix2d> lltOfA(A); // compute the Cholesky decomposition of A
    Matrix2d L = lltOfA.matrixL(); // retrieve factor L  in the decomposition

    cout << "The Cholesky factor L is" << endl << L << endl;
    cout << "To check this, let us compute L * L.transpose()" << endl;
    cout << L * L.transpose() << endl;
    cout << "This should equal the matrix A";

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    int counter=0;
    while(std::chrono::steady_clock::now() - start <= std::chrono::seconds(60))
    {
        MatrixXf w(2,n);
        for(int i=0; i<n; i++){
            Vector2d tempMat;
            tempMat << ICDFnormal(dis(gen)), ICDFnormal(dis(gen));
            Vector2d tempVec;
            tempVec << L*tempMat;
            w(0,i) = tempVec(0,0);
            w(1, i) = tempVec(1, 0);
            // To check we are indeed generating normals
            //double temp = dis(gen);
            //cout << temp << endl;
            //cout << ICDFnormal(temp) << endl;
            //cout << 0.5 * boost::math::erfc(-(ICDFnormal(temp))/(sqrt(2))) << endl;
        }
        counter+=1;
    }
    cout << "We have generated " + std::to_string(counter) + " normals with covariance sigma."<< endl;
}

