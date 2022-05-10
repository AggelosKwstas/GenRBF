
#include <iostream>
#include "kmeans.h"
#include "rbf.h"

using namespace std;

int main() {

    Dataset test("iris.test");
    Dataset train("iris.train");
    rbf myrbf;
    myrbf.setTrainSet(&train);
    myrbf.setTestSet(&test);
    myrbf.setNumberOfWeights(6);
    myrbf.RbfTrain();
//    double result = myrbf.TestSumSquaredError();
//    double res = myrbf.TrainSumSquaredError();
//    cout<<"Mean Squared Test Error : "<<result<<endl;
//    cout<<"Mean Squared Train Error : "<<res<<endl;
//    myrbf.setClasses();
//    myrbf.MinimumClassificationError();
    myrbf.setBounds();
    int n;
    cout<<"enter number of chromosomes:"<<endl;
    cin>>n;
    myrbf.runGen(n);
    return 0;
}