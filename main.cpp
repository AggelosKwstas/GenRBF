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
    myrbf.run();
    double result = myrbf.TestSumSquaredError();
    double res = myrbf.TrainSumSquaredError();
    cout<<" *** K-Means RBF Output ***"<<endl;
    cout<<endl;
    cout<<"Mean Sum Squared Test Error : "<<result<<endl;
    cout<<"Mean Sum Squared Train Error : "<<res<<endl;
    myrbf.setClasses();
    myrbf.MinimumClassificationError();
    cout<<endl;
    cout << " *** Genetic algorithm approach ***" << endl;
    myrbf.setBounds();
    int n;
    cout<<endl;
    cout<<"enter number of chromosomes : ";
    cin>>n;
    myrbf.runGen(n);
    return 0;
}

