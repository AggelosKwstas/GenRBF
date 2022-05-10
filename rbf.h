#ifndef KMEANS_CPP_RBF_H
#define KMEANS_CPP_RBF_H

# include "dataset.h"
# include "kmeans.h"

class rbf {
private:
    int NumberOfWeights;
    Data weights;
    Matrix centers;
    Data variance;
    Dataset *train, *test;
    int dimension;
    Data Classes;
    Data Left, Right;
    Data copyLeft, copyRight;
    double F_constant;

    int count;
    vector<int> members;
    vector<int> NumberOfEach;

    Matrix Transpose(Matrix &A);

    Matrix Multiply(Matrix &A, Matrix &B);

    Matrix Inverse(Matrix &mat);

    Matrix Pseudo_Inverse(Matrix &A);

    void Initialization();

    double gauss(Data &p, Data &c, double s);

    double distance(Data &x, Data &y);

    Matrix chromosomes;
    Matrix copychrom;

    double maxRight;
    double minLeft;
public:
    rbf();

    void setTrainSet(Dataset *d);

    void setTestSet(Dataset *d);

    void setNumberOfWeights(int n);

    int getNumberOfWeights();

    void setClasses();

    bool isEqual(double a, double b);

    void RbfTrain();

    double TrainSumSquaredError();

    double TestSumSquaredError();

    void MinimumClassificationError();

//  Genetic Algorithm implemnentaion
    double genTrainSumSquaredError(Matrix genCenter);

    void setBounds();

    void setChromosomes(int n, double max, double min);

    void runGen(int n);

    void decode(double element);

    Matrix normalize(int index);

    double findNthLargestElement(vector<double> &v, int element);

    double findNthMinimumElement(vector<double> &v, int index);

//    Data unNormalize(Matrix child);

//    void decodeBounds();

    Data decodeChild(Data child);

    double Delta(int iter, int iterMax, double y, double r);

    vector<int> findBestIndices(vector<double> times, const int &N);

    vector<int> findWorstIndices(vector<double> times, const int &N);


};

#endif //KMEANS_CPP_RBF_H