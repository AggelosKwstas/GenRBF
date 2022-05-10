#ifndef MAIN_CPP_KMEANS_H
#define MAIN_CPP_KMEANS_H

#include "dataset.h"
class kmeans {
private:
    Dataset *dataset;
    Matrix centroid;
    int teams;

    Matrix TestCoord;
    int dimension;
    int count;
    vector<int> indexes;
    vector<int> members;
    vector<int> NumberOfEach;
    double distance(Data &x,Data &y);
    Data variance;

public:
    kmeans(Dataset *d , int NumberOfTeams);
    void AssignTeams();
    void CalculateMean();
    void run();
    int nearestTeam(Data &x);
    static bool compare(Matrix A,Matrix B);
    void calculateVariance();
    void CalculateVariance();
    void calculateGenVariance();
    Matrix getCenters();
    Data getVariance();
};

#endif //MAIN_CPP_KMEANS_H