#ifndef MAIN_CPP_DATASET_H
#define MAIN_CPP_DATASET_H
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <vector>
#include <string>
using namespace std;
typedef vector<double> Data;
typedef vector<Data> Matrix;
class Dataset
{
private:
    Matrix x;
    Data PatternClass;
    Data y;
public:
    Dataset();
    Dataset(string filename);
    int rows() const;
    int cols() const;
    Data getX(int i) const;
    double getY(int i) const;
    void findClasses(vector<double> test);
    Data ReturnPatternClasses();
    Matrix returnX();
};

#endif //MAIN_CPP_DATASET_H