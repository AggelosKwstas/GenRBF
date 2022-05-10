#include <fstream>
#include <iostream>
#include <algorithm>
#include "dataset.h"
using namespace std;

Dataset::Dataset() {
    x.resize(0);
    y.resize(0);
}
Dataset::Dataset(string filename) {
    ifstream file;
    file.open(filename);
    int columns;
    int rows;
    if (!file){
        cout<<"error cant open file";
        exit(0);
    }
    if(file.is_open()) {
        file>>columns;
        file>>rows;
        x.resize(rows);
        y.resize(rows);
        for(int i=0;i<rows;i++) {
            x[i].resize(columns);
            for(int j=0;j<columns;j++)
                file>>x[i][j];
            file>>y[i];
        }
    }
    findClasses(y);
}
void Dataset::findClasses(vector<double> test) {
    PatternClass=test;
    sort(PatternClass.begin(),PatternClass.end());
    PatternClass.erase(unique(PatternClass.begin(),
                              PatternClass.end()), PatternClass.end());
}
int Dataset::rows() const {

    return y.size();
}
int Dataset::cols() const {
    return x[0].size();
}
Data Dataset::getX(int i) const {
    if(i<0 || i>rows())
        cout << "out of bounds";
    return x[i];
}
Data Dataset::ReturnPatternClasses() {
    return PatternClass;
}
double Dataset::getY(int i) const {
    if(i<0 || i>rows())
        cout<<"out of bounds";
    return y[i];
}
Matrix Dataset::returnX() {
    return x;
}