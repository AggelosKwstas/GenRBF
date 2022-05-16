#include "rbf.h"
# include <math.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>

rbf::rbf() {
    NumberOfWeights = 1;
    dimension = 1;
    train = NULL;
    test = NULL;
    Initialization();
}

void rbf::setTrainSet(Dataset *d) {
    train = d;
    dimension = d->cols();
    count = d->rows();
}

void rbf::setTestSet(Dataset *d) {
    if (train == NULL)
        exit(0);
    if (train->cols() != d->cols())
        exit(0);
    else
        test = d;
}

void rbf::setNumberOfWeights(int n) {
    NumberOfWeights = n;
    Initialization();
}

int rbf::getNumberOfWeights() {
    return NumberOfWeights;
}

void rbf::Initialization() {
    weights.resize(NumberOfWeights);
    centers.resize(NumberOfWeights);
    variance.resize(NumberOfWeights);
    for (int i = 0; i < NumberOfWeights; i++) {
        centers[i].resize(dimension); //analoga poses diastaseis einai
        for (int j = 0; j < dimension; j++)
            centers[i][j] = 0; // arxikopoihsh se 0 twn kentrwn
        variance[i] = 0; // arxikopoihsh se 0 tou σ
        weights[i] = 2.0 * (rand() * 1.0 / RAND_MAX - 1.0);
    }
}

Matrix rbf::Transpose(Matrix &A) {
    int rows = A.size();
    if (rows == 0) return {{}};
    int cols = A[0].size();
    Matrix r(cols, vector<double>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            r[j][i] = A[i][j];
        }
    }
    return r;
}

Matrix rbf::Multiply(Matrix &A, Matrix &B) {
    int n = A.size();
    int m = A[0].size();
    int p = B[0].size();
    int q = B.size();

    if (m != q) {

        cout << "Cannot multiply these 2 Matrices ,"
                " the number of columns in the first matrix must be equal to the number of rows in the second matrix";
        exit(0);
    }
    Matrix c(n, vector<double>(p, 0));
    for (auto j = 0; j < p; ++j) {
        for (auto k = 0; k < m; ++k) {
            for (auto i = 0; i < n; ++i) {
                c[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return c;
}

Matrix rbf::Inverse(Matrix &mat) {

    auto height = mat.size();
    auto width = mat[0].size();
    Matrix result(height, Matrix::value_type(width));
    for (auto i = 0; i < width; ++i) {
        result[i][i] = 1;
    }
    for (auto j = 0; j < width; ++j) {
        auto maxRow = j;
        for (auto i = j; i < height; ++i) {
            maxRow = mat[i][j] > mat[maxRow][j] ? i : maxRow;
        }
        mat[j].swap(mat[maxRow]);
        result[j].swap(result[maxRow]);
        auto pivot = mat[j][j];
        auto &row1L = mat[j];
        auto &row1R = result[j];
        for (auto i = j + 1; i < height; ++i) {
            auto &row2L = mat[i];
            auto &row2R = result[i];
            auto temp = row2L[j];
            for (auto k = 0; k < width; ++k) {
                row2L[k] -= temp / pivot * row1L[k];
                row2R[k] -= temp / pivot * row1R[k];
            }
        }
        for (auto k = 0; k < width; ++k) {
            row1L[k] /= pivot;
            row1R[k] /= pivot;
        }
    }
    for (auto j = width - 1;; --j) {
        auto &row1L = mat[j];

        auto &row1R = result[j];
        for (auto i = 0; i < j; ++i) {
            auto &row2L = mat[i];
            auto &row2R = result[i];
            auto temp = row2L[j];
            for (auto k = 0; k < width; ++k) {
                row2L[k] -= temp * row1L[k];
                row2R[k] -= temp * row1R[k];
            }
        }
        if (j == 0) break;
    }
    return result;
}

Matrix rbf::Pseudo_Inverse(Matrix &A) {
    Matrix B = Transpose(A); //[173][10] => [10][173]
    Matrix V = Multiply(B, A); //[10][173] x [173][10] => [10][10]
    Matrix L = Inverse(V); // [10][10]
    Matrix C = Multiply(L, B);
    return C;
}

void rbf::setClasses() {
    Classes = test->ReturnPatternClasses();
}

double rbf::distance(Data &x, Data &y) {
    double s = 0.0;
    int i;
    for (i = 0; i < x.size(); i++)
        s += pow(x[i] - y[i], 2.0);
    return sqrt(s);
}

double rbf::gauss(Data &p, Data &c, double s) {
    double result = distance(p, c);
    return exp(-result * result / (s * s));
}

double rbf::TestSumSquaredError() {
    Data summ;
    summ.resize(centers.size());
    double cnt;
    double sum = 0;
    for (int i = 0; i < test->rows(); i++) {
        summ.clear();
        cnt = 0;
        Data x = test->getX(i);
        for (int j = 0; j < centers.size(); j++)
            summ.push_back(gauss(x, centers[j], variance[j]));
        for (int k = 0; k < summ.size(); k++) {
            cnt += weights[k] * summ[k];
        }
        sum += pow(test->getY(i) - cnt, 2);
    }
    return 100 * sum / test->rows();
}

double rbf::TrainSumSquaredError() {
    Data summ;
    summ.resize(centers.size());
    double cnt;
    double sum = 0;
    for (int i = 0; i < train->rows(); i++) {
        summ.clear();
        cnt = 0;
        Data x = train->getX(i);
        for (int j = 0; j < centers.size(); j++) {
            summ.push_back(gauss(x, centers[j], variance[j]));
        }
        for (int k = 0; k < summ.size(); k++) {
            cnt += weights[k] * summ[k];
        }
        sum += pow(train->getY(i) - cnt, 2);
    }
    return 100 * sum / train->rows();
}

bool rbf::isEqual(double a, double b) {
    double epsilon = 0.0000001;
    return std::abs(a - b) < epsilon;
}

void rbf::MinimumClassificationError() {
    Data summ;
    Matrix classes;
    classes.resize(Classes.size());
    for (int i = 0; i < classes.size(); i++)
        classes[i].resize(2);
    int random = 0 + (rand() % test->rows());
    Data xo = test->getX(random);
    double cnt, cnt1 = 0;
    for (int j = 0; j < centers.size(); j++)
        summ.push_back(gauss(xo, centers[j], variance[j]));
    for (int k = 0; k < summ.size(); k++) {
        cnt1 += weights[k] * summ[k];
    }
    Data min, max;
    min.resize(Classes.size());
    for (int i = 0; i < min.size(); i++)
        min[i] = 1e+100;
    max.resize(Classes.size());
    for (int i = 0; i < max.size(); i++)
        max[i] = 0;
    for (int i = 0; i < classes.size(); i++) {
        classes[i][0] = min[i];
        classes[i][1] = max[i];
    }
    int c = 0;
    for (int i = 0; i < test->rows(); i++) {
        summ.clear();
        cnt = 0;
        Data x = test->getX(i);
        double yy = test->getY(i);
        for (int n = 0; n < Classes.size(); n++) {
            if (isEqual(yy, Classes[n])) {
                for (int j = 0; j < centers.size(); j++)
                    summ.push_back(gauss(x, centers[j], variance[j]));
                for (int k = 0; k < summ.size(); k++) {
                    cnt += weights[k] * summ[k];
                }
                if (cnt < min[n]) {
                    min[n] = cnt;
                    classes[n][0] = cnt;
                } else if (cnt > max[n]) {
                    max[n] = cnt;
                    classes[n][1] = cnt;
                }
            }
        }
    }
    Data s;
    double cntt;
    double missed = 0;
    double category;
    for (int i = 0; i < test->rows(); i++) {
        s.clear();
        cntt = 0;
        Data x = test->getX(i);
        for (int j = 0; j < centers.size(); j++) {
            s.push_back(gauss(x, centers[j], variance[j]));
        }
        for (int k = 0; k < s.size(); k++) {
            cntt += weights[k] * s[k];
        }
        int index = i;
        for (int k = 0; k < classes.size(); k++) {
            if (classes[k][0] <= cntt && cntt <= classes[k][1]) {
                category = k;
                if (!(isEqual(test->getY(index), category))) {
                    missed++;
                    break;
                }
            }
        }
    }
    cout << "Minimum Classification error : " << 100 * missed / train->rows();
}

double rbf::findNthLargestElement(vector<double> &v, int element) {
    sort(v.begin(), v.end());
    return v[v.size() - element];
}

double rbf::findNthMinimumElement(vector<double> &v, int index) {
    std::nth_element(v.begin(), v.begin() + 1, v.end(), std::greater<int>());
    return v[index - 1];
}

void rbf::decode(double element) {

    copychrom = chromosomes;


    for (int i = 0; i < copychrom.size(); i++) {
        for (int j = 0; j < copychrom.size(); j++) {
            if (j == i)
                continue;
            for (int k = 0; k < copychrom[j].size(); k++) {
                double elementToRemove = element;
                vector<double>::iterator it = copychrom[i].begin();
                while (it != copychrom[i].end()) {
                    if ((*it) == elementToRemove) {
                        it = copychrom[i].erase(it);
                    } else {
                        it++;
                    }
                }
            }
        }
    }
}

Matrix rbf::normalize(int index) {

    Matrix copyCenter;
    copyCenter.resize(NumberOfWeights);
    for (int i = 0; i < NumberOfWeights; i++) {
        copyCenter[i].resize(dimension);
        for (int j = 0; j < dimension; j++)
            copyCenter[i][j] = 0;
    }

    int cnt;
    int cn = 1;
    int j = 0;
    for (int k = 0; k < getNumberOfWeights(); k++) {
        cnt = 0;
        while (j < copychrom[0].size()) {
            if (cn % train->cols() == 0) {
                copyCenter[k][cnt] = copychrom[index][j];
                cn++;
                cnt++;
                j++;
                break;
            }
            copyCenter[k][cnt] = copychrom[index][j];
            cn++;
            cnt++;
            j++;
        }
    }

    return copyCenter;
}


double rbf::genTrainSumSquaredError(Matrix genCenter) {
    Data summ;
    summ.resize(genCenter.size());
    double cnt;
    double sum = 0;
    for (int i = 0; i < train->rows(); i++) {
        summ.clear();
        cnt = 0;
        Data x = train->getX(i);
        for (int j = 0; j < genCenter.size(); j++) {
            summ.push_back(gauss(x, genCenter[j], variance[j]));
        }
        for (int k = 0; k < summ.size(); k++) {
            cnt += weights[k] * summ[k];
        }
        sum += pow(train->getY(i) - cnt, 2);
    }
    return 100 * sum / train->rows();
}

void rbf::setBounds() {
    Left.resize((train->cols() + 1) * getNumberOfWeights());
    Right.resize((train->cols() + 1) * getNumberOfWeights());
    F_constant = 1.5;
    int k = getNumberOfWeights();
    int m = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < train->cols(); j++) {
            Left.at(m) = -F_constant * centers[i][j];
            Right.at(m) = F_constant * centers[i][j];
            m = m + 1;
        }
        Left.at(m) = -F_constant * variance[i];
        Right.at(m) = F_constant * variance[i];
        m = m + 1;
    }
    //bounds gia ta chromosomes
    srand(time(0));
//    int random1 = rand() % Right.size();
//    int random2 = rand() % Left.size();
//
//    maxRight = Right[random1];
//    minLeft = Left[random2];
//
//    if (isEqual(maxRight, F_constant * variance[0])) {
//        maxRight = findNthLargestElement(Right, 2);
//    }
//
//    if (isEqual(minLeft, -F_constant * variance[0])) {
//        minLeft = findNthMinimumElement(Left, 2);
//    }
}


void rbf::setChromosomes(int n, double max, double min) {
    chromosomes.resize(n);
    for (int i = 0; i < chromosomes.size(); i++)
        chromosomes[i].resize((train->cols() + 1) * getNumberOfWeights());
    int m = 0;
    double lower_bound = min;
    double upper_bound = max;
    double a_random_double;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;
    re.seed(time(NULL));
    for (int i = 0; i < n; i++) {
        m = 0;
        for (int j = 0; j < getNumberOfWeights(); j++) {
            for (int k = 0; k < train->cols(); k++) {
                a_random_double = unif(re);
                chromosomes[i][m] = a_random_double;
                m = m + 1;
            }
            chromosomes[i][m] = variance[0];
            m = m + 1;
        }
    }
}

double rbf::Delta(int iter, int iterMax, double y, double r) {
    double b = 5;

    double result;

    result = y * (1 - pow(r, pow((1 - (double) iter / (double) iterMax), b)));
    return result;
}

vector<int> rbf::findBestIndices(vector<double> times, const int &N) {
    vector<int> indices(times.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + N, indices.end(),
                      [&times](int i, int j) { return times[i] < times[j]; });

    return vector<int>(indices.begin(), indices.begin() + N);
}

vector<int> rbf::findWorstIndices(vector<double> times, const int &N) {
    vector<int> indices(times.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + N, indices.end(),
                      [&times](int i, int j) { return times[i] > times[j]; });

    return vector<int>(indices.begin(), indices.begin() + N);
}

void rbf::runGen(int n) {
    maxRight = 0;
    Data decodedRight = decodeChild(Right);
    for (int i = 0; i < decodedRight.size(); i++) {
        if (decodedRight[i] > maxRight)
            maxRight = decodedRight[i];
    }
    minLeft = 1e6;
    Data decodedLeft = decodeChild(Left);
    for (int i = 0; i < decodedLeft.size(); i++) {
        if (decodedLeft[i] < minLeft)
            minLeft = decodedLeft[i];
    }
    setChromosomes(n, maxRight, minLeft);
    decode(variance[0]);
    double selectionRate = 0.2;
    int GenSize = abs(1 - selectionRate * n);
    Data sum;
    double lower_bound = -0.5;
    double upper_bound = 1.5;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;
    re.seed(time(NULL));
    double a = unif(re);
    double low = 0;
    double up = 1;
    std::uniform_real_distribution<double> uni(low, up);
    std::default_random_engine r;
    r.seed(time(NULL));
    double r_constant = 0.5;
    double t;
    int T;
    vector<int> min;
    Data childPlaceholder;
    vector<int> indexes;
    Matrix holder;
    Data summ;

    for (int iter = 0; iter < 45; iter++) {

        sum.clear();
        min.clear();
        indexes.clear();
        holder.clear();
        summ.clear();

        for (int i = 0; i < n; i++) {
            double sqe = genTrainSumSquaredError(normalize(i));
            sum.push_back(sqe);
//            cout<<"SSE is : "<<sqe<<endl;
        }

        min = findBestIndices(sum, GenSize);

        // 1) Number of parents that will be used. -> (even numbers within the range of [2,n-GenSize])
        T = ((2 / 2) + rand() % ((copychrom.size() - GenSize - 2 + 2) / 2)) * 2;
        sort(min.begin(), min.end(), greater<int>());

        for (int i = 0; i < min.size(); i++) {
            holder.push_back(copychrom[min[i]]);
            copychrom.erase(copychrom.begin() + min[i]);
        }

        // 2) Get the #T best chromosomes of the remainings based on their fitness.
        // Get the indexes of the chromosome to be selected for crossover.
        for (int i = 0; i < copychrom.size(); i++) {
            double sqe = genTrainSumSquaredError(normalize(i));
            summ.push_back(sqe);
        }
        indexes = findBestIndices(summ, T);

        // 3) Perform Whole Arithmetic Recombination on the selected parents.
        for (int i = 1; i < indexes.size(); i += 2) {
            for (int j = 0; j < copychrom[0].size(); j++) {
                copychrom[indexes[i]][j] = (copychrom[indexes[i]][j] * a) + ((1 - a) * copychrom[indexes[i - 1]][j]);
                copychrom[indexes[i - 1]][j] = copychrom[indexes[i - 1]][j] * a + (1 + a) * copychrom[indexes[i]][j];;
            }
        }

        // Reassign best chromosomes
        for (int i = 0; i < holder.size(); i++)
            copychrom.push_back(holder[i]);

        //mutation procedure
        for (int i = 0; i < copychrom.size(); i++) {
            for (int j = 0; j < copychrom[i].size(); j++) {
                t = uni(r);
                if (t > 0.5) {
                    double delta = Delta(iter, 45, decodedRight[j] - copychrom[i][j], r_constant);
                    copychrom[i][j] = copychrom[i][j] + delta;
                } else {
                    double delta = Delta(iter, 45, copychrom[i][j] - decodedLeft[j], r_constant);
                    copychrom[i][j] = copychrom[i][j] - delta;
                }
            }
        }
    }
}

void rbf::RbfTrain() {
    kmeans alg(train, NumberOfWeights);
    alg.run();
    //alg.CalculateVariance();
    //alg.calculateVariance();
    alg.calculateGenVariance();
    centers = alg.getCenters();
    variance = alg.getVariance();

    Matrix A;
    A.resize(count);
    for (int i = 0; i < A.size(); i++)
        A[i].resize(NumberOfWeights);

    for (int i = 0; i < train->rows(); i++) {
        Data x = train->getX(i);
        for (int j = 0; j < NumberOfWeights; j++)
            A[i][j] = gauss(x, centers[j], variance[j]); //? idia variance h diaforetiko variance gia kathe center?
    }

    Matrix AA = Pseudo_Inverse(A); // ean o A htan [20][10] o AA einai [10][20]

    Matrix output;
    output.resize(train->rows());
    for (int i = 0; i < train->rows(); i++)
        output[i].push_back(train->getY(i)); // o output einai [20][1]

    Matrix result = Multiply(AA,
                             output); // epomenws o result tha einai [10][1] kai ekxwroume ta apotelesmata sto weights

    for (int i = 0; i < result.size(); i++)
        weights[i] = result[i][0];


}

Data rbf::decodeChild(Data child) {
    vector<double>::iterator it = child.begin();
    for (int i = 0; i < child.size(); i++) {
        while (it != child.end()) {
            if ((*it) == F_constant * variance[0]) {
                it = child.erase(it);
            } else {
                it++;
            }
        }
    }
    return child;
}
