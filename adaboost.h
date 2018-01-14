#ifndef _ADABOOST_H_
#define _ADABOOST_H_

#include<iostream>
#include<vector>

using namespace std;

class AdaBoost{
    public:
        AdaBoost(vector<vector<double>> x, vector<int> y, vector<vector<double>> test): features(x), label(y), test_features(test), T(50){}
        void init();
        void Processor();
        void TestProcessor();
        void SaveResult();

    private:
        int T;
        vector<vector<double>> features, test_features;
        vector<int> label;
        int size, D;
        int pnums, nnums;
        double tmpRu, tmpBeta, tmpMinError;
        int tmpDim;
        vector<double> weights, alphas, dimensions, thresholds;
        vector<string> test_result;
        void TrainProcessor();
        void RenewWeight();
        virtual int S(double z, double ru) {return z <= ru ? 1 : -1;}
};
#endif //_ADABOOST_H_
