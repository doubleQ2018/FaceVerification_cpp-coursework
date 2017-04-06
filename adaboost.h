#include<iostream>
#include<vector>

using namespace std;

class AdaBoost
{
    public:
        AdaBoost(vector<vector<double> > x, vector<int> y, vector<vector<double> > test);
        void Processor();
        void TestProcessor();
        void SaveResult();

    private:
        int T;
        vector<vector<double> > features;
        vector<vector<double> > test_features;
        vector<int> label;
        int size;
        int D;
        int pnums;
        int nnums;
        double tmpRu;
        double tmpBeta;
        double tmpMinError;
        int tmpDim;
        vector<double> weights;
        vector<double> alphas;
        vector<double> dimensions;
        vector<double> thresholds;
        vector<string> test_result;
        virtual int S(double z, double ru) {return z <= ru ? 1 : -1;}
        void TrainProcessor();
        void RenewWeight();


};
