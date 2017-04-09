#include<iostream>
#include<vector>

using namespace std;

class AdaBoost
{
    public:
        AdaBoost(vector<vector<double> > x, vector<int> y, vector<vector<double> > test, int m, int t1, int k1);
        void Processor();
        void TestProcessor();
        void SaveResult();
        double tmpS(int n, int k);
        double CalculateMLE(double alpha, int j);
        double TestS(vector<double> tf, int k);
    private:
        int T;
        vector<vector<double> > features;
        vector<vector<double> > test_features;
        vector<int> label;
        vector<int> dd;  //dict for feature belongs to which candidate
        vector<vector<int> > d2d; //dict for candidate have which features 
        int K;
        int M;
        int size;
        int D;
        int pnums;  // positive features in total
        int nnums;  // negative features in total
        int spnums; // posituve features for one candidate
        int snnums; // negative features for one candidate
        double tmpRu;
        //double tmpBeta;
        double tmpAlpha;
        double tmpMinError;
        int tmpDim;
        vector<vector<double> > weights;// K x size
        vector<vector<double> > q;      // M x K
        vector<double> fa;              // K
        vector<vector<double> > alphas;     // K x T
        vector<vector<double> > dimensions; // K x T
        vector<vector<double> > thresholds; // K x T
        vector<string> test_result;
        vector<vector<double> > scores;
        virtual int S(double z, double ru) {return z <= ru ? 1 : -1;}
        void TrainProcessor();
        void RenewWeight(int k);
        void RenewAlpha(int k);


};
