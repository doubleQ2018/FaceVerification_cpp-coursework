#ifndef _GET_FEATURE_H_
#define _GET_FEATURE_H_

#include<vector>
#include<string>

using namespace std;

class FeatureProcessor
{
    public:
        FeatureProcessor(string f, string t): FFileName(f), TFileName(t){}
        void LoadPair();
        void GetFeature();
        vector<vector<double>> OutFeature();
        vector<vector<double>> OutTestFeature();
    private:
        string FFileName, TFileName;
        vector<pair<string, string>> fpair, tpair;
        vector<vector<double>> feature, test_feature;
};
#endif //_GET_FEATURE_H_
