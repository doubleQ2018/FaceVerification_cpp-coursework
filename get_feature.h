#ifndef _GET_FEATURE_H_
#define _GET_FEATURE_H_

#include<vector>
#include<string>

using namespace std;

class FeatureProcessor
{
    public:
        FeatureProcessor(string f, string t);
        void LoadPair();
        void GetFeature();
        vector<vector<double> > OutFeature();
        vector<vector<double> > OutTestFeature();
    private:
        string FFileName;
        string TFileName;
        vector<pair<string, string> > fpair;
        vector<pair<string, string> > tpair;
        vector<vector<double> > feature;
        vector<vector<double> > test_feature;
};
#endif //_GET_FEATURE_H_
