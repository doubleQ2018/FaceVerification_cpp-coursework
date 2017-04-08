#include"lbp.cpp"
#include<iostream>
#include<vector>
#include<string>
#include<math.h>
#include<fstream>
#include<sstream>
#include <iomanip>

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

FeatureProcessor::FeatureProcessor(string f, string t)
{
    FFileName = f;
    TFileName = t;
}

string FixedString(string x)
{
    stringstream strStream;
    strStream<<setw(4)<<setfill('0')<<x;
    return strStream.str() + ".jpg";
}
void FeatureProcessor::LoadPair()
{
    ifstream in(FFileName);
    string line;
    int i = 0;
    while(getline(in, line))
    {
        if(i < 50)
        {
            istringstream inn(line);
            string x, a, b;
            inn >> x >> a >> b;
            pair<string, string> tmp(x + "/" + x + "_" + FixedString(a), x + "/" + x + "_" + FixedString(b));
            fpair.push_back(tmp);
        }
        else
        {
            istringstream inn(line);
            string x, y, a, b;
            inn >> x >> a >> y >> b;
            pair<string, string> tmp(x + "/" + x + "_" + FixedString(a), y + "/" + y + "_" + FixedString(b));
            fpair.push_back(tmp);
        }
        i++;
    }
    ifstream in1(TFileName);
    string line1;
    i = 0;
    while(getline(in1, line1))
    {
        if(i < 50)
        {
            istringstream inn(line1);
            string x, a, b;
            inn >> x >> a >> b;
            pair<string, string> tmp(x + "/" + x + "_" + FixedString(a), x + "/" + x + "_" + FixedString(b));
            tpair.push_back(tmp);
        }
        else
        {
            istringstream inn(line1);
            string x, y, a, b;
            inn >> x >> a >> y >> b;
            pair<string, string> tmp(x + "/" + x + "_" + FixedString(a), y + "/" + y + "_" + FixedString(b));
            tpair.push_back(tmp);
        }
        i++;
    }

}

vector<double> calculate(vector<double> &x, vector<double> y)
{
    for(int i = 0; i < x.size(); i++)
    {
        x[i] = x[i] + y[i] == 0 ? 0 : pow(x[i] - y[i], 2) / (x[i] + y[i]);
    }
    return x;
}
void FeatureProcessor::GetFeature()
{
    string path = "/Users/zhangqi/STUDY/qq/data/pre_data/";
    cout<<"Preparint train set..."<<endl;
    for(int i = 0; i < fpair.size(); i++)
    //for(int i = 1790; i < 1810; i++)
    {
        //cout<<fpair[i].first<<endl;
        //cout<<fpair[i].second<<endl;
        LBPextractor A(path + fpair[i].first);
        LBPextractor B(path + fpair[i].second);
        vector<double> x1 = A.getFeature();
        vector<double> x2 = B.getFeature();
        vector<double> f = calculate(x1, x2);
        feature.push_back(f);
    }
    cout<<"Preparing test set..."<<endl;
    for(int i = 0; i < tpair.size(); i++)
    //for(int i = 1790; i < 1810; i++)
    {
        //cout<<tpair[i].first<<endl;
        //cout<<tpair[i].second<<endl;
        LBPextractor A(path + tpair[i].first);
        LBPextractor B(path + tpair[i].second);
        vector<double> x1 = A.getFeature();
        vector<double> x2 = B.getFeature();
        vector<double> f = calculate(x1, x2);
        test_feature.push_back(f);
    }
}

vector<vector<double> > FeatureProcessor::OutFeature() {return feature;}
vector<vector<double> > FeatureProcessor::OutTestFeature() {return test_feature;}

/*
int main()
{
    string name = "pairsDevTrain.txt";   
    FeatureProcessor A(name);
    A.LoadPair();
    vector<vector<double> > res = A.GetFeature();
    cout<<res.size();
}*/
