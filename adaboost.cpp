#include"adaboost.h"
#include<iostream>
#include<fstream>
#include<math.h>

using namespace std;

AdaBoost::AdaBoost(vector<vector<double> > x, vector<int> y, vector<vector<double> > test)
{
    features = x;
    test_features = test;
    label = y;
    size = x.size();
    D = x[0].size();
    T = 50;
    for(int i = 0; i < size; i++)
    {
        if(label[i] < 0) 
        {
            pnums = i;
            break;
        }
    }
    nnums = size - pnums;
    for(int i = 0; i < size; i++)
    {
        if(i < pnums) weights.push_back(1.0/(pnums + nnums));
        else weights.push_back(1.0/(pnums + nnums));
    }
}

void AdaBoost::Processor()
{
    TrainProcessor();
}
void AdaBoost::TrainProcessor()
{
    for(int i = 0; i < T; i++)
    {
        cout<<"iteration = "<<(i+1)<<endl;
        RenewWeight();
        if(tmpMinError == 0)
        {

            alphas.push_back(100);
            dimensions.push_back(tmpDim);
            thresholds.push_back(tmpRu);
            return;
        }
        double sums = 0;
        for(int j = 0; j < pnums; j++)
        {
            if(features[j][tmpDim] <= tmpRu) weights[j] *= tmpBeta;
            sums += weights[j];
        }
        for(int j = pnums; j < size; j++)
        {
            if(features[j][tmpDim] > tmpRu) weights[j] *= tmpBeta;
            sums += weights[j];
        }
        for(int j = 0; j < size; j++) weights[j] /= sums;
        alphas.push_back(log(1/tmpBeta));
        dimensions.push_back(tmpDim);
        thresholds.push_back(tmpRu);
        cout<<"choose tmpMinError = "<<tmpMinError<<endl;
        cout<<"choose tmpBeta = "<<tmpBeta<<endl;
        cout<<"choose dimension = "<<tmpDim<<endl;
        cout<<"choose thresholds = "<<tmpRu<<endl;
    }
}
bool cmp(const pair<double, double> x, const pair<double, double> y) 
{
    return x.first < y.first;
}
void AdaBoost::RenewWeight()
{
    double MinError = 1;
    for(int d = 0; d < D; d++)
    {
        vector<pair<double, double> > IndexAndWeight;
        for(int i = 0; i < size; i++)
        {
            pair<double, double> a(features[i][d], -label[i] * weights[i]);
            IndexAndWeight.push_back(a);
        }
        sort(IndexAndWeight.begin(), IndexAndWeight.end(), cmp);
        //for(auto x:IndexAndWeight) cout<<x.first<<" "<<x.second<<endl;
        //for(int i = 0; i < 100; i++) cout<<IndexAndWeight[i].second<<" ";
        double error = 0;
        for(int i = 0; i < pnums; i++) error += weights[i];
        int t = 0;
        while(IndexAndWeight[t].first == 0 && t < size) {error += IndexAndWeight[t].second; t++;}
        if(MinError > error && t > 0) 
        {
            MinError = error;
            tmpDim = d;
            tmpRu = IndexAndWeight[t-1].first;
            tmpBeta = MinError / (1 - MinError);
        }
        //cout<<"error = "<<error<<endl;
        for(int i = t; i < size; i++)
        {
            error += IndexAndWeight[i].second;
            if(MinError > error) 
            {
                MinError = error;
                tmpDim = d;
                tmpRu = IndexAndWeight[i].first;
                tmpBeta = MinError / (1 - MinError);
            }
        }
    }
    tmpMinError = MinError;
            
}
void AdaBoost::TestProcessor()
{
   int n = test_features.size();
   for(int i = 0; i < n; i++)
   {
       double score = 0;
       for(int j = 0; j < T; j++) score += alphas[j] * S(test_features[i][dimensions[j]], thresholds[j]);
       cout<<score<<endl;
       test_result.push_back(score>0 ? "1": "-1");
   } 
}

void AdaBoost::SaveResult()
{
    ofstream outfile("result.txt");
    for(int i = 0; i < test_result.size(); i++) outfile<<test_result[i]<<endl;
    outfile.close();
}
