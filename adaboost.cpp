#include"adaboost.h"
#include<iostream>
#include<fstream>
#include<math.h>

using namespace std;

void AdaBoost::init(){
    size = features.size();
    D = features[0].size();     // features dimension
    vector<int>::iterator p = find_if(label.begin(), label.end(), [](const int &x){return x < 0;})
    pnums = p - label.begin();  // positive case number 
    nnums = size - pnums;       // negative case number
    weights = vector<double>(size, 1.0 / size); // weighs initiate with 1/size
}

void AdaBoost::Processor(){
    TrainProcessor();
}
void AdaBoost::TrainProcessor(){
    for(int i = 0; i < T; i++){
        cout<<"iteration = "<<(i+1)<<endl;
        // update weights
        RenewWeight();
        if(tmpMinError == 0){
            alphas.push_back(100);
            dimensions.push_back(tmpDim);
            thresholds.push_back(tmpRu);
            return;
        }
        double sums = 0;
        // update weight for present dimendion of feature for positive case when right
        // according to present threshold(< threshold)
        for(int j = 0; j < pnums; j++i){ 
            if(features[j][tmpDim] <= tmpRu) weights[j] *= tmpBeta;
            sums += weights[j];
        }
        // update weight for present dimendion of feature for negative case when right
        // according to present threshold(> threshold)
        for(int j = pnums; j < size; j++){
            if(features[j][tmpDim] > tmpRu) weights[j] *= tmpBeta;
            sums += weights[j];
        }
        // weights normalization, sums to 1
        for(int j = 0; j < size; j++) weights[j] /= sums;
        // record the parameter
        alphas.push_back(log(1/tmpBeta));
        dimensions.push_back(tmpDim);
        thresholds.push_back(tmpRu);
        cout<<"choose tmpMinError = "<<tmpMinError<<endl;
        cout<<"choose tmpBeta = "<<tmpBeta<<endl;
        cout<<"choose dimension = "<<tmpDim<<endl;
        cout<<"choose thresholds = "<<tmpRu<<endl;
    }
}

void AdaBoost::RenewWeight(){
    // update all parameters according adaboost rules
    double MinError = 1;
    for(int d = 0; d < D; d++){
        vector<pair<double, double> > IndexAndWeight;
        for(int i = 0; i < size; i++){
            pair<double, double> a(features[i][d], -label[i] * weights[i]);
            IndexAndWeight.push_back(a);
        }
        sort(IndexAndWeight.begin(), IndexAndWeight.end(), [](const pair<double, double> &x, const pair<double, double> &y){return x.first < y.first;});
        //for(auto x:IndexAndWeight) cout<<x.first<<" "<<x.second<<endl;
        //for(int i = 0; i < 100; i++) cout<<IndexAndWeight[i].second<<" ";
        double error = 0;
        for(int i = 0; i < pnums; i++) error += weights[i];
        int t = 0;
        while(IndexAndWeight[t].first == 0 && t < size) {
            error += IndexAndWeight[t].second; 
            t++;
        }
        if(MinError > error && t > 0) {
            MinError = error;
            tmpDim = d;
            tmpRu = IndexAndWeight[t-1].first;
            tmpBeta = MinError / (1 - MinError);
        }
        //cout<<"error = "<<error<<endl;
        for(int i = t; i < size; i++){
            error += IndexAndWeight[i].second;
            if(MinError > error) {
                MinError = error;
                tmpDim = d;
                tmpRu = IndexAndWeight[i].first;
                tmpBeta = MinError / (1 - MinError);
            }
        }
    }
    tmpMinError = MinError;
}

void AdaBoost::TestProcessor(){
   int n = test_features.size();
   for(int i = 0; i < n; i++){
       double score = 0;
       for(int j = 0; j < T; j++) 
           score += alphas[j] * S(test_features[i][dimensions[j]], thresholds[j]);
       cout<<score<<endl;
       test_result.push_back(score > 0 ? "1": "-1");
   } 
}

void AdaBoost::SaveResult(){
    ofstream outfile("result.txt");
    for(int i = 0; i < test_result.size(); i++) outfile<<test_result[i]<<endl;
    outfile.close();
}
