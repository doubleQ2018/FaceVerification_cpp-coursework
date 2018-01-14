#include"get_feature.cpp"
#include "adaboost.h"

int main(){
    string train_data = "/Users/zhangqi/STUDY/qq/data/pre_data/pairsDevTrain.txt";
    string test_data = "/Users/zhangqi/STUDY/qq/data/pre_data/pairsDevTest.txt";
    FeatureProcessor photos(train_data, test_data);
    photos.LoadPair();
    photos.GetFeature();
    vector<vector<double> > features = photos.OutFeature();
    vector<vector<double> > test_features = photos.OutTestFeature();
    vector<int> label(1100, 1);
    vector<int> tmp(1100, -1);
    label.insert(label.end(), tmp.begin(), tmp.end());
    AdaBoost adb(features, label, test_features);
    adb.Processor();
    adb.TestProcessor();
    adb.SaveResult();
}
