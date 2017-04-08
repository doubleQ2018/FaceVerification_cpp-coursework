#include"get_feature.cpp"
#include "adaboost.h"

int main()
{
    FeatureProcessor photos("/Users/zhangqi/STUDY/qq/multi_adaboost/train_data.txt1", "/Users/zhangqi/STUDY/qq/multi_adaboost/test_data.txt1");
    photos.LoadPair();
    photos.GetFeature();
    vector<vector<double> > features = photos.OutFeature();
    vector<vector<double> > test_features = photos.OutTestFeature();
    /*
    cout<<features[0].size()<<endl;
    cout<<test_features.size()<<endl;
    for(int i = 0; i < 20; i++) cout<<features[0][i]<<" ";
    cout<<endl;
    for(int i = 0; i < 20; i++) cout<<test_features[0][i]<<" ";
    cout<<endl;*/
    vector<int> label(50, 1);
    vector<int> tmp(50, -1);
    label.insert(label.end(), tmp.begin(), tmp.end());
    AdaBoost adb(features, label, test_features, 5);
    adb.Processor();
    adb.TestProcessor();
    adb.SaveResult();

}
