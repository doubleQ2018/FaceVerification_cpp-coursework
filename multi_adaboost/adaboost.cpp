#include"adaboost.h"
#include<iostream>
#include<fstream>
#include<math.h>
#include<random>

using namespace std;

AdaBoost::AdaBoost(vector<vector<double> > x, vector<int> y, vector<vector<double> > test, int m, int t1, int k1)
{
    features = x;
    test_features = test;
    label = y;
    size = x.size();
    D = x[0].size();
    T = t1;
    K = k1;
    for(int i = 0; i < size; i++)
    {
        if(label[i] < 0) 
        {
            pnums = i;
            break;
        }
    }
    nnums = size - pnums;
    M = m;
    spnums = pnums / M;
    snnums = nnums / M;

    // initiate dd, d2d
    for(int i = 0; i < M; i++)
    {
        vector<int> vec;
        d2d.push_back(vec);
    }
    for(int i = 0; i < pnums; i++) 
    {
        int mm = i / spnums;
        dd.push_back(mm);
        d2d[mm].push_back(i);
    }
    for(int i = pnums; i < size; i++) 
    {
        int mm = (i - pnums) / snnums;
        dd.push_back(mm);
        d2d[mm].push_back(i);
    }
    
    // initiate alphas, dimendions, thresholds
    for(int i = 0; i < K; i++)
    {
        fa.push_back(0.0);
        vector<double> vec;
        alphas.push_back(vec);
        dimensions.push_back(vec);
        thresholds.push_back(vec);
        vector<double> vec1(size, 0);
        weights.push_back(vec1);
    }
    
    // initiate q, weights
    default_random_engine generator;  
    uniform_int_distribution<int> dis(0,K-1);  
    for(int i = 0; i < M; i++)
    {
        vector<double> vec(K, 0.0);
        int k = dis(generator);
        vec[k] = 1;
        q.push_back(vec);
        for(int j = 0; j < spnums; j++) weights[k][d2d[i][j]] = 1.0 / spnums;
        for(int j = spnums; j < spnums + snnums; j++) weights[k][d2d[i][j]] = - 1.0 / snnums;
    }
    for(int i = 0; i < K; i++)
    {
        double sums = 0;
        for(int j = 0; j < size; j++) sums += fabs(weights[i][j]);
        for(int j = 0; j < size; j++) weights[i][j] /= sums;
    }
    cout<<"q:"<<endl;
    for(int i = 0; i < q.size(); i++)
    {
        for(auto x : q[i]) cout << x <<" ";
        cout<<endl;
    }   
    cout<<"w:"<<endl;
    for(int i = 0; i < weights.size(); i++)
    {
        for(auto x : weights[i]) cout << x <<" ";
        cout<<endl;
    }   
    /*

    for(int i = 0; i < size; i++)
    {
        if(i < pnums) weights.push_back(1.0/(pnums + nnums));
        else weights.push_back(1.0/(pnums + nnums));
    }*/
}

void AdaBoost::Processor()
{
    TrainProcessor();
}
void AdaBoost::TrainProcessor()
{
    for(int i = 0; i < T; i++)
    {
        cout<<"iteration: "<<(i+1)<<endl;
        for(int j = 0; j < K; j++)
        {
            cout<<"K = "<<(j+1)<<endl;
            RenewWeight(j);
            RenewAlpha(j);
            alphas[j].push_back(tmpAlpha);
            dimensions[j].push_back(tmpDim);
            thresholds[j].push_back(tmpRu);
        }

        // Renew fa
        for(int i = 0; i < K; i++)
        {
            double tmpFa = 0;
            for(int j = 0; j < M; j++) tmpFa += q[j][i];
            fa[i] = tmpFa;
        }

        cout<<"fa:"<<endl;
        for(int i = 0; i < fa.size(); i++)
            cout << fa[i] <<" ";
        cout<<endl;
        // Renew q
        for(int i = 0; i < M; i++)
        {
            vector<int> candidate = d2d[i]; 
            vector<double> tmp;
            double sums = 0;
            for(int j = 0; j < K; j++)
            {
                double s = 1;
                for(int t = 0; t < pnums / M; t++) 
                    s *= 1 / (1 + exp(-tmpS(candidate[t], j)));
                for(int t = pnums / M; t < candidate.size(); t++) 
                {
                    double x = exp(-tmpS(candidate[t], j));
                    s *= x / (1 + x);
                }
                sums += s * fa[j];
                tmp.push_back(s * fa[j]);
            }
            //cout<<"tmp:"<<endl;
            //for(int j = 0; j < K; j++) cout<<tmp[j]<<" ";
            //cout<<endl;
            for(int j = 0; j < K; j++) q[i][j] = tmp[j] / sums;
        }
        
        cout<<"q:"<<endl;
        for(int i = 0; i < q.size(); i++)
        {
            for(auto x : q[i]) cout << x <<" ";
            cout<<endl;
        }   

        // Renew w
        for(int i = 0; i < K; i++)
        {
            double sums = 0;
            for(int j = 0; j < pnums; j++)
            {
                double x = exp(-tmpS(j, i));
                weights[i][j] = q[dd[j]][i] * x / (1 + x);
                sums += weights[i][j];
            }
            for(int j = pnums; j < size; j++) 
            {
                weights[i][j] = -q[dd[j]][i] / (1 + exp(-tmpS(j, i)));
                sums += -weights[i][j];
            }
            for(int j = 0; j < size; j++) weights[i][j] /= sums;
        }
        
        cout<<"w:"<<endl;
        for(int i = 0; i < weights.size(); i++)
        {
            for(auto x : weights[i]) cout << x <<" ";
            cout<<endl;
        }

        //done

    }
}

bool cmp(const pair<double, double> x, const pair<double, double> y) 
{
    return x.first < y.first;
}


void AdaBoost::RenewWeight(int k)
{
    double maxp = -99999;
    for(int d = 0; d < D; d++)  //change
    {
        vector<pair<double, double> > IndexAndWeight;
        for(int i = 0; i < size; i++)
        {
            //pair<double, double> a(features[i][d], -label[i] * weights[k][i]);
            pair<double, double> a(features[i][d], weights[k][i]);
            IndexAndWeight.push_back(a);
        }
        sort(IndexAndWeight.begin(), IndexAndWeight.end(), cmp);
        //for(auto x:IndexAndWeight) cout<<"("<<x.first<<", "<<x.second<<"), ";
        //cout<<endl;
        double p = 0;
        for(int i = 0; i < size; i++) p -= weights[k][i];
        //for(int i = 0; i < pnums; i++) p += weights[k][i];
        int t = 0;
        while(IndexAndWeight[t].first == 0 && t < size) {p += IndexAndWeight[t].second; t++;}
        if(maxp <= p && t > 0) 
        {
            maxp = p;
            tmpDim = d;
            tmpRu = IndexAndWeight[t-1].first;
        }
        for(int i = t; i < size; i++)
        {
            p += IndexAndWeight[i].second;
            if(maxp <= p) 
            {
                maxp = p;
                tmpDim = d;
                tmpRu = IndexAndWeight[i].first;
            }
        }
    }
    //tmpMinError = maxp;
    //cout<<"maxp = "<<maxp<<endl;
    cout<<"choose dimension = "<<tmpDim<<endl;
    //for(int i = 0; i < size; i++) cout<<features[i][tmpDim]<<" ";
    //cout<<endl;
    cout<<"choose ru = "<<tmpRu<<endl;

            
}

double AdaBoost::tmpS(int n, int k) // h(k) for candidate n
{
    double h = 0;
    for(int t = 0; t < alphas[k].size(); t++)
        h += alphas[k][t] * (features[n][dimensions[k][t]] < thresholds[k][t] ? 1 : -1);
    return h;
}


double AdaBoost::CalculateMLE(double alpha, int j)
{

    double mle = 0;
    for(int i = 0; i < pnums; i++)
    {
        double x = - tmpS(i, j) - alpha * (features[i][tmpDim] < tmpRu ? 1 : -1);
        //cout<<"i = "<<(i+1)<<", feature = "<<features[i][tmpDim]<<", h(m, n) = "<<x<<endl;
        //cout<<(features[i][tmpDim] < tmpRu ? 1 : -1)<<endl;
        mle += q[dd[i]][j] * log(1 / (1 + exp(x)));   
        //cout<<"q = "<<q[dd[i]][j]<<", mle = "<<mle<<endl;
    }
    for(int i = pnums; i < size; i++)
    {
        double x = - tmpS(i, j) - alpha * (features[i][tmpDim] < tmpRu ? 1 : -1);
        //cout<<"i = "<<(i+1)<<", feature = "<<features[i][tmpDim]<<", h(m, n) = "<<x<<endl;
        //cout<<(features[i][tmpDim] < tmpRu ? 1 : -1)<<endl;
        mle += q[dd[i]][j] * log(exp(x) / (1 + exp(x)));
        //cout<<"q = "<<q[dd[i]][j]<<", mle = "<<mle<<endl;
    }
    return mle;
}
    
void AdaBoost::RenewAlpha(int j)
{
    double MaxMle = -999999;
    for(double s = 0.001; s < 1; s += 0.001) // step=0.001 for searching
    {
        double m = CalculateMLE(s, j);
        //cout<<"s = "<<s<<", MLE = "<<m<<endl;
        if(m > MaxMle)
        {
            MaxMle = m;
            tmpAlpha = s;
        }
    } 
    cout<<"choose alpha = "<<tmpAlpha<<endl;
}


double AdaBoost::TestS(vector<double> tf, int k) // h(k) for test candidate n
{
    double h = 0;
    for(int t = 0; t < alphas[k].size(); t++)
        h += alphas[k][t] * (tf[dimensions[k][t]] < thresholds[k][t] ? 1 : -1);
    return h;
}


void AdaBoost::TestProcessor()
{
   for(int i = 0; i < features.size(); i++)
   {
       double plikely = 0, nlikely = 0;
       for(int j = 0; j < K; j++)
       {
           double x = exp(-TestS(features[i], j));
           plikely += q[dd[i]][j] / (1 + x);                 
           nlikely += q[dd[i]][j] * x / (1 + x);
       }
       vector<double> tmp;
       tmp.push_back(plikely);
       tmp.push_back(nlikely);
       scores.push_back(tmp);
       if(plikely > nlikely) test_result.push_back("1");
       else test_result.push_back("-1");

   }

   int n = test_features.size();
   for(int i = 0; i < n; i++)
   {
       double plikely = 0, nlikely = 0;
       for(int j = 0; j < K; j++)
       {
           double x = exp(-TestS(test_features[i], j));
           plikely += q[dd[i]][j] / (1 + x);                 
           nlikely += q[dd[i]][j] * x / (1 + x);
       }
       vector<double> tmp;
       tmp.push_back(plikely);
       tmp.push_back(nlikely);
       scores.push_back(tmp);
       if(plikely > nlikely) test_result.push_back("1");
       else test_result.push_back("-1");

   } 
}

void AdaBoost::SaveResult()
{
    string result_file = "result_T" + to_string(T) + "_K" + to_string(K) + ".txt";
    ofstream outfile(result_file);
    for(int i = 0; i < test_result.size(); i++) outfile<<test_result[i]<<"  "<<scores[i][0]<<"  "<<scores[i][1]<<endl;
    outfile.close();
}
