#pragma once

#include "ofMain.h"
#include "GMM.h"
#include "KMeans.h"


class ofxGMM

{
public:
    void setup(int dim, int mixNum);
    
    void addSample(vector<double> sample);
    void train();
    float getProbability(vector<double> sample);
    
    void save(string path);
    void load(string path);
    

    
private:
    
    vector<vector<double> > samples;
    GMM gmm;
    int dim;
    int mixNum;

};






// kmeans
/*
 const int size = 10; //Number of samples
 const int dim = 3;   //Dimension of feature
 const int cluster_num = 4; //Cluster number
 
 KMeans* kmeans = new KMeans(dim,cluster_num);
 int* labels = new int[size];
 kmeans->SetInitMode(KMeans::InitUniform);
 kmeans->Cluster(data,size,labels);
 
 
 printf("Clustering result by k-meams:\n");
 for(int i = 0; i < size; ++i)
 {
 printf("%f, %f, %f belongs to %d cluster\n", data[i*dim+0], data[i*dim+1], data[i*dim+2], labels[i]);
 }
 
 delete []labels;
 delete kmeans;
 */

