#pragma once

#include "ofMain.h"
#include "GMM.h"
#include "KMeans.h"


class ofxGMM

{
public:
    void setup(int dim, int numGaussians);
    
    void addSample(vector<double> sample);
    void train();
    
    void setGaussians(vector<double> priors, vector<vector<double> > means, vector<vector<double> > variances);

    // get gaussians
    vector<double> getMean(int mixNumber);
    vector<double> getVariance(int mixNumber);
    vector<double> getStandardDeviation(int mixNumber);
    double getPrior(int mixNumber);

    // get a single probability
    float getProbability(vector<double> sample);

    void save(string path);
    void load(string path);
    
private:
    
    vector<vector<double> > samples;
    GMM gmm;
    int dim;
    int numGaussians;

};