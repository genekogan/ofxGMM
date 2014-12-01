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

    int getDimension() {return dim;}
    int getNumGaussians() {return numGaussians;}
    
    // get gaussians
    vector<double> getMean(int mixNumber);
    vector<double> getVariance(int mixNumber);
    vector<double> getStandardDeviation(int mixNumber);
    double getPrior(int mixNumber);
    
    // get a single probability
    float getProbability(vector<double> sample);
    
    // sample from GMM
    vector<double> getRandomSample(int idxMix=-1);
    
    // save/load gmm to file
    void save(string path);
    void load(string path);
    
private:
    
    float getRandomGaussian();
    
    vector<vector<double> > samples;
    GMM gmm;
    int dim;
    int numGaussians;

};