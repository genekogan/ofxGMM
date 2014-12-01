#include "ofxGMM.h"


void ofxGMM::setup(int dim, int numGaussians) {
    this->dim = dim;
    this->numGaussians = numGaussians;
    samples.clear();
    gmm.setup(dim, numGaussians);
}

void ofxGMM::addSample(vector<double> sample) {
    if (sample.size() != dim) {
        ofLog(OF_LOG_ERROR, "Error: sample size ("+ofToString(sample.size())+") doesn't match set dimension ("+ofToString(dim)+")");
        return;
    }
    samples.push_back(sample);
}

void ofxGMM::train() {
    int numSamples = samples.size();
    double data[dim * numSamples];
    int idx = 0;
    for (int s=0; s<samples.size(); s++) {
        for (int i=0; i<samples[s].size(); i++) {
            data[idx++] = samples[s][i];
        }
    }
    gmm.Train(data, numSamples); //Training GMM
}

void ofxGMM::setGaussians(vector<double> priors, vector<vector<double> > means, vector<vector<double> > variances) {
    if (priors.size() != means.size() || priors.size() != variances.size()) {
        ofLog(OF_LOG_ERROR, "Error: priors, means, and variances must have same number of gaussians");
        return;
    }
    numGaussians = priors.size();
    dim = means[0].size();
    
    setup(dim, numGaussians);
    for (int i=0; i<numGaussians; i++) {
        gmm.setPrior(i, priors[i]);
        double mean[dim];
        double variance[dim];
        for (int j=0; j<dim; j++) {
            mean[j] = means[i][j];
            variance[j] = variances[i][j];
        }
        gmm.setMean(i, mean);
        gmm.setVariance(i, variance);
    }
}

vector<double> ofxGMM::getMean(int mixNumber) {
    double * m = gmm.Mean(mixNumber);
    vector<double> mean;
    for (int j=0; j<dim; j++) {
        mean.push_back(m[j]);
    }
    return mean;
}

vector<double> ofxGMM::getVariance(int mixNumber) {
    double * var = gmm.Variance(mixNumber);
    vector<double> variance;
    for (int j=0; j<dim; j++) {
        variance.push_back(var[j]);
    }
    return variance;
}

vector<double> ofxGMM::getStandardDeviation(int mixNumber) {
    double * var = gmm.Variance(mixNumber);
    vector<double> standardDeviation;
    for (int j=0; j<dim; j++) {
        standardDeviation.push_back(sqrt(var[j]));
    }
    return standardDeviation;
}

double ofxGMM::getPrior(int mixNumber) {
    return gmm.Prior(mixNumber);
}

float ofxGMM::getProbability(vector<double> sample) {
    if (sample.size() != dim) {
        ofLog(OF_LOG_ERROR, "Error: sample size ("+ofToString(sample.size())+") doesn't match set dimension ("+ofToString(dim)+")");
        return;
    }
    double testData[sample.size()];
    for (int i=0; i<sample.size(); i++) {
        testData[i] = (double) sample[i];
    }
    return gmm.GetProbability(testData);
}

vector<double> ofxGMM::getRandomSample(int idxMix) {
    // first select which gaussian to sample from if none given as argument (default)
    if (idxMix == -1) {
        idxMix = 0;
        float cumulativePrior = 0.0;
        float randn = ofRandom(1);
        while (cumulativePrior < randn) {
            cumulativePrior += gmm.Prior(idxMix++);
        }
        idxMix--;
    }
    double * m = gmm.Mean(idxMix);
    double * var = gmm.Variance(idxMix);
    
    vector<double> randomSample;
    randomSample.resize(dim);
    for (int i=0; i<dim; i++) {
        randomSample[i] = m[i] + getRandomGaussian() * sqrt(var[i]);
    }
    return randomSample;
}

void ofxGMM::save(string path) {
    const char *filepath = path.c_str();
	ofstream gmm_file(filepath);
	assert(gmm_file);
	gmm_file << gmm;
	gmm_file.close();

}

void ofxGMM::load(string path) {
    const char *filepath = path.c_str();
	ifstream gmm_file(filepath);
	assert(gmm_file);
	gmm_file >> gmm;
	gmm_file.close();
}

// implementation of Box–Muller method for sampling a random gaussian
// taken from https://github.com/andyr0id/ofxGaussian/blob/master/src/ofxGaussian.cpp
float ofxGMM::getRandomGaussian() {
    float v1, v2, s;
    do {
        v1 = 2 * ofRandomf() - 1;
        v2 = 2 * ofRandomf() - 1;
        s = v1 * v1 + v2 * v2;
    }
    while (s >= 1 || s == 0);
    float multiplier = sqrt(-2 * log(s)/s);
    return v1 * multiplier;
}
