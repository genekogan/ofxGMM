#include "ofxGMM.h"


void ofxGMM::setup(int dim, int mixNum) {
    this->dim = dim;
    this->mixNum = mixNum;
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
    
    gmm.setup(dim, mixNum);
    gmm.Train(data, numSamples); //Training GMM
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

void ofxGMM::save(string path) {
    const char *filepath = path.c_str();
	ofstream gmm_file(filepath);
	assert(gmm_file);
	gmm_file << gmm;
	gmm_file.close();

}

void ofxGMM::load(string path) {
}
