#include "ofApp.h"


void ofApp::setup() {
    trainGMMFromData();
}

void ofApp::trainGMMFromData() {

    // setup a GMM with 2 dimensions (x,y) and 3 gaussians
    numGaussians = 3;
    gmm.setup(2, numGaussians);
    
    // SAMPLE DATA
    // add 100 samples centered around (220, 150)
    // add 150 samples centered around (430, 570)
    // add 100 samples centered around (820, 320)
    
    for (int i=0; i<100; i++) {
        addSample(220 + ofRandom(-100, 100),
                  150 + ofRandom(-95, 95));
    }
    for (int i=0; i<150; i++) {
        addSample(430 + ofRandom(-180, 180),
                  570 + ofRandom(-120, 120));
    }
    for (int i=0; i<100; i++) {
        addSample(820 + ofRandom(-90, 90),
                  320 + ofRandom(-150, 150));
    }

    // train the GMM from data
    gmm.train();
    
    // get a random sample from the GMM
    randSample = gmm.getRandomSample();
}

void ofApp::setGMMExplicitly() {
    
    // set the GMM explicitly
    vector<double> priors;
    vector<vector<double> > means, variances;
    
    numGaussians = 2;
    gmm.setup(2, numGaussians);
    
    // first gaussian
    float prior1 = 0.4;
    vector<double> mean1, variance1;
    mean1.push_back(150);   // x
    mean1.push_back(500);   // y
    variance1.push_back(pow(130.f, 2));   // x
    variance1.push_back(pow(130.f, 2));   // y

    // second gaussian
    float prior2 = 0.6;
    vector<double> mean2, variance2;
    mean2.push_back(780);   // x
    mean2.push_back(250);   // y
    variance2.push_back(pow(80.f, 2));   // x
    variance2.push_back(pow(100.f, 2));   // y
    
    priors.push_back(prior1);
    priors.push_back(prior2);
    means.push_back(mean1);
    means.push_back(mean2);
    variances.push_back(variance1);
    variances.push_back(variance2);
    
    gmm.setGaussians(priors, means, variances);
}

void ofApp::addSample(double x, double y) {
    vector<double> sample;
    sample.push_back(x);
    sample.push_back(y);
    gmm.addSample(sample);
}

float ofApp::testSample(double x, double y) {
    vector<double> sample;
    sample.push_back(x);
    sample.push_back(y);
    return gmm.getProbability(sample);
}

void ofApp::update() {
    
}

void ofApp::draw() {
    ofBackground(0);
    
    // get grid of 100 x 100 probabilities
    for (int i=0; i<100; i++) {
        for (int j=0; j<100; j++) {
            double x = ofMap(i, 0, 100, 0, ofGetWidth());
            double y = ofMap(j, 0, 100, 0, ofGetHeight());
            float probability = testSample(x, y);

            ofFill();
            ofSetColor(ofClamp(ofMap(probability, 0, 0.00001, 0, 255), 0, 255));
            ofRect(x, y, ofGetWidth()/100, ofGetHeight()/100);
        }
    }
    
    // draw random sample on grid
    ofSetColor(0, 0, 255);
    ofCircle(randSample[0], randSample[1], 10);
    
    // get mixture parameters
    for (int i=0; i<numGaussians; i++) {
        vector<double> mean = gmm.getMean(i);
        vector<double> variance = gmm.getVariance(i);
        vector<double> std = gmm.getStandardDeviation(i);
        double prior = gmm.getPrior(i);
        string msg = "Gaussian #"+ofToString(i)+" : prior ("+ofToString(prior)+"), mean ("+ofToString(mean[0])+", "+ofToString(mean[1])+"), standard deviation ("+ofToString(std[0])+", "+ofToString(std[1])+")";
        ofSetColor(0, 255, 0);
        ofDrawBitmapString(msg, 15, 20 + 20*i);
    }
    
    // get sample from mouse
    double x = (double) ofGetMouseX();
    double y = (double) ofGetMouseY();
    float probability = testSample(x, y);
    
    string msg = "P("+ofToString(x) + ", " + ofToString(y) + ") = " + ofToString(probability);
    ofDrawBitmapString(msg, x, y);

    // message about controls
    ofSetColor(255, 0, 0);
    ofDrawBitmapString("Press '1' for example of training GMM from data\nPress '2' for example of setting GMM explicitly\nPress spacebar to sample random point from GMM", 20, ofGetHeight()-50);
}

void ofApp::keyPressed(int key) {
    if      (key=='1')  trainGMMFromData();
    else if (key=='2')  setGMMExplicitly();
    else if (key==' ')  randSample = gmm.getRandomSample();
}