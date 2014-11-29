#include "ofApp.h"


void ofApp::setup() {
    gmm.setup(2, 3);
    
    
    // add 100 samples centered around (220, 150)
    // add 150 samples centered around (150, 650)
    // add 100 samples centered around (740, 300)
    
    
    for (int i=0; i<100; i++) {
        addSample(220 + ofRandom(-100, 100),
                  150 + ofRandom(-95, 95));
    }
    for (int i=0; i<150; i++) {
        addSample(450 + ofRandom(-180, 180),
                  650 + ofRandom(-120, 120));
    }
    for (int i=0; i<100; i++) {
        addSample(820 + ofRandom(-90, 90),
                  320 + ofRandom(-150, 150));
    }
    
    gmm.train();
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
            ofSetColor(ofMap(probability, 0, 0.00001, 0, 255));
            ofRect(x, y, ofGetWidth()/100, ofGetHeight()/100);
        }
    }
    
    
    // get sample from mouse
    double x = (double) ofGetMouseX();
    double y = (double) ofGetMouseY();
    float probability = testSample(x, y);
    
    string msg = "P("+ofToString(x) + ", " + ofToString(y) + ") = " + ofToString(probability);
    ofSetColor(0, 255, 0);
    ofDrawBitmapString(msg, 15, 20);
}

void ofApp::keyPressed(int key) {
    
}