#pragma once

#include "ofMain.h"
#include "ofxGMM.h"


class ofApp : public ofBaseApp
{
public:
	void setup();
	void update();
	void draw();
	void keyPressed(int key);

    void addSample(double x, double y);
    float testSample(double x, double y);
    
	ofxGMM gmm;
};
