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

	ofxGMM gmm;
};
