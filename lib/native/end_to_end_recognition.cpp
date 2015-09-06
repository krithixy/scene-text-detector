/*
 * textdetection.cpp
 *
 * A demo program of End-to-end Scene Text Detection and Recognition:
 * Shows the use of the Tesseract OCR API with the Extremal Region Filter algorithm described in:
 * Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
 *
 * Created on: Jul 31, 2014
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::text;

// function to find text using OCR
string findText(string path);


//Calculate edit distance netween two words
size_t edit_distance(const string& A, const string& B);
size_t min(size_t x, size_t y, size_t z);
bool   isRepetitive(const string& s);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);


//Perform text detection and recognition and evaluate results using edit distance
int main(int argc, char* argv[]) {
    string path = "/Users/prjha/Documents/git/pranavjha/scene-text-detector/samples/011.jpg";
    cout<< findText(path);
    // waitKey(0);
    return 0;
}


string findText(string path) {
    Mat image  = imread(path);

    cout << "image width:  " << image.cols << endl;
    cout << "image height: " << image.rows << endl;
    
    // Text Detection
    // Extract channels to be processed individually
    vector<Mat> channels;
    Mat grey;
    cvtColor(image,grey,COLOR_RGB2GRAY);
    
    
    // Notice here we are only using grey channel, see textdetection.cpp for example with more channels
    channels.push_back(grey);
    channels.push_back(255-grey);
    double t_d = (double)getTickCount();
    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("/Users/prjha/Documents/git/pranavjha/scene-text-detector/resources/trained_classifierNM1.xml"), 8,0.00015f,0.13f,0.2f,true,0.1f);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("/Users/prjha/Documents/git/pranavjha/scene-text-detector/resources/trained_classifierNM2.xml"), 0.5);
    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    for (int c=0; c<(int)channels.size(); c++) {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }
    cout << "TIME_REGION_DETECTION = " << ((double)getTickCount() - t_d)*1000/getTickFrequency() << endl;
    Mat out_img_decomposition= Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
    vector<Vec2i> tmp_group;
    for (int i=0; i<(int)regions.size(); i++) {
        for (int j=0; j<(int)regions[i].size();j++) {
            tmp_group.push_back(Vec2i(i,j));
        }
        Mat tmp= Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        er_draw(channels, regions, tmp_group, tmp);
        if (i > 0)
            tmp = tmp / 2;
        out_img_decomposition = out_img_decomposition | tmp;
        tmp_group.clear();
    }
    
    double t_g = (double)getTickCount();
    
    // Detect character groups
    vector< vector<Vec2i> > nm_region_groups;
    vector<Rect> nm_boxes;
    erGrouping(image, channels, regions, nm_region_groups, nm_boxes,ERGROUPING_ORIENTATION_HORIZ);
    cout << "TIME_GROUPING = " << ((double)getTickCount() - t_g)*1000/getTickFrequency() << endl;
    // Text Recognition (OCR)
    double t_r = (double)getTickCount();
    Ptr<OCRTesseract> ocr = OCRTesseract::create();
    cout << "TIME_OCR_INITIALIZATION = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;
    string output = "";
    Mat out_img;
    image.copyTo(out_img);
    float scale_img  = 600.f/image.rows;
    float scale_font = (float)(2-scale_img)/1.4f;
    vector<string> words_detection;
    
    t_r = (double)getTickCount();
    
    for (int i=0; i<(int)nm_boxes.size(); i++) {
        string thisOutput;
        
        Mat group_img = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        er_draw(channels, regions, nm_region_groups[i], group_img);
        Mat group_segmentation;
        group_img.copyTo(group_segmentation);
        group_img(nm_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));
        
        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;
        ocr->run(group_img, thisOutput, &boxes, &words, &confidences, OCR_LEVEL_WORD);
        
        thisOutput.erase(remove(thisOutput.begin(), thisOutput.end(), '\n'), thisOutput.end());
        cout << "OCR output = \"" << thisOutput << "\" length = " << thisOutput.size() << endl;
        output += thisOutput;
        for (int j=0; j<(int)boxes.size(); j++) {
            boxes[j].x += nm_boxes[i].x-15;
            boxes[j].y += nm_boxes[i].y-15;
            cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
            if ((words[j].size() < 2) || (confidences[j] < 51) ||
                ((words[j].size()==2) && (words[j][0] == words[j][1])) ||
                ((words[j].size()< 4) && (confidences[j] < 60)) ||
                isRepetitive(words[j]))
                continue;
            words_detection.push_back(words[j]);
            rectangle(out_img, boxes[j].tl(), boxes[j].br(), Scalar(255,0,255),3);
            Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3*scale_font), NULL);
            rectangle(out_img, boxes[j].tl()-Point(3,word_size.height+3), boxes[j].tl()+Point(word_size.width,0), Scalar(255,0,255),-1);
            putText(out_img, words[j], boxes[j].tl()-Point(1,1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255,255,255),(int)(3*scale_font));
        }
    }
    
    cout << "TIME_OCR = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;
    
    cout<< "-------------------------------------------------------------------------------"<<endl;
    // imshow("recognition", out_img);
    return output;
}




bool isRepetitive(const string& s) {
    int count = 0;
    for (int i=0; i<(int)s.size(); i++) {
        if ((s[i] == 'i') ||
            (s[i] == 'l') ||
            (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2) {
        return true;
    }
    return false;
}


void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation) {
    for (int r=0; r<(int)group.size(); r++) {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) {
            floodFill(
                channels[group[r][0]],
                segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                Scalar(255),
                0,
                Scalar(er.level),
                Scalar(0),
                (4 + (255 << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY)
            );
        }
    }
}