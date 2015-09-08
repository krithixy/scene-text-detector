/* textdetection.cpp
 *
 * A demo program of the Extremal Region Filter algorithm described in
 * Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
 *
 * Created on: Sep 23, 2013
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include  "opencv2/text.hpp"
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include  <vector>
#include  <iostream>
#include  <iomanip>

using namespace std;
using namespace cv;
using namespace cv::text;

void show_help_and_exit(const char *cmd);
Rect groups_draw(Mat &src, vector<Rect> &groups);
void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions);
void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
vector<Rect> computeGroupsWithMinArea(Mat &src,vector<Mat> &channels,float minArea);
void doOCR(Mat &image,vector<Mat> channels,vector<vector<ERStat> > regions,vector< vector<Vec2i> > nm_region_groups,vector<Rect> nm_boxes);
class Regions{
public:
    vector<vector<ERStat> > regions;
    vector<Rect> groups_boxes;
    vector< vector<Vec2i> > region_groups;
    
};
Regions computeRegionGroups(Mat &src,vector<Mat> &channels,float minArea);
int main(int argc, const char * argv[])
{
    string path = "/Users/prjha/Documents/git/pranavjha/_scene-text-detector/samples/textpage300.png";
    cout << endl << argv[0] << endl << endl;
    cout << "Demo program of the Extremal Region Filter algorithm described in " << endl;
    cout << "Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012" << endl << endl;
    
    namedWindow("grouping",WINDOW_NORMAL);
    Mat src = imread(path);
    
    // Extract channels to be processed individually
    vector<Mat> channels;
    computeNMChannels(src, channels);
    
    int cn = (int)channels.size();
    int minArea= 0.0080f;
    // Append negative channels to detect ER- (bright regions over dark background)
    for (int c = 0; c < cn-1; c++)
        channels.push_back(255-channels[c]);
    vector<Rect> groups_boxes;
    Regions region=computeRegionGroups(src,channels,minArea);
    groups_boxes=region.groups_boxes;
    if(groups_boxes.size()==0){
        groups_boxes=computeGroupsWithMinArea(src,channels,minArea);
    }
    cout<<"received boxes"<<groups_boxes.size();
    // draw groups
   	Rect group_box= groups_draw(src, groups_boxes);
    vector<Rect> boxes;
    boxes.push_back(group_box);
    doOCR(src,channels,region.regions,region.region_groups,groups_boxes);
    imshow("grouping",src);
    //crop image to only the box containing the text
    imwrite("/Users/prjha/Documents/git/pranavjha/_scene-text-detector/samples/_2.jpg",src);
    cout << "Done!" << endl << endl;
    
    
    // memory clean-up
    if (!groups_boxes.empty())
    {
        groups_boxes.clear();
    }
}
void doOCR(Mat &image,vector<Mat> channels,vector<vector<ERStat> > regions,vector< vector<Vec2i> > nm_region_groups,vector<Rect> nm_boxes){
    /*Text Recognition (OCR)*/
    
    double t_r = (double)getTickCount();
    Ptr<OCRTesseract> ocr = OCRTesseract::create();
    cout << "TIME_OCR_INITIALIZATION = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;
    string output;
    
    Mat out_img;
    Mat out_img_detection;
    Mat out_img_segmentation = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
    image.copyTo(out_img);
    image.copyTo(out_img_detection);
    float scale_img  = 600.f/image.rows;
    float scale_font = (float)(2-scale_img)/1.4f;
    vector<string> words_detection;
    
    t_r = (double)getTickCount();
    
    for (int i=0; i<(int)nm_boxes.size(); i++)
    {
        cout<<nm_boxes[i].tl()<<"\t"<<nm_boxes[i].br();
        rectangle(out_img_detection, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(0,255,255), 3);
        
        Mat group_img = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        er_draw(channels, regions, nm_region_groups[i], group_img);
        Mat group_segmentation;
        group_img.copyTo(group_segmentation);
        //image(nm_boxes[i]).copyTo(group_img);
        group_img(nm_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));
        
        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;
        ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);
        
        output.erase(remove(output.begin(), output.end(), '\n'), output.end());
        cout << "OCR output = \"" << output << "\" lenght = " << output.size() << endl;
        for (int j=0; j<(int)boxes.size(); j++)
        {
            boxes[j].x += nm_boxes[i].x-15;
            boxes[j].y += nm_boxes[i].y-15;
            
            cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
            words_detection.push_back(words[j]);
            rectangle(out_img, boxes[j].tl(), boxes[j].br(), Scalar(255,0,255),3);
            Size word_size = getTextSize(words[j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3*scale_font), NULL);
            rectangle(out_img, boxes[j].tl()-Point(3,word_size.height+3), boxes[j].tl()+Point(word_size.width,0), Scalar(255,0,255),-1);
            //putText(out_img, words[j], boxes[j].tl()-Point(1,1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255,255,255),(int)(3*scale_font));
            out_img_segmentation = out_img_segmentation | group_segmentation;
        }
        
    }
}
vector<Rect> computeGroupsWithMinArea(Mat &src,vector<Mat> &channels,float minArea){
    Regions region= computeRegionGroups(src,channels,minArea);
    vector<Rect> groups_boxes=region.groups_boxes;
    if(groups_boxes.size()==0 && minArea-0.00075 > 0){
        return computeGroupsWithMinArea(src,channels,minArea-0.00075);
    }
    else if(groups_boxes.size()==0 && minArea-0.00015 >0) {
        return computeGroupsWithMinArea(src,channels,minArea-0.00075);
    }
    else{
        return groups_boxes;
    }
}
Regions computeRegionGroups(Mat &src,vector<Mat> &channels,float minArea){
    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("/Users/prjha/Documents/git/pranavjha/_scene-text-detector/resources/trained_classifierNM1.xml"),100,minArea,0.13f,0.4f,true,0.1f);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("/Users/prjha/Documents/git/pranavjha/_scene-text-detector/resources/trained_classifierNM2.xml"),0.5);
    cout<<minArea<<"\n";
    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    cout << "Extracting Class Specific Extremal Regions from " << (int)channels.size() << " channels ..." << endl;
    cout << "    (...) this may take a while (...)" << endl << endl;
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }
    
    // Detect character groups
    cout << "Grouping extracted ERs ... "<<endl;
    vector< vector<Vec2i> > region_groups;
    vector<Rect> groups_boxes;
    erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ);
    Regions region;
    region.regions=regions;
    region.groups_boxes=groups_boxes;
    region.region_groups=region_groups;
    
    return region;
}

// helper functions

void show_help_and_exit(const char *cmd)
{
    cout << "    Usage: " << cmd << " <input_image> " << endl;
    cout << "    Default classifier files (trained_classifierNM*.xml) must be in current directory" << endl << endl;
    exit(-1);
}
Rect groups_draw(Mat &src, vector<Rect> &groups)
{
    Point tl,br;
    for ( int i=0; i<(int )groups.size(); i++)
    {
        if(i==0){
            tl=groups.at(i).tl();
            br=groups.at(i).br();
        }
        if(groups.at(i).tl().x<tl.x){
            tl.x=groups.at(i).tl().x;
        }
        if(groups.at(i).br().x>br.x){
            br.x=groups.at(i).br().x;
        }
        if(groups.at(i).tl().y<tl.y){
            tl.y=groups.at(i).tl().y;
        }
        if(groups.at(i).br().y>br.y){
            br.y=groups.at(i).br().y;
        }
    }
    tl.x= tl.x >=0 ? tl.x : tl.x;
    tl.y= tl.y >=0 ? tl.y : tl.y;
    br.x= br.x < src.size().width ? br.x: br.x;
    br.y = br.y < src.size().height? br.y:br.y;
    Rect rect(tl,br);
    return rect;
  
}

void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions)
{
    for (int c=0; c<(int)channels.size(); c++)
    {
        Mat dst = Mat::zeros(channels[0].rows+2,channels[0].cols+2,CV_8UC1);
        for (int r=0; r<(int)regions[c].size(); r++)
        {
            ERStat er = regions[c][r];
            if (er.parent != NULL) // deprecate the root region
            {
                int newMaskVal = 255;
                int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
                floodFill(channels[c],dst,Point(er.pixel%channels[c].cols,er.pixel/channels[c].cols),
                          Scalar(255),0,Scalar(er.level),Scalar(0),flags);
            }
        }
        char buff[10]; char *buff_ptr = buff;
        sprintf(buff, "channel %d", c);
        imshow(buff_ptr, dst);
    }
    waitKey(-1);
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
