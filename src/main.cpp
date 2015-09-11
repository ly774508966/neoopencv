#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> //For SiftFeatureDetector

int main(void)
{
  printf("test\n");
    cv::Mat imageL;
    cv::Mat imageR;
    imageL = cv::imread("I1_000000.png", 0);   // Read the file
    imageR = cv::imread("I2_000000.png", 0);   // Read the file
    
    cv::SiftFeatureDetector sift_detector;
    std::vector<cv::KeyPoint> sift_keypoints_L;
    std::vector<cv::KeyPoint> sift_keypoints_R;
    sift_detector.detect(imageL, sift_keypoints_L);
    sift_detector.detect(imageR, sift_keypoints_R);
    //cvCvtColor(imageL,out,CV_BGR2GRAY);
    //IplImage* out = cvCreateImage( cvGetSize(imageL), IPL_DEPTH_8U, 1 );
//detector.detect(out, keypoints);

    cv::Mat outputL;
    cv::drawKeypoints(imageL, sift_keypoints_L, outputL);
   cv::Mat outputR;
    cv::drawKeypoints(imageR, sift_keypoints_R, outputR);
  
    //cv::namedWindow ("imageL",cv::WINDOW_AUTOSIZE);
    //cv::imshow( "imageL", imageL );  
    //cv::namedWindow ("imageR",cv::WINDOW_AUTOSIZE);
    //cv::imshow( "imageR", imageR );    
    cv::namedWindow ("imageL sift",cv::WINDOW_AUTOSIZE);
    cv::imshow( "imageL sift", outputL );  
    cv::namedWindow ("imageR sift",cv::WINDOW_AUTOSIZE);
    cv::imshow( "imageR sift", outputR );  
    
    cv::waitKey(0);
    
    //std::system("PAUSE");
}
