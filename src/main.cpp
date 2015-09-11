#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> //For SiftFeatureDetector
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor

int main(void)
{
  printf("test\n");
    cv::Mat image;
    image = cv::imread("ar1.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    //轉灰階
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, CV_RGB2GRAY);  

    //取adaptive threshold來二值化
    cv::Mat thresholdImg;
    //輸入影像grayscale
    //輸出影像thresholdImg
    //使用CV_THRESH_BINARY和CV_THRESH_BINARY_INV 的最大值255
    //閥值算法使用CV_ADAPTIVE_THRESH_MEAN_C或CV_ADAPTIVE_THRESH_GAUSSIAN_C  
    //閥值類型使用CV_THRESH_BINARY或CV_THRESH_BINARY_INV
    //用來計算閥值的鄰近大小,可為3,5,7...
    cv::adaptiveThreshold(grayscale, thresholdImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 7, 7);  
    
    //檢測輪廓
    std::vector< std::vector<cv::Point> > contours;
    std::vector< std::vector<cv::Point> > allContours;  
    int minContoursPointAllowed = 10;
    //輪廓搜索模式:
    //    CV_RETR_EXTERNAL只檢測外輪廓 
    //    CV_RETR_LIST檢測的輪廓不建立等級關系
    //    CV_RETR_CCOMP兩個等級的輪廓,外輪廓跟內輪廓
    //    CV_RETR_TREE樹等級的輪廓
    //輪廓近似方法:
    //    CV_CHAIN_APPROX_NONE所有輪廓點距離不超過1,即max（abs（x1-x2），abs（y2-y1））==1 
    //    CV_CHAIN_APPROX_SIMPLE壓縮水平,垂直跟對角方向的輪廓資訊, 只取端點, 例如矩形輪廓只需4點資訊
    //    CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法 
    cv::findContours(thresholdImg, allContours, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1);  
    contours.clear(); 
    for (size_t i = 0; i < allContours.size(); i++)  
    {  
        int size = allContours[i].size();  
        if (size > minContoursPointAllowed)  
        {  
            contours.push_back(allContours[i]);  
        }  
    }   


    for (size_t i = 0; i < contours.size(); i++)  
    {  
        //cv::Point( 200, 200 )
        //cv::Point p = contours[i];
        //cv::circle(image,contours[i],10,CV_RGB(255,0,0),3,CV_AA,0);
        //cv::circle( image, cv::Point( 200, 200 ), 32.0, cv::Scalar( 0, 0, 255 ), 1, 8 );
        //cv::approxPolyDP(contours[i], approxCurve, eps, true);  
    }


    const int nOffset=20; 
  //輸入4頂點
  //輸出4頂點
  // 設定變換[之前]與[之後]的坐標 (左上,左下,右下,右上)
  cv::Point2f pts1[] = {cv::Point2f(0,0),cv::Point2f(0,grayscale.rows),cv::Point2f(grayscale.cols,grayscale.rows),cv::Point2f(grayscale.cols,0)};

  cv::Point2f pts2[] = {cv::Point2f(0,0),cv::Point2f(0+nOffset,grayscale.rows),cv::Point2f(grayscale.cols-nOffset,grayscale.rows),cv::Point2f(grayscale.cols,0)};
  // 透視變換行列計算
  cv::Mat perspective_matrix = cv::getPerspectiveTransform(pts1, pts2);
  cv::Mat dst_img;
  // 變換
  cv::warpPerspective(grayscale, dst_img, perspective_matrix, grayscale.size(), cv::INTER_LINEAR);
  //
  //cv::threshold(grey, grey, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  
  //畫原始圖型
  cv::line(image, pts1[0], pts1[1], cv::Scalar(255,255,0), 2, CV_AA);
  cv::line(image, pts1[1], pts1[2], cv::Scalar(255,255,0), 2, CV_AA);
  cv::line(image, pts1[2], pts1[3], cv::Scalar(255,255,0), 2, CV_AA);
  cv::line(image, pts1[3], pts1[0], cv::Scalar(255,255,0), 2, CV_AA);
  cv::line(image, pts2[0], pts2[1], cv::Scalar(255,0,255), 2, CV_AA);
  cv::line(image, pts2[1], pts2[2], cv::Scalar(255,0,255), 2, CV_AA);
  cv::line(image, pts2[2], pts2[3], cv::Scalar(255,0,255), 2, CV_AA);
  cv::line(image, pts2[3], pts2[0], cv::Scalar(255,0,255), 2, CV_AA);
  cv::namedWindow ("srcimage",cv::WINDOW_AUTOSIZE);
  cv::imshow( "srcimage", image );  
  //
  cv::namedWindow ("image",cv::WINDOW_AUTOSIZE); 
  cv::imshow( "image", dst_img );  
  cv::waitKey(0);   
    //std::system("PAUSE");
}
