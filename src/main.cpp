#include<string.h>

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include<opencv2/opencv.hpp>

#include "include/yolov5.h"
#include "include/ProcessVector.h"
#include "include/RCNN.h"
#include "include/Calibration.h"
#include "include/LoadIMG.h"
#include <vector>

void GetPointTopCenterMove(const cv::Mat& matTop, const cv::Mat& matBelow,int w, int h, \
cv::Point& pointTopCenter, cv::Point& pointTopCenterMove, std::vector<cv::Point>& VectorMoveNBP1NNBP2)
{
    assert(!matTop.empty() && !matBelow.empty());
    int cx=0,cy = 0;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(matTop, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Moments m = cv::moments(contours[0],false);

    if (m.m00 > 5000) {
        cx = static_cast<int>(m.m10 / m.m00); // Tọa độ x của tâm maskTop
        cy = static_cast<int>(m.m01 / m.m00); // Tọa độ y của tâm maskTop
    
        // std::cout << "Tâm của maskTop: (" << cx << ", " << cy  <<")" << std::endl;

    }
    assert(cx !=0 && cy != 0);
    pointTopCenter = cv::Point(cx,cy);

    cv::Mat blurred_below,blurred_top;
    cv::GaussianBlur(matBelow, blurred_below, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(matTop, blurred_top, cv::Size(5, 5), 1.5);
    // Apply Canny edge detection
    double lowerThreshold = 50; // Lower threshold for edge linking
    double upperThreshold = 150; // Upper threshold for edge linking
    cv::Mat edges_Top,edges_Below;
    cv::Canny(blurred_top, edges_Top, lowerThreshold, upperThreshold);    
    cv::Canny(blurred_below, edges_Below, lowerThreshold, upperThreshold); 
    std::vector<cv::Point> point_t,point_bw;
    //Find the mask edge point
    cv::findNonZero(edges_Top,point_t);
    cv::findNonZero(edges_Below,point_bw);
    if(point_bw.empty())
    {
        pointTopCenterMove = cv::Point(cx,cy);
        return;
    }
    //Find the intersection of two edges of each face
    PointNB tk = IndexP1P2MinPointP1ToP2Condition(point_bw,point_t,5);
    
    cv::Mat imgzero = cv::Mat::zeros(w,h,CV_8UC1);

    for(auto &i:tk.NBP1.getPoints())
    {
        imgzero.at<uchar>(i) = 255;
    }


    //calculate distance and displacement vector
    auto disNBP1toNNBP2 = tk.NNBP1 | tk.NBP1; //distance
    cv::Point MaxminP1P2 = PointMinMaxDisP1P2(disNBP1toNNBP2);
    cv::circle(edges_Below, tk.NBP1[MaxminP1P2.x], 3,cv::Scalar(255));      
    cv::circle(edges_Below, tk.NNBP1[MaxminP1P2.y], 3,cv::Scalar(255)); 

    // get sample point in NBP1
    size_t numberSampleX2 = 20;

    PointCollection NBSampleP1 = tk.getPointNBP1(MaxminP1P2.x-numberSampleX2,MaxminP1P2.x+ numberSampleX2) ;

    // move point

    PointCollection VecMoveNBP1toNNBP2 = NBSampleP1 - tk.NNBP1[MaxminP1P2.y];  
    auto NNBP1Move = tk.NBP1 + VecMoveNBP1toNNBP2;

    // Check for duplicate edges at intersection and end edges 
    size_t indexMove = NNBP1Move.indexPointSumVectorMinToNNBP2(tk.NNBP1);

    for(auto& i:NBSampleP1.getPoints())
    {
        cv::circle(edges_Below, i, 1,cv::Scalar(255));        
    }

    pointTopCenterMove = VecMoveNBP1toNNBP2[indexMove]+pointTopCenter;
    VectorMoveNBP1NNBP2 = std::vector<cv::Point>{tk.NBP1[indexMove],tk.NNBP1[MaxminP1P2.y]};
    // cv::circle(edges_Top, pointTopCenterMove , 2,cv::Scalar(255));
    // cv::circle(edges_Top, cv::Point(cx,cy), 3,cv::Scalar(255));
    // VecMoveNBP1toNNBP2.dump();
    // cv::imshow("img2",matTop);
    // cv::imshow("img",edges_Top);

    // cv::imshow("img_",edges_Below);
    // cv::imshow("imf",imgzero);

    // cv::waitKey(0);
}

cv::Point ConvertPointSquareToOrinIMG(const cv::Mat& img, const cv::Rect& rec, const cv::Point& p)
{
    cv::Size sz = img.size();
    int max_ = std::max(sz.width,sz.height);
    cv::Point Pmove = cv::Point(max_,max_) - cv::Point(sz.width,sz.height);
    cv::Point P = Pmove/2;
    cv::Point orin = cv::Point(rec.x,rec.y);
    return orin-P+p;

}


int main()
{
    // Load path img, model
    std::string pathIMG = "./dataset";
    std::string pathModelYOLOv5 = "./model/model_yolov5.onnx";
    std::string pathModelRCNN = "./model/mask_rcnn_dataset_fix.onnx";
    std::string pathImgCalibration = "./CalibrationIMG/calibration.jpg";
    // model initialization
    LoadIMG imgload(pathIMG);
    Calibration calibration(pathImgCalibration);
    Yolov5 yolov5(pathModelYOLOv5,640);
    RCNN rcnn(pathModelRCNN);

    auto& imgs = imgload.getImgs();
    // Timer t;    
    
    for(auto& img:imgs)
    {
        try{

            cv::Rect boxDetection;
            cv::Mat CropObject = yolov5.get_frame(img,boxDetection);
            cv::Size size_zeros_img = CropObject.size();
            // cv::imshow("orin",CropObject);
            // cv::waitKey(0);
            rcnn.run(CropObject);
            cv::Mat MaskTop = rcnn.masktop();
            cv::Mat MaskBelow = rcnn.maskbelow();

            std::vector<cv::Point> pT, pBl;
            cv::findNonZero(MaskTop,pT);
            cv::findNonZero(MaskBelow,pBl);

            cv::Point pointTopCenter, pointTopCenterMove;
            std::vector<cv::Point> VectorMoveNBP1NNBP2;
            GetPointTopCenterMove(MaskTop, MaskBelow, CropObject.rows, CropObject.cols, pointTopCenter, pointTopCenterMove,VectorMoveNBP1NNBP2);
            cv::Mat img_clone = img.clone();
            cv::Point PointT = ConvertPointSquareToOrinIMG(img,boxDetection,pointTopCenterMove);
            // cv::circle(CropObject, VectorMoveNBP1NNBP2[0] , 2,cv::Scalar(255,0,0),-1);
            // cv::circle(CropObject, VectorMoveNBP1NNBP2[1] , 2,cv::Scalar(255,0,0),-1);
            cv::circle(CropObject, pointTopCenter , 2,cv::Scalar(255,0,0),-1);
            std::cout<<"Pixel coordinates of point P: "<<PointT<<std::endl;
            calibration.calPoint(PointT, img_clone);
            assert (CropObject.size().width > 0 || CropObject.size().height > 0 && "Size not true");

        //     cv::Mat zeros_img_top = cv::Mat::zeros(size_zeros_img,CV_8UC1);
        //     cv::Mat zeros_img_below = zeros_img_top.clone();

        //     for(const auto& pTp:pT)
        //     {
        //         zeros_img_top.at<uchar>(pTp)=255;
        //     }
        //     for(const auto& pBlp:pBl)
        //     {
        //         zeros_img_below.at<uchar>(pBlp) = 255;
        //     }
        // CropObject.setTo(cv::Scalar(0,255,0),zeros_img_top);
        // CropObject.setTo(cv::Scalar(255,255,0),zeros_img_below);
        // cv::imshow("img_top",MaskTop);
        // cv::imshow("img_below",MaskBelow);

        // cv::imshow("orin",img_clone);
        // cv::imshow("mask",CropObject);

        // cv::waitKey(0);


        }
        catch(const std::exception& e)
        {
            std::cerr << "Error  " << e.what() << std::endl;
            continue;
        }
         catch (...) {
            std::cerr << "Unknown exception occurred." << std::endl;
        }        

    }
    std::cerr<<"End !!!!"<<std::endl;
    return 0;

}
