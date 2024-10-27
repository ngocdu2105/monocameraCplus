#include<string.h>

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include<opencv2/opencv.hpp>
#include "include/Calibration.h"
#include "include/ProcessVector.h"
#include"include/LoadIMG.h"

Calibration::Calibration(const std::string& pathNoObject): pathNoObject(pathNoObject)
{
    Timer t;
    cv::Mat gray_image = cv::imread(pathNoObject,0);
    bool found = cv::findChessboardCorners(gray_image, board_sz, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);

    if(found)
    {
        cv::cornerSubPix(gray_image, corners, cv::Size(13, 9), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.1));
        cv::drawChessboardCorners(gray_image, board_sz, corners, found);
    }
    double dis = distancePointToLinePoint(corners[13],corners[0],corners[1]);
    std::vector<float> diss;
    for(int i=1;i< corners.size();i++)
    {
        double dis_ = cv::norm(corners[i-1]-corners[i]);
        if(dis_ < dis+10) {diss.push_back(static_cast<float>(dis_));}
    }
    disTB = std::accumulate(diss.begin(),diss.end(),0.0) / diss.size();
    
    Orin = corners[0];
    Orin_x = corners[1];
    Orin_y = corners[13];

    std::cerr<<"Load success Calibration ";
    t.stop();  

}

const double& Calibration::getDisTB() const{
    return disTB;
}
void Calibration::calPoint(const cv::Point2f& p,cv::Mat& img)
{
    // auto pt = corners[20];

    auto pt = p;
    dis_Ox = distancePointToLinePoint(pt,Orin,Orin_x)/disTB;
    dis_Oy = distancePointToLinePoint(pt,Orin,Orin_y)/disTB;
    cv::circle(img, pt, 2, cv::Scalar(0,0,255), 2);
    cv::circle(img, Orin, 2, cv::Scalar(0,0,255), 2);
    cv::circle(img, Orin_x, 2, cv::Scalar(0,0,255), 2);
    cv::circle(img, Orin_y, 2, cv::Scalar(0,0,255), 2);    
    results(img);
    std::cerr<<"Successful conversion of 2D calibration chessboard prediction points: P("<<dis_Ox<<", "<<dis_Oy<<") .\n";

}
void Calibration::results(cv::Mat& img)
{
    cv::putText(img,\
    "P("+ std::to_string(dis_Ox)+", "+std::to_string(dis_Oy)+")",\
    cv::Point(50,50),\
    cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0, 0, 255),2);
}

