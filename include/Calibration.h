#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// #include<include/ProcessVector.h>


class Calibration
{
private:
    std::vector<cv::Point2f> corners;
    std::string pathNoObject;
    cv::Size board_sz = cv::Size(13,9);
    cv::Mat img;
    cv::Point2f Orin,Orin_x,Orin_y,Ox,Oy;
    double disTB,dis_Ox,dis_Oy;

public:
    Calibration(const std::string& pathNoObject);
    const double& getDisTB() const;
    void calPoint(const cv::Point2f& pt,cv::Mat& img);
    void results( cv::Mat& img);
    // ~Calibration();
};