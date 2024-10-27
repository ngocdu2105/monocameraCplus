#ifndef LOADIMG_H
#define LOADIMG_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include<string>

class LoadIMG
{
public:
    explicit LoadIMG(const std::string& pathForder);
    const std::vector<cv::Mat>& getImgs() const;
private:
    std::vector<cv::Mat> images;
    size_t coutImg = 0;
    bool isImageFile(const std::string& filename);
    void load(const std::string& pathForder);
};


class Timer
{
public:
    Timer();
    void stop();
private:
    std::chrono::high_resolution_clock::time_point start_time_point;
    std::chrono::high_resolution_clock::time_point end_time_point;
};

#endif