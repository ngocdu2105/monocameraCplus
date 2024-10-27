#ifndef PROCESSVECTOR_H
#define PROCESSVECTOR_H

#include<vector>
#include<algorithm>
#include<numeric>
#include<iostream>
#include <iterator>
#include<opencv2/opencv.hpp>
#include "include/ProcessVector.tpp"
#include<onnxruntime_cxx_api.h>


class PointCollection;

struct VectorPointCollection
{
    std::vector<PointCollection> v;
    const PointCollection& operator[](size_t t) const;
    PointCollection& operator[](size_t);
    size_t indexPointSumVectorMinToNNBP2(const PointCollection& NNBP2);
};

class PointCollection
{
private:
    /* data */
    std::vector<cv::Point> points;
public:
    using value_type = cv::Point; 
    PointCollection(/* args */)= default;
    PointCollection(const std::vector<cv::Point>& p1);
    void addPoint(const cv::Point& p);
    PointCollection& operator+=(const cv::Point&p);
    PointCollection operator-(const cv::Point&p) const;
    PointCollection operator+(const cv::Point&p) const;
    const cv::Point& operator[](size_t index) const;
    cv::Point& operator[](size_t index);
    auto begin() { return points.begin(); }
    auto end() { return points.end(); }
    auto begin() const { return points.begin(); }
    auto end() const { return points.end(); }
    std::vector<std::vector<double>> operator|(const PointCollection& p2) const;
    VectorPointCollection operator+(const PointCollection& p2) const;
    const std::vector<cv::Point>& getPoints() const;
    void dump() const;
    size_t size();
    ~PointCollection();
};




struct PointNB
{
    // std::vector<cv::Point> NBP1;
    // std::vector<cv::Point> NBP2;
    // std::vector<cv::Point> NNBP1;
    // std::vector<cv::Point> NNBP2;

    PointCollection NBP1;
    PointCollection NBP2;
    PointCollection NNBP1;
    PointCollection NNBP2; 
    void dumpNBP1() const;
    void dumpNBP2() const;
    void dumpNNBP1() const;
    void dumpNNBP2() const;
    PointCollection getPointNBP1(size_t begin, size_t end);
};



PointNB createStructPointNN(const std::vector<cv::Point> &NBP1, const std::vector<cv::Point> &NBP2,const std::vector<cv::Point> &NNBP1,const std::vector<cv::Point> &NNBP2);
VectorPointCollection createStructVectorPointCollection(const std::vector<PointCollection>& vp);
std::vector<float> matToVectorCols(const cv::Mat* channels);
std::vector<float> matToVector(const cv::Mat& img);

std::vector<std::vector<float>> vec1dTo2d(const std::vector<float> & vec, int rows);

std::vector<cv::Mat> vectoMatNchannels(const std::vector<float>& vec, int w, int h, int channels);
PointNB IndexP1P2MinPointP1ToP2Condition(std::vector<cv::Point>& p1,std::vector<cv::Point>& p2, double condition = 50);
PointNB IndexP1P2MinPointP1ToP2Condition(std::vector<cv::Point>& p1,std::vector<cv::Point>& p2, double condition);
void pointArrTovector2d(const float* f, size_t fat, std::vector<std::vector<float>>& vec2d, int rows=6);
void pointArrTovector2d(const float* f, size_t fat, std::vector<std::vector<float>>& vec2d, int rows);

template<typename T> void print_Vector(const std::vector<T> &v);

// template<typename T>
// void print_Vector(const std::vector<T> &v)
// {
//     std::cout<<"\n[";
//     std::copy(v.begin(),v.end(),std::ostream_iterator<T>(std::cout,","));
//     std::cout<<"]\n";
// }

template<typename T> void print_Vector2(const std::vector<std::vector<T>> &v);

template<typename T> std::vector<size_t> sort_index(const std::vector<T> &v);
std::vector<int> GenerateRandomVector(int NumberCount,int minimum, int maximum);

double SumAllRowMin(const std::vector<std::vector<double>> &disP1P2);


std::vector<std::vector<double>> Distance_VectorPointP1P2(const std::vector<cv::Point> &p1,const std::vector<cv::Point> &p2);

std::vector<double> IndexDistanceVectorP1minP2(const std::vector<std::vector<double>> &disP1P2);
cv::Point PointMinMaxDisP1P2(const std::vector<std::vector<double>> &disP1P2);


std::vector<cv::Point> MoveVectorToPoint(const std::vector<cv::Point> &P1,const cv::Point &P);

std::vector<float> generate_scale(const cv::Mat& im, const std::vector<int>& target_size_ = {224,224});
std::vector<float> generate_scale(const cv::Mat& im, const std::vector<int>& target_size_);
void getSumMaskFromIndex(const std::vector<cv::Mat>&mat, const std::vector<size_t>& index, cv::Mat& sumMat);

double distancePointToLinePoint(const cv::Point& P, const cv::Point& A, const cv::Point& B);

// void GetOutputOnnx(std::vector<Ort::Value>& results, cv::Mat& matTop, cv::Mat& matBelow ,int w, int h, float thresh = 0.75  );
void GetOutputOnnx(std::vector<Ort::Value>& results, cv::Mat& matTop, cv::Mat& matBelow, int w, int h , float thresh);

#endif