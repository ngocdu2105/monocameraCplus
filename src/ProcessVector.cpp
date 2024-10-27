

#include<vector>
#include<algorithm>
#include<numeric>
#include<iostream>
#include <iterator>
#include<opencv2/opencv.hpp>
#include "include/ProcessVector.h"
#include<onnxruntime_cxx_api.h>
#include<fstream>
#include <cassert>


PointCollection::PointCollection(const std::vector<cv::Point>& p1):points(p1){};
const std::vector<cv::Point>& PointCollection::getPoints() const{
    return points;
};
PointCollection PointCollection::operator-(const cv::Point&p) const
{
    std::vector<cv::Point> results;
    std::transform(points.begin(),points.end(),std::back_inserter(results),[&p](const cv::Point& t){return p-t;});
    return PointCollection(results);
}
PointCollection PointCollection::operator+(const cv::Point&p) const
{
    std::vector<cv::Point> results;
    std::transform(points.begin(),points.end(),std::back_inserter(results),[&p](const cv::Point& t){return t+p;});
    return PointCollection(results);
}

PointCollection& PointCollection::operator+=(const cv::Point&p)
{
    points.push_back(p);
}
void PointCollection::addPoint(const cv::Point& p)
{
    points.push_back(p);
}
void PointCollection::dump() const
{
    print_Vector(points);
}
cv::Point& PointCollection::operator[](size_t index)
{
    points.at(index);
};
const cv::Point& PointCollection::operator[](size_t index) const
{
    points.at(index);
};
std::vector<std::vector<double>> PointCollection::operator|(const PointCollection& p2) const
{
    // std::vector<std::vector<double>> result;
    
    return Distance_VectorPointP1P2(p2.getPoints(),points);

}
VectorPointCollection PointCollection::operator+(const PointCollection& p2) const
{
    std::vector<PointCollection> result;
    std::transform(p2.begin(),p2.end(),std::back_inserter(result),[this](const cv::Point& p)
    {
        std::vector<cv::Point> pmove;
        std::transform(points.begin(), points.end(),std::back_inserter(pmove),[&p](const cv::Point& pv){
            return p+pv;
        });
        return PointCollection(pmove);
    });


    return createStructVectorPointCollection(result);

}
PointCollection& VectorPointCollection::operator[](size_t t) {
    return v.at(t);
}
const PointCollection & VectorPointCollection::operator[](size_t t) const {
    return v.at(t);
}

size_t PointCollection::size()
{
    return points.size();
}

PointCollection::~PointCollection()
{
    points.clear();
}



void PointNB::dumpNBP1() const
{
    NBP1.dump();
};
void PointNB::dumpNBP2() const
{
    NBP2.dump();
}
void PointNB::dumpNNBP1() const
{
    NNBP1.dump();
}
void PointNB::dumpNNBP2() const
{
    NNBP2.dump();
}
PointCollection PointNB::getPointNBP1(size_t begin, size_t end)
{
    // assert(begin < NBP1.size() && end <= NBP1.size() && begin <= end && begin >= 0) ;
    std::vector<cv::Point> valueIndex(NBP1.begin()+begin,NBP1.begin()+end);
    return PointCollection(valueIndex);
}



size_t VectorPointCollection::indexPointSumVectorMinToNNBP2(const PointCollection& NNBP2)
{
    assert(v.empty()== false);
    std::vector<double> min;
    std::transform(v.begin(),v.end(),std::back_inserter(min), [&NNBP2](const PointCollection& nnbp1)
    {
        std::vector<std::vector<double>> dis = NNBP2 | nnbp1;
        return SumAllRowMin(dis);

    });

    return std::distance(min.begin(),std::min_element(min.begin(),min.end()));

}




PointNB createStructPointNN(const std::vector<cv::Point> &NBP1, const std::vector<cv::Point> &NBP2,const std::vector<cv::Point> &NNBP1,const std::vector<cv::Point> &NNBP2)
{
    assert(!NBP1.empty() && !NBP2.empty() && !NNBP1.empty() && !NNBP2.empty() && "createStructPointNN input some of them empty !!");
    PointNB p;
    p.NBP1= PointCollection(NBP1);
    p.NBP2=PointCollection(NBP2);
    p.NNBP1 = PointCollection(NNBP1);
    p.NNBP2 = PointCollection(NNBP2);
    return p;
}


VectorPointCollection createStructVectorPointCollection(const std::vector<PointCollection>& vp)
{
    assert(!vp.empty() && " funtion createStructVectorPointCollection must input not empty !!!");
    VectorPointCollection vpt;
    vpt.v = vp;
    return vpt;
};




std::vector<float> matToVectorCols(const cv::Mat* channels) {
    size_t totalPixels= channels[0].total();
    std::vector<float> imgVector(3 * totalPixels);


    for (size_t i = 0; i < totalPixels; ++i) {
        imgVector[i * 3 + 0] = channels[0].at<float>(i); 
        imgVector[i * 3 + 1] = channels[1].at<float>(i); 
        imgVector[i * 3 + 2] = channels[2].at<float>(i); 
    }

    return imgVector;
}

std::vector<float> matToVector(const cv::Mat& img) {
    assert(!img.empty() && "funtion matToVector must be not empty img!!!");
    size_t totalPixels= img.total();
    std::vector<float> imgVector(3 * totalPixels);

    std::memcpy(imgVector.data(), img.data, imgVector.size()*sizeof(float));

    return imgVector;
}

std::vector<std::vector<float>> vec1dTo2d(const std::vector<float> & vec, int rows)
{
    assert(!vec.empty() && "funtion vec1dTo2d must be not empty vector  !!!");
    size_t cols = vec.size()/rows;
    std::vector<std::vector<float>> vec2d(cols,std::vector<float> (rows));
    for(int i = 0;i <cols;i++)
    {
        std::copy(vec.begin()+i*rows,vec.begin()+i*rows+rows,vec2d[i].begin());
    }

    return vec2d;
}

std::vector<cv::Mat> vectoMatNchannels(const std::vector<float>& vec, int w, int h, int channels)
{
    assert(vec.size() == w*h*channels && "vector size not match channels!");
    cv::Mat img(w,h,CV_32FC(channels), const_cast<float*> (vec.data()));



    cv::Mat preprocessedImage;
    std::vector<cv::Mat> channel; 
    cv::Mat img_;
   
    std::vector<std::vector<float>> cv = vec1dTo2d(vec,w*h);

    // std::vector<std::vector<float>> cv(channels, std::vector<float>(w * h));
    // for(auto i = 0;i< channels;i++)
    // {
    //     std::copy(vec.begin()+w*h*i,vec.begin()+w*h*i+w*h,cv[i].begin());
    //     print_Vector(cv[0]);
    // }



    for(const auto& i: cv)
    {

        cv::Mat mat(w,h,CV_32F,(float*)i.data()) ;
      
        channel.push_back(mat.clone());
    }


    return channel;
}


void pointArrTovector2d(const float* f, size_t fat, std::vector<std::vector<float>>& vec2d, int rows)
{
    assert(f!=nullptr && "Invalid pointer (nullptr)");
    std::vector<float> fv(f,f+fat);
    vec2d = vec1dTo2d(fv,rows);

}


std::vector<int> GenerateRandomVector(int NumberCount,int minimum, int maximum) {
    std::random_device rd; 
    std::mt19937 gen(rd()); // these can be global and/or static, depending on how you use random elsewhere

    std::vector<int> values(NumberCount); 
    std::uniform_int_distribution<int> dis(minimum, maximum);
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}


// Khoang cach tung diem p1 den tung diem p2
std::vector<std::vector<double>> Distance_VectorPointP1P2(const std::vector<cv::Point> &p1,const std::vector<cv::Point> &p2)
{
    assert(!p1.empty() && !p2.empty() && "Funtion must be not empty input!!");

    std::vector<std::vector<double>> disP1P2;
    std::transform(p1.begin(),p1.end(),std::back_inserter(disP1P2),[&p2](const cv::Point &pP1)
    {
        std::vector<double> dispP1P2(p2.size());
        std::transform(p2.begin(),p2.end(),dispP1P2.begin(),[&pP1](const cv::Point &pP2){return cv::norm(pP1-pP2);});
        return dispP1P2;

    });
    return disP1P2;
}

//Tim tung diem p1 den p2 gan nhat neu khoang cach nho hon condition
PointNB IndexP1P2MinPointP1ToP2Condition(std::vector<cv::Point>& p1,std::vector<cv::Point>& p2, double condition)
{
    assert(!p1.empty() && !p2.empty() && "Funtion must be not empty input !!!");
    std::vector<std::vector<double>> disP1P2 = Distance_VectorPointP1P2(p1,p2);
    std::vector<size_t> PointP1P2;
    std::transform(disP1P2.begin(),disP1P2.end(), std::back_inserter(PointP1P2),[&condition](const std::vector<double>& vp2 )
    {
        double min_ = *std::min_element(vp2.begin(),vp2.end());
        return min_ < condition ? std::distance(vp2.begin(),std::min_element(vp2.begin(),vp2.end())): 0;
    });

    std::vector<size_t> index(PointP1P2.size());
    std::iota(index.begin(),index.end(),0);
    std::vector<cv::Point> NBP1,NBP2, NNBP1, NNBP2;
    for(auto & i: index)
    {
        if(PointP1P2[i] > 0)
        {
            NBP1.push_back(p1[i]);
            NBP2.push_back(p2[PointP1P2[i]]);

        }
        else
        {
            NNBP1.push_back(p1[i]);
        }
    }
    p2.erase(std::remove_if(p2.begin(),p2.end(),[&](const cv::Point& nnp2)
    {
        return std::find(NBP2.begin(),NBP2.end(), nnp2) != NBP2.end();
    }), p2.end());



    PointNB p = createStructPointNN(NBP1,NBP2,NNBP1,p2);



    return p;
}

double SumAllRowMin(const std::vector<std::vector<double>> &disP1P2)
{
    assert(!disP1P2.empty() && "input funtion SumAllRowMin empty !!!");

    std::vector<double> indexValueMinDisP1ToP2;
    std::transform(disP1P2.begin(),disP1P2.end(),std::back_inserter(indexValueMinDisP1ToP2),[](const std::vector<double> &disp1p2){
        return *std::min_element(disp1p2.begin(),disp1p2.end());

    });
    
    return std::accumulate(indexValueMinDisP1ToP2.begin(), indexValueMinDisP1ToP2.end(), 0.0);
}



std::vector<double> IndexDistanceVectorP1minP2(const std::vector<std::vector<double>> &disP1P2)
{
    assert(!disP1P2.empty() && "Input funtion IndexDistanceVectorP1minP2 empty !!!");

    std::vector<double> indexDisP1ToP2;
    std::transform(disP1P2.begin(),disP1P2.end(),std::back_inserter(indexDisP1ToP2),[](const std::vector<double> &disp1p2){

        return  std::accumulate(disp1p2.begin(), disp1p2.end(), 0.0);

    });

    return indexDisP1ToP2;
}

cv::Point PointMinMaxDisP1P2(const std::vector<std::vector<double>> &disP1P2)
{
    assert(!disP1P2.empty() && "input funtion PointMinMaxDisP1P2 empty !!!");
    std::vector<double> minP2;
    std::transform(disP1P2.begin(),disP1P2.end(),std::back_inserter(minP2),[](const std::vector<double> &disp1p2){
        // return std::distance(disp1p2.begin(),std::min_element(disp1p2.begin(),disp1p2.end()));
        // return *std::min_element(disp1p2.begin(),disp1p2.end());
        double min_ = *std::min_element(disp1p2.begin(),disp1p2.end());
        if(std::isnan(min_))
        {
            min_ = 0;
        }
        return min_ ;

    });


    size_t IndexMinpP1 =  std::distance(minP2.begin(),std::max_element(minP2.begin(),minP2.end()));

    std::vector<double> minpP1P2 = disP1P2.at(IndexMinpP1);
    int IndexMinpP2 = std::distance(minpP1P2.begin(),std::min_element(minpP1P2.begin(),minpP1P2.end()));


    return cv::Point(IndexMinpP1,IndexMinpP2);
}



std::vector<cv::Point> MoveVectorToPoint(const std::vector<cv::Point> &P1,const cv::Point &P)
{
    assert(!P1.empty() && "input funtion is not empty !!!");
    
    std::vector<cv::Point> P1move;
    std::transform(P1.begin(),P1.end(),std::back_inserter(P1move),[&P](const cv::Point pP1){return pP1+P;});
    return P1move;
}


std::vector<float> generate_scale(const cv::Mat& im, const std::vector<int>& target_size_) {
    assert(!im.empty() && "input funtion is not empty !!!");
    cv::Size origin_shape = im.size();
    float im_scale_x, im_scale_y;
    bool keep_ratio_ = true;
    if (keep_ratio_) {
        int im_size_min = std::min(origin_shape.width, origin_shape.height);
        int im_size_max = std::max(origin_shape.width, origin_shape.height);
        int target_size_min = *std::min_element(target_size_.begin(), target_size_.end());
        int target_size_max = *std::max_element(target_size_.begin(), target_size_.end());

        float im_scale = static_cast<float>(target_size_min) / static_cast<float>(im_size_min);
        if (std::round(static_cast<float>(im_scale * im_size_max)) > target_size_max) {
            im_scale = static_cast<float>(target_size_max) / static_cast<float>(im_size_max);
        }
        im_scale_x = im_scale;
        im_scale_y = im_scale;
    } else {
        im_scale_y = static_cast<float>(target_size_[0]) / static_cast<float>(origin_shape.height);
        im_scale_x = static_cast<float>(target_size_[1]) / static_cast<float>(origin_shape.width);
    }
    return {im_scale_y, im_scale_x};
}

void getSumMaskFromIndex(const std::vector<cv::Mat>&mat, const std::vector<size_t>& index, cv::Mat& sumMat)
{
    assert(!mat.empty() && "empty vector<cv::Mat>&mat is not Empty!!! ");

    cv::Mat sum= cv::Mat::zeros(mat[0].size(),mat[0].type());
    
    for(size_t in:index)
    {
        cv::add(sum,mat[in],sum);
    }
    sumMat = (sum > 0);
}
double distancePointToLinePoint(const cv::Point& P, const cv::Point& A, const cv::Point& B) {
    // Khoang cach P den AB
    double numerator = std::abs((B.y - A.y) * P.x - (B.x - A.x) * P.y + B.x * A.y - B.y * A.x);
    double denominator = std::sqrt(std::pow(B.y - A.y, 2) + std::pow(B.x - A.x, 2));
    return numerator / denominator;
}


void GetOutputOnnx(std::vector<Ort::Value>& results, cv::Mat& matTop, cv::Mat& matBelow, int w, int h, float thresh)
{
    // 
    assert(!results.empty() && "Funtion is not empty input !!!");
    const float* result1 = results[0].GetTensorData<float>();
    const float*result2 = results[1].GetTensorData<float>();
    const float*result3 = results[2].GetTensorData<float>();
    Ort::TensorTypeAndShapeInfo tensor_info1 = results[0].GetTensorTypeAndShapeInfo();
    Ort::TensorTypeAndShapeInfo tensor_info2 = results[1].GetTensorTypeAndShapeInfo();
    Ort::TensorTypeAndShapeInfo tensor_info3 = results[2].GetTensorTypeAndShapeInfo();
    size_t Narr = std::max(tensor_info1.GetElementCount(),static_cast<size_t>(1));
    size_t NFat2 = tensor_info2.GetElementCount();
    size_t NFat3 = tensor_info3.GetElementCount();
    std::vector<std::vector<float>> LabelBox;
    pointArrTovector2d(result2, NFat2,LabelBox,6); // 
    std::vector<int64_t> shapeOut3 = tensor_info3.GetShape();
    std::vector<int64_t> shapeOut2 = tensor_info2.GetShape();
    std::vector<int64_t> shapeOut1 = tensor_info1.GetShape();
    std::vector<size_t> indexTop;
    std::vector<size_t> indexBelow;
    size_t index = 0;
    std::for_each(LabelBox.begin(), LabelBox.end(),[&index,&indexTop,&indexBelow,thresh](const std::vector<float>& row) mutable
    {
        if(static_cast<int>(row[0])==1 && row[1]> thresh )
        {
            indexBelow.push_back(index);
        }
        if(static_cast<int> (row[0])==0 && row[1] > thresh)
        {
            indexTop.push_back(index);
        }
        index++;
    });
    // print_Vector(shapeOut3);
    // print_Vector(shapeOut2);
    // print_Vector(shapeOut1);

    std::vector<float> outmask(result3,result3+NFat3);

    // std::ofstream outline("vect.txt");
    // for(const auto& value :outmask)
    // {
    //     outline<<value<<std::endl;
    // }
    // outline.close();
    std::vector<cv::Mat> MatMask = vectoMatNchannels(outmask,shapeOut3[1],shapeOut3[2],(int)index);

    getSumMaskFromIndex(MatMask,indexTop,matTop);
    getSumMaskFromIndex(MatMask,indexBelow,matBelow); 



}
