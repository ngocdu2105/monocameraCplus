#include <dirent.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "include/LoadIMG.h"
#include <chrono>
#include <iomanip>

LoadIMG::LoadIMG(const std::string& pathForder)
{
    load(pathForder);
}
void LoadIMG::load(const std::string& pathForder)
{
    auto start = std::chrono::high_resolution_clock::now();
    DIR* dir = opendir(pathForder.c_str());
    if (!dir) {
        std::cerr << "Invalid directory: " << pathForder << std::endl;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (isImageFile(filename)) {
            std::string fullPath = pathForder + "/" + filename;
            cv::Mat img = cv::imread(fullPath);
            if (!img.empty()) {
                images.push_back(img);
                coutImg ++;
                std::cout << "Loaded image: " << fullPath << std::endl;
            } else {
                std::cout << "Failed to load image: " << fullPath << std::endl;
            }
        }
    }

    closedir(dir);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    double seconds = duration * 1e-6;
    std::cerr<<" Total: " << coutImg << " images, time take: " << std::fixed << std::setprecision(6) << seconds<<"s"<<std::endl;
    
}

const std::vector<cv::Mat>& LoadIMG::getImgs() const
{
    return images;
}

bool LoadIMG::isImageFile(const std::string& filename) {
    std::string extension = filename.substr(filename.find_last_of('.') + 1);
    return (extension == "jpg" || extension == "jpeg" || extension == "png" || extension == "bmp");
    }

Timer::Timer(): start_time_point(std::chrono::high_resolution_clock::now()){};
void Timer::stop()
{
    end_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_point-start_time_point).count();
    double seconds = duration * 1e-6;
    std::cerr<<"time take: " << std::fixed << std::setprecision(6) << seconds<<"s"<<std::endl;
}