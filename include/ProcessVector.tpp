#include<vector>
#include<algorithm>
#include<numeric>
#include<iostream>
#include <iterator>

template<typename T>
void print_Vector(const std::vector<T> &v)
{
    std::cout<<"\n[";
    std::copy(v.begin(),v.end(),std::ostream_iterator<T>(std::cout,","));
    std::cout<<"]\n";
}

template<typename T>
void print_Vector2(const std::vector<std::vector<T>> &v)
{
    std::cout<<"\n[";
    for(int i=0;i< v.size();i++)
    {
        std::cout<<"[";
        for (auto j= v[i].begin(); j!=v[i].end();j++)
        {
            std::cout<<*j<<",";
        }
        std::cout<<"],";
    }
    std::cout<<"]\n";
}
template<typename T>
std::vector<size_t> sort_index(const std::vector<T> &v)
{
    std::vector<size_t> index(v.size());
    std::iota(v.begin(),v.end(),0);

    std::stable_sort(index.begin(),index.end(),[&v](size_t l1, size_t l2){
        return v[l1]<v[l2];
    });

}