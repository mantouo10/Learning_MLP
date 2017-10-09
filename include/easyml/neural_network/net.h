//
// File name: net.h
// Created by ronny on 16-12-2.
//

#ifndef EASYML_NERUALNETWORK_NET_H
#define EASYML_NERUALNETWORK_NET_H


#include <memory>
#include <opencv2/core/core.hpp>
#include <easyml/neural_network/layer.h>

namespace easyml {
namespace nn {

struct NNTrainParam {
    int epochs = 30;							// 迭代次数
    int mini_batch_size = 10;					// 最小批次大小
    float eta = 0.1;							// 学习速率
    float lambda = 5;						// 正则化惩罚系数
};


class Net {
public:
    void PushBack(std::shared_ptr<Layer> layer);
    void PushFront(std::shared_ptr<Layer> layer);
    void Insert(std::shared_ptr<Layer> layer, int index);
    void Remove(int index);
    void Train(const cv::Mat &training_set, const cv::Mat &lables, NNTrainParam param,
               const cv::Mat &testing_set = cv::Mat(), const cv::Mat &testing_label = cv::Mat());
    void Predict(const cv::Mat &input, cv::Mat &output);
private:
    void UpdateMiniBatch(
            const std::vector<cv::Mat> &training_data,
            const std::vector<cv::Mat> &labels,
            float eta, float lambda);
private:
    std::vector<std::shared_ptr<Layer>> layers_;
};

} // namespace nn
} // namespace easyml

#endif // EASYML_NERUALNETWORK_NET_H

