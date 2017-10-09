//
// File name: net.cpp
// Created by ronny on 16-12-2.
//

#include <glog/logging.h>
#include <easyml/neural_network/net.h>
#include <easyml/util/util.h>
#include <iostream>

namespace easyml {
namespace nn {

void Net::Train(const cv::Mat &training_set, const cv::Mat &label_set, NNTrainParam param,
    const cv::Mat &testing_set, const cv::Mat &testing_label)			// 测试的数据没用到
{
    cv::Mat training_data = training_set.clone();
    cv::Mat labels = label_set.clone();

    // training and testing in every epoch
    for (int i = 0; i < param.epochs; i++) {
        // sampling randomly from the training data
        util::RandomShuffle(training_data, labels);
        for (int k = 0; k < training_data.rows; k = k + param.mini_batch_size) {
            int stop_row = std::min(k + param.mini_batch_size, training_data.rows);			// 防止训练数据少于批次大小
            std::vector<cv::Mat> input;
            std::vector<cv::Mat> target;
            for (int j = k; j < stop_row; j++) {
                input.push_back(training_data.row(j).t());									// 把样本行转成列保存到input以及target
                target.push_back(labels.row(j).t());
            }
            // for one iteration.
            UpdateMiniBatch(input, target, param.eta, param.lambda / training_set.rows);	// lambda是除以批次训练集合的大小
        }

        if (testing_set.data) {
            int accuracy = 0;
            cv::Mat predict;
            Predict(testing_set, predict);												// predict为测试数据输出的结果
            for (int i = 0; i < predict.rows; i++) {
                cv::Point pos1;
                cv::minMaxLoc(predict.row(i), nullptr, nullptr, nullptr, &pos1);				// 向量中; pos1:最大值的位置
                cv::Point pos2;
                cv::minMaxLoc(testing_label.row(i), nullptr, nullptr, nullptr, &pos2);			// 向量中;pos2:最大值的位置
                accuracy += (pos1.x == pos2.x);
            }
            LOG(INFO) << "Epoch " << i << ": " << accuracy								// accuracy 是0/1
                      << " / " << testing_set.rows << std::endl;
        }
        else {
            LOG(INFO) << "Epoch " << i << ": completed!" << std::endl;
        }
    }
}

void Net::PushBack(std::shared_ptr<Layer> layer)
{
    layers_.push_back(layer);										//  layers_ 是一个layer容器
}

void Net::PushFront(std::shared_ptr<Layer> layer)
{
    Insert(layer, 0);
}

void Net::Insert(std::shared_ptr<Layer> layer, int index)
{
    auto iter = layers_.begin() + index;
    layers_.insert(iter, layer);										// 利用容器插入实现layer的插入
}

void Net::Remove(int index)
{
    auto iter = layers_.begin() + index;
    layers_.erase(iter);
}


void Net::Predict(const cv::Mat &input, cv::Mat &output)
{
    std::vector<cv::Mat> input_data;
    std::vector<cv::Mat> output_data;
    for (int i = 0; i < input.rows; ++i) {
        input_data.push_back(input.row(i).t());
    }
    for (size_t i = 0; i < layers_.size(); ++i) {						//  循环把前一层的输出作为下一层的输入
        layers_[i]->FeedForward(input_data, output_data);
        input_data = output_data;
    }
    output = cv::Mat(input.rows, output_data[0].rows, CV_32FC1);	// 由于标签是onehot类型,所以output为二维的
    for (size_t i = 0; i < output_data.size(); ++i) {
        output.row(i) = output_data[i].t();
    }
}



void Net::UpdateMiniBatch(
        const std::vector<cv::Mat> &training_data,
        const std::vector<cv::Mat> &labels,
        float eta, float lambda)
{
    int batch_size = training_data.size();
    std::vector<cv::Mat> input = training_data;
    std::vector<cv::Mat> output;

    int num_layer = layers_.size();

    // forward throung the netwrok
    for (int i = 0; i < num_layer; i++) {					// 循环把前一层的输出作为下一层的输入
        layers_[i]->FeedForward(input, output);
        input = output;
    }
    layers_[num_layer - 1]->SetLabels(labels);			// 对最后一层加入标签

    
    
    // backpropagation through the network
    // the delta in the last layer will compute in the output layer.
    std::vector<cv::Mat> delta;
    std::vector<cv::Mat> delta_pre;
    for (int i = num_layer - 1; i >= 0; --i) {				// 循环把最后一层的梯度往前一层传递, delta对最后一层来说没用
        layers_[i]->BackPropagation(delta, delta_pre, eta, lambda);
        delta = delta_pre;
    }
}



} // namespace nn
} // namespace easyml


