//
// File name: layer.h
// Created by ronny on 16-12-2.
//

#ifndef EASYML_NERUALNETWORK_LAYER_H
#define EASYML_NERUALNETWORK_LAYER_H

#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>

#include <easyml/util/activation_function.h>

namespace easyml {

namespace nn {

enum LayerType {
    LAYER_INPUT = 0,
    LAYER_FC = 1,
    LAYER_COV = 2
};

class LayerParameter {										// 每层所具有的层属性
public:
    LayerParameter(
            const std::string &layer_name,
            int layer_type,
            std::shared_ptr<util::ActivationFunction> activation_fun = 	// 使用共享智能指针调用激活函数
            std::shared_ptr<util::ActivationFunction>())
    {
        name = layer_name;
        type = layer_type;
        activation = activation_fun;
    }

    std::string name;
    int type;
    std::shared_ptr<util::ActivationFunction> activation;
};


class Layer {
public:

    /// @brief forward computation

    virtual void FeedForward(								// 纯虚函数-前馈网络
            const std::vector<cv::Mat> &input,
            std::vector<cv::Mat> &output) = 0;


    /// @brief the loss propagate back throung the layer
    
    virtual void BackPropagation(							// 反向传播网络
            const std::vector<cv::Mat> &delta_in,
            std::vector<cv::Mat> &delta_out,
            float eta,
            float lambda) = 0;

    // only for output layer
    virtual void SetLabels(const std::vector<cv::Mat> &labels) = 0; // 输出层设置标签

    virtual ~Layer() = default;

    std::string Name()
    {
        return name_;
    }
    

protected:
    cv::Mat weights_;
    cv::Mat biases_;

    std::vector<cv::Mat> input_;
    std::vector<cv::Mat> weighted_output_;					// 输出权重?  -- kexi=w*x+b

    std::string name_;
    int type_;
    std::shared_ptr<util::ActivationFunction> activation_;
};

} // namespace nn

} // namespace easyml

#endif // EASYML_NERUALNETWORK_LAYER_H

