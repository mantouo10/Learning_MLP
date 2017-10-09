//
// File name: cost_function.h
// Created by ronny on 16-12-2.
// Copyright (c) 2016 SenseNets. All rights reserved.
//

#ifndef EASYML_NEURALNETWORK_COSTFUNCTION_H
#define EASYML_NEURALNETWORK_COSTFUNCTION_H

#include <string>
#include <opencv2/core/core.hpp>

namespace easyml {
namespace util {

class CostFunction {
public:
    virtual float operator()(const cv::Mat &output, const cv::Mat &label) = 0;
    virtual cv::Mat CostDerivation(const cv::Mat &output, const cv::Mat &label) = 0;
    virtual ~CostFunction() { }
};

class MSEFuntion : public CostFunction
{
public:
    inline float operator()(const cv::Mat &output, const cv::Mat &label) override				// L1 范数损失函数
    {
        return cv::norm(label - output);
    }
    inline cv::Mat CostDerivation(const cv::Mat &output, const cv::Mat &label) override
    {
        return (output - label);															// L1 的偏导数
    }
    ~MSEFuntion() {}
};


class CEEFunction : public CostFunction												//交叉熵函数
{
public:
    inline float operator()(const cv::Mat &output, const cv::Mat &label) override
    {
        cv::Mat log_1, log_2;
        cv::log(output, log_1);
        cv::log(1 - output, log_2);
        return cv::sum(-(label.mul(log_1) + (1 - label).mul(log_2)))[0];
    }

    inline cv::Mat CostDerivation(const cv::Mat &output, const cv::Mat &label) override
    {
        return (output - label).mul(1.0f / (output.mul(1.0f - output)));							// 交叉熵的偏导数
    }

    ~CEEFunction() { }
};

} // namepsace util
} // namespace easyml


#endif // EASYML_NEURALNETWORK_COSTFUNCTION_H


