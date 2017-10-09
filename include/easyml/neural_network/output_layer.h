//
// File name: outpu_layer.h
// Created by ronny on 16-12-6.
//

#ifndef EASYML_NERUALNETWORK_OUTPUT_LAYER_H
#define EASYML_NERUALNETWORK_OUTPUT_LAYER_H


#include <easyml/common.h>
#include <easyml/neural_network/layer.h>
#include <easyml/util/cost_function.h>
#include<vector>
using namespace std;
namespace easyml {
namespace nn {

class OutputLayerParameter : public LayerParameter
{
public:
    OutputLayerParameter(
            const std::string &layer_name,
            Dim in_dim,
            Dim out_dim,
            std::shared_ptr<util::ActivationFunction> activation_fun,
            std::shared_ptr<util::CostFunction> cost)
            :LayerParameter(layer_name, LAYER_FC, activation_fun)
    {
        input_dim = in_dim;
        output_dim = out_dim;
        cost_function = cost;
    }
    Dim input_dim;
    Dim output_dim;
    std::shared_ptr<util::CostFunction> cost_function;
};


class OutputLayer : public Layer
{
public:

    vector<cv::Mat> a_output_;					//为了保存a的输出值, 额外求cost大小加上的. 
    
    explicit OutputLayer(
            const OutputLayerParameter &param);

    /// @brief forward computation
    void FeedForward(
            const std::vector<cv::Mat> &input,
            std::vector<cv::Mat> &output) override;


    /// @brief the loss propagate back throung the layer
    void BackPropagation(
            const std::vector<cv::Mat> &delta_in,
            std::vector<cv::Mat> &delta_out,
            float eta,
            float lambda) override;

    void SetLabels(const std::vector<cv::Mat> &labels) override;
private:
    std::shared_ptr<util::CostFunction> cost_;
    std::vector<cv::Mat> labels_;

};


} // namespace easyml
} // namespace nn



#endif // EASYML_NERUALNETWORK_OUTPUT_LAYER_H

