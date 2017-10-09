#include <easyml/neural_network/fully_connected_layer.h>
#include <glog/logging.h>

namespace easyml {
namespace nn {

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedLayerParameter &param)
{
    name_ = param.name;															// 层的名字
    type_ = param.type;																// 层的类型

    biases_ = cv::Mat(param.output_dim.height, 1, CV_32FC1);								// output_dim 具有[batch channel row hight]
    cv::randn(biases_, cv::Scalar::all(0.0f), cv::Scalar::all(1.0f));

    weights_ = cv::Mat(param.output_dim.height, param.input_dim.height, CV_32FC1);		// 初始化化权重[out in]
    float init_weight = 1.0f / sqrt(static_cast<float>(param.input_dim.height));
    cv::randn(weights_, cv::Scalar::all(0.0f), cv::Scalar::all(init_weight));

    activation_ = param.activation;													// 层的激活函数

}


void FullyConnectedLayer::FeedForward(
        const std::vector<cv::Mat> &input,
        std::vector<cv::Mat> &output)
{
    input_.assign(input.size(), cv::Mat());

    int batch_size = input_.size();
    output.assign(batch_size, cv::Mat());
    weighted_output_.assign(batch_size, cv::Mat());

    for (int i = 0; i < batch_size; i++) {
        input_[i] = input[i].clone();
        weighted_output_[i] = weights_ * input_[i] + biases_;				// z = w * x + b
        output[i] = (*activation_)(weighted_output_[i]);						// a = sigmoid( z )
    }
}

void FullyConnectedLayer::BackPropagation(
        const std::vector<cv::Mat> &delta_in,
        std::vector<cv::Mat> &delta_out,
        float eta,
        float lambda)
{
    cv::Mat nabla_w_sum(weights_.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat nabla_b_sum(biases_.size(), CV_32FC1, cv::Scalar(0.0f));

    int batch_size = input_.size();
    delta_out.assign(batch_size, cv::Mat());

    for (int i = 0; i < batch_size; i++) {										// 以下实现对 x/w/b 三个变量的求导, 以及更新
        delta_out[i] = delta_in[i].mul(activation_->primer(weighted_output_[i]));	//  x  -->  wx+b  -->   z -->   sigmoid  --> a;  则delta_in为delta(a); dleta_out为delta(z)
        cv::Mat nabla_b = delta_out[i].clone();									// delta(b) = delta(z)
        cv::Mat nabla_w = nabla_b * input_[i].t();								// delta(w) = delta(z) * X^T
        nabla_w_sum += nabla_w;
        nabla_b_sum += nabla_b;											// 把所有的样本批次的梯度求和
        delta_out[i] = weights_.t() * delta_out[i];								// delta(x) = w^T * dleta(z) 供下一层使用,即delta_in(i-1)
    }
    weights_ *= (1 - eta * lambda); // L2 regularization						// eta为学习率,lambda为L2惩罚系数 
    weights_ -= (eta / batch_size) * nabla_w_sum;							// weights_更新后权重
    biases_ -= (eta / batch_size) * nabla_b_sum;								// biases_更新后权重
}

} // namespace nn
} // namespace easyml
