#include <easyml/neural_network/output_layer.h>
#include <glog/logging.h>

namespace easyml {
namespace nn {



OutputLayer::OutputLayer(const OutputLayerParameter &param)
{
    name_ = param.name;
    type_ = param.type;
    cost_ = param.cost_function;
    activation_ = param.activation;

    biases_ = cv::Mat(param.output_dim.height, 1, CV_32FC1);
    cv::randn(biases_, cv::Scalar::all(0.0f), cv::Scalar::all(1.0f));

    weights_ = cv::Mat(param.output_dim.height, param.input_dim.height, CV_32FC1);
    float init_weight = 1.0f / sqrt(static_cast<float>(param.input_dim.height));
    cv::randn(weights_, cv::Scalar::all(0.0f), cv::Scalar::all(init_weight));
}

void OutputLayer::SetLabels(const std::vector<cv::Mat> &labels) {
    labels_.assign(labels.size(), cv::Mat());
    for (size_t i = 0; i < labels.size(); i++) {
        labels_[i] = labels[i].clone();
    }
}


void OutputLayer::FeedForward(
        const std::vector<cv::Mat> &input,
        std::vector<cv::Mat> &output)
{

    input_.assign(input.size(), cv::Mat());
    int batch_size = input_.size();
    output.assign(batch_size, cv::Mat());
    weighted_output_.assign(batch_size, cv::Mat());

    a_output_.assign(batch_size, cv::Mat());
    
    for (int i = 0; i < batch_size; i++) {
        input_[i] = input[i].clone();
        weighted_output_[i] = weights_ * input_[i] + biases_;								// 这就是 z
        output[i] = (*activation_)(weighted_output_[i]);										// 这是 a
	a_output_[i] = output[i].clone();													// 保存a输出值 额外求cost大小加上的. 
    }
}

void OutputLayer::BackPropagation(
        const std::vector<cv::Mat> &delta_in,
        std::vector<cv::Mat> &delta_out,
        float eta,
        float lambda)
{
    cv::Mat nabla_w_sum(weights_.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat nabla_b_sum(biases_.size(), CV_32FC1, cv::Scalar(0.0f));

    int batch_size = input_.size();
    delta_out.assign(batch_size, cv::Mat());

    for (int i = 0; i < batch_size; i++) {													//  x  -->  wx+b  -->   z -->   sigmoid  --> a --->cost = (a-y)^2;  则delta_in为delta(a); dleta_out为delta(z)
        
        float cos = (*cost_)(a_output_[i], labels_[i]);										// 额外求cost大小加上的. 

        cv::Mat a1 = cost_->CostDerivation((*activation_)(weighted_output_[i]), labels_[i]);		// 得到delta( a )
        cv::Mat b1 = activation_->primer(weighted_output_[i]);								// a(1-a) 为sigmoid()导数
        delta_out[i] = cost_->CostDerivation((*activation_)(weighted_output_[i]), labels_[i]).mul(activation_->primer(weighted_output_[i]));
        cv::Mat nabla_b = delta_out[i].clone();												// activation_->primer(weighted_output_[i])可能有错?,没错!! 应为里面已经实现求sigmoid的值了,输出应该为delta( z )
        cv::Mat nabla_w = nabla_b * input_[i].t();											//delta( w )  =  delta(z) * X^T
        nabla_w_sum += nabla_w;														// 对批量的梯度求和
        nabla_b_sum += nabla_b;
        delta_out[i] = weights_.t() * delta_out[i];											// delta( x ) = W^T * delta( z )
    }
    weights_ *= (1 - eta * lambda); // L2 regularization
    weights_ -= (eta / batch_size) * nabla_w_sum;
    biases_ -= (eta / batch_size) * nabla_b_sum;
}

} // namespace easyml
} // namespace nn


