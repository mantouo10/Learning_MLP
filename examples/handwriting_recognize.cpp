#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>


#include "neural_network/net.h"
#include "neural_network/input_layer.h"

#include "neural_network/fully_connected_layer.h"
#include "neural_network/output_layer.h"
#include <util/util.h>



int main(int argc, char *argv[]) {
    using namespace easyml;
    using namespace nn;
    // set glog configure
    google::InitGoogleLogging(argv[0]);					// 重定向log的文件位置
    FLAGS_stderrthreshold=google::INFO;				// 输出为出来error外
    FLAGS_colorlogtostderr=true;						// 标上颜色


    cv::Mat training_data;
    cv::Mat testing_data;
    cv::Mat train_labels;
    cv::Mat test_labels;

    if (!util::LoadMNIST("/home/usrp/EasyML-master/data/", training_data, train_labels, testing_data, test_labels)) {
        LOG(ERROR) << "Failed to load mnist data" << std::endl;
        exit(EXIT_FAILURE);
    }
    // data format transfer
    training_data.convertTo(training_data, CV_32F);
    training_data /= 255.0f;								// 对数据进行归一化
    train_labels.convertTo(train_labels, CV_32F);
    testing_data.convertTo(testing_data, CV_32F);
    testing_data /= 255.0f;
    test_labels.convertTo(test_labels, CV_32F);


    // initilize the net
    std::unique_ptr<nn::Net> net(new nn::Net());			// 唯一的不可复制智能指针

    // create input layer
    InputLayerParameter input_param("input");
    std::shared_ptr<nn::Layer> input_layer(new nn::InputLayer(input_param));
    net->PushBack(input_layer);							// 加入第一个层

    std::cout << "complete initilize the input layer" << std::endl;

    FullyConnectedLayerParameter fc1_param(
            "fc1",											// 名字
            Dim(10, 1, 784, 1),								// 输入[batch, channel, height, width ]
            Dim(10, 1, 50, 1),								// 输出
            std::shared_ptr<util::SigmoidFunction>(new util::SigmoidFunction())
    );													// 激活函数 
    std::shared_ptr<nn::Layer> fc1 = std::make_shared<nn::FullyConnectedLayer>(fc1_param);// 创建一个共享的全连接隐藏层
    net->PushBack(fc1);									// 加入第一层隐藏层

    FullyConnectedLayerParameter fc2_param(
            "fc2",
            Dim(10, 1, 50, 1),
            Dim(10, 1, 50, 1),
            std::shared_ptr<util::SigmoidFunction>(new util::SigmoidFunction())
    );
    std::shared_ptr<nn::Layer> fc2 = std::make_shared<nn::FullyConnectedLayer>(fc2_param);
    net->PushBack(fc2);

    std::cout << "complete initilize the fully connected layer" << std::endl;

    OutputLayerParameter output_param(
            "output",
            Dim(10, 1, 50, 1),
            Dim(10, 1, 10, 1),
            std::shared_ptr<util::SigmoidFunction>(new util::SigmoidFunction()),
            std::shared_ptr<util::CostFunction>(new util::CEEFunction())				// 使用交叉熵损失函数
    );
    std::shared_ptr<nn::Layer> output_layer 
        = std::make_shared<nn::OutputLayer>(output_param);
    net->PushBack(output_layer);
    std::cout << "complete initilize the output layer" << std::endl;

    LOG(INFO) << "Start to train" << std::endl;

    nn::NNTrainParam param;												// 训练参数

    net->Train(training_data, train_labels, param, testing_data, test_labels);
    LOG(INFO) << "Complete to train" << std::endl;

    return 0;
}
