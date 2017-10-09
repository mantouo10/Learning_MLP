//
// File name: common.h
// Created by ronny on 16-12-2.
//

#ifndef EASYML_COMMON_H
#define EASYML_COMMON_H

namespace easyml {

struct Dim {
    Dim() = default;										// 定义默认构造函数
    explicit Dim(int b, int c, int h, int w):					// 取消隐式转换,减少逻辑出错.
        batch_size(b), channels(c), height(h), width(w) {}

    Dim(const Dim &dim) {
        batch_size = dim.batch_size;
        channels = dim.channels;
        height = dim.height;
        width = dim.width;
    }

    Dim &operator=(const Dim &dim) {
        batch_size = dim.batch_size;
        channels = dim.channels;
        height = dim.height;
        width = dim.width;
        return *this;										// 重载赋值运算符号,返回一定加上&
    }

    int batch_size;
    int channels;
    int height;
    int width;
};

} // namespace easyml

#endif // EASYML_COMMON_H

