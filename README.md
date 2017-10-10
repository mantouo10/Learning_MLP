# This project copy from https://github.com/ronnyyoung/EasyML.

I just make some notes(chinese).

# 1.Recomend: Using the kdevelop IDE.
(You can build the project by cmake/make, just like the command line in the shell. It is different with VS in windows)

This project is very easy, but it is very hard when you try to re-write it.
# 2.So, I hope you can establish the project by yourself.

# 3.http://blog.csdn.net/taigw/article/details/50612963 shows the back propagation(BP) algorithm. If you forget the BP theory, you can take a look.

I have changed the building mode to "Debug". if you want to debug the project, you just need to set stop-points.

The opencv is need for this project. If you never install opencv, you can find the install methods in http://www.cnblogs.com/TooyLee/p/6055330.html. 




# EasyML
An easy reading implementation of popular machine learning algorithm  with cpp for the beginner.


# Quick Start

## 1. Build using cmake

```
$ cd $root_dir
$ mkdir build
$ cd build
$ cmake ..
$ make

```

## 2. Run the handwritting recognization exmaple

```
$ cd root_dir/build/
$ ./examples/handwriting_recognize.bin
```

## 3. Check the result

After execuate above command. You'll find some debug message will print like:

```
I1110 14:54:44.299903 19049 handwriting_recognize.cpp:23] Start to train
I1110 14:54:52.198222 19049 neural_network.cpp:93] Epoch 0: 8655 / 10000
I1110 14:55:00.036250 19049 neural_network.cpp:93] Epoch 1: 9056 / 10000
I1110 14:55:07.868829 19049 neural_network.cpp:93] Epoch 2: 9245 / 10000
I1110 14:55:15.929745 19049 neural_network.cpp:93] Epoch 3: 9332 / 10000
I1110 14:55:23.337919 19049 neural_network.cpp:93] Epoch 4: 9391 / 10000
I1110 14:55:30.378495 19049 neural_network.cpp:93] Epoch 5: 9439 / 10000
...
```


## 4. TODO

1. Refactor the structure
2. Add ReLU activation
3. Add dropout
4. Add SoftMax layer
5. Add convolutional layer and pooling layer
6. Add moment optimation
