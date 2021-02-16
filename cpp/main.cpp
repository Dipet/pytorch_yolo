#include <iostream>
#include <filesystem>

#include <inference_engine.hpp>
#include <thread>
#include <atomic>

#include <opencv2/opencv.hpp>
#include "Network.h"

using namespace std;
using namespace InferenceEngine;

void showAvailableDevices() {
    Core ie;
    vector<string> devices = ie.GetAvailableDevices();

    cout << endl;
    cout << "Available target devices:";
    for (const auto& device : devices) {
        std::cout << "  " << device;
    }
    cout << endl;
}

filesystem::path getConfigPath(int argc, char **argv) {
    if (argc < 2) {
        throw std::logic_error("Need config path");
    }

    string config = argv[1];

    if (config.empty()) {
        throw std::logic_error("Parameter -c is not set");
    }

    filesystem::path config_path = config;
    if (!filesystem::exists(config_path)) {
        throw std::invalid_argument("Config path does not exists: " + config);
    }

    return config_path;
}

template <typename T>
T getMedianValue(const vector<T>& vec) {
    if(vec.empty()) {
        return 0;
    }
    if(vec.size() == 1) {
        return vec.front();
    }

    vector<T> sorted_vec(vec);
    sort(sorted_vec.begin(), sorted_vec.end());

    if(sorted_vec.size() % 2 != 0) {
        return sorted_vec[sorted_vec.size() / 2];
    }
    return (sorted_vec[sorted_vec.size() / 2] + sorted_vec[sorted_vec.size() / 2 - 1]) / static_cast<T>(2.0);
}

void resultGetter(InferRequestsQueue* infer_queue, atomic_bool* stop_flag)
{
    while(!(*stop_flag) || !infer_queue->isEmpty()) {
        auto result = infer_queue->getResult();
        cout << "ID: " << to_string(result.id) << "; Latency: " << to_string(result.latency) << endl;
    }
}

int main(int argc, char **argv)
{
//    const std::string net_path = "/home/druzhinin/HDD/GitRepos/OpenVINO-YoloV3/lrmodels/YoloV3/FP16/frozen_yolo_v3_320.xml";
//    const std::string net_path = "/home/druzhinin/HDD/GitRepos/OpenVINO-YoloV3/lrmodels/YoloV3/FP16/frozen_yolo_v3_320x96.xml";
//    const std::string net_path = "/home/druzhinin/HDD/GitRepos/OpenVINO-YoloV3/lrmodels/YoloV3/FP16/frozen_yolo_v3_416x128.xml";
//    const std::string net_path = "/home/druzhinin/HDD/GitRepos/OpenVINO-YoloV3/lrmodels/YoloV3/FP16/frozen_yolo_v3_416.xml";
    const std::string net_path = "/home/druzhinin/HDD/GitRepos/OpenVINO-YoloV3/lrmodels/tiny-YoloV3/FP16/frozen_tiny_yolo_v3.xml";
//    const std::string net_path = "/home/druzhinin/HDD/GitRepos/OpenVINO-YoloV3/lrmodels/YoloV3/FP16/frozen_yolo_v3_608x192.xml";

    cout << "InferenceEngine: " << GetInferenceEngineVersion() << endl;
    showAvailableDevices();

    cout << "Create and load network: ";

    auto start_time = Time::now();
    auto exec_time = chrono::duration_cast<ns>(Time::now() - start_time).count() * 1e-9;

    Network network(net_path, { "MYRIAD" });
    InferRequestsQueue infer_queue(&network, 4);

    exec_time = chrono::duration_cast<ns>(Time::now() - start_time).count() * 1e-9;
    cout << to_string(exec_time) << endl;

    cv::Mat image = cv::Mat(1080, 1920, CV_8UC3);

    cout << "Warming up: ";
    start_time = Time::now();

    // warming up
    infer_queue.createRequestSync(image);
    infer_queue.getResult();
    infer_queue.resetTimes();

    exec_time = chrono::duration_cast<ns>(Time::now() - start_time).count() * 1e-9;
    cout << to_string(exec_time) << endl;

    atomic_bool stop_flag = false;
    thread reader(resultGetter, &infer_queue, &stop_flag);

    start_time = Time::now();

    const size_t NUM_ITERS = 100;
    for(size_t i = 0; i < NUM_ITERS; ++i) {
        infer_queue.createRequestAsync(image);
    }
    stop_flag = true;
    if(reader.joinable()) {
        reader.join();
    }
    exec_time = std::chrono::duration_cast<ns>(Time::now() - start_time).count() * 1e-9;

    auto latencies = infer_queue.getLatencies();

    double median_latency = getMedianValue(latencies);

    cout << "Total infer time: " << to_string(exec_time) << endl;
    cout << "FPS by total infer time: " << to_string(NUM_ITERS / exec_time) << endl;

    cout << "Median latency: " << to_string(median_latency) << endl;
    cout << "FPS by latency: " << to_string(1 / median_latency) << endl;

    return 0;
}
