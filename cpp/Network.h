#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <condition_variable>
#include <functional>

#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace InferenceEngine;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

struct RequestResult {
    size_t id;
    double latency;
    map<string, Blob::Ptr> data;
};

typedef std::function<void(RequestResult result)> InferCallbackFunction;

class InferReqWrap final {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    ~InferReqWrap() = default;

    explicit InferReqWrap(ExecutableNetwork& net, size_t id,
                          const InputsDataMap& input_info,
                          const OutputsDataMap& output_info,
                          InferCallbackFunction callbackQueue) :
        request_(net.CreateInferRequest()),
        id_(id),
        callbackQueue_(callbackQueue)
    {
        SizeVector input_dims = input_info.begin()->second->getTensorDesc().getDims();
        input_info.begin()->second->setPrecision(Precision::U8);

        auto input_blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, input_dims, Layout::NCHW));
        input_blob->allocate();
        BlobMap inputs;
        inputs[input_info.begin()->first] = input_blob;

        BlobMap output_blobs;
        for(auto& item : output_info) {
            output_names_.push_back(item.first);

            SizeVector outputDims = item.second->getTensorDesc().getDims();
            item.second->setPrecision(Precision::FP32);

            TBlob<float>::Ptr output;
            if(outputDims.size() == 2) {
                output = make_shared_blob<float>(TensorDesc(Precision::FP32, outputDims, Layout::NC));
            } else {
                output = make_shared_blob<float>(TensorDesc(Precision::FP32, outputDims, Layout::NCHW));
            }
            output->allocate();
            output_blobs[item.first] = output;
        }

        request_.SetInput(inputs);
        request_.SetOutput(output_blobs);

        request_.SetCompletionCallback(
                [&]() {
                    endTime_ = Time::now();
                    callbackQueue_({ id_, getExecutionTimeInSeconds(), getBlobs(output_names_) });
                });
    }

    void startAsync()
    {
        startTime_ = Time::now();
        request_.StartAsync();
    }

    void infer()
    {
        startTime_ = Time::now();
        request_.Infer();
        endTime_ = Time::now();
        callbackQueue_({ id_, getExecutionTimeInSeconds(), getBlobs(output_names_) });
    }

    map<string, InferenceEngineProfileInfo> getPerformanceCounts()
    {
        return request_.GetPerformanceCounts();
    }

    Blob::Ptr getBlob(const string& name)
    {
        return request_.GetBlob(name);
    }

    map<string, Blob::Ptr> getBlobs(const vector<string>& names)
    {
        map<string, Blob::Ptr> blobs;

        for(const string& name : names) {
            blobs[name] = request_.GetBlob(name);
        }

        return blobs;
    }

    double getExecutionTimeInSeconds() const
    {
        auto execTime = chrono::duration_cast<ns>(endTime_ - startTime_);
        return static_cast<double>(execTime.count()) * 1e-9;
    }

    inline size_t id() const { return id_; }

private:
    vector<string> output_names_;

    InferRequest request_;
    Time::time_point startTime_;
    Time::time_point endTime_;
    size_t id_;
    InferCallbackFunction callbackQueue_;
};

class Network
{
public:
    Network(const filesystem::path& xml_path_, const string& device_);

    inline size_t getBatchSize() { return network_.getBatchSize(); }

    InferReqWrap::Ptr createRequest(const cv::Mat& image, InferCallbackFunction callback_func);

private:
    void readNetwork();
    void setDeviceConfig();
    void loadToDevice();


private:
    filesystem::path xml_path_;
    filesystem::path bin_path_;

    string device_;

    CNNNetwork network_;
    ExecutableNetwork exe_network_;
    Core inference_engine_;

    InputsDataMap input_info_;
    OutputsDataMap output_info_;
};

class InferRequestsQueue final {
public:
    InferRequestsQueue(Network* net, size_t max_size = 4)
        :
          net_(net), max_size_(max_size)
    {
    }
    ~InferRequestsQueue() = default;

    void resetTimes()
    {
        latencies_.clear();
    }

    void waitAll() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]{ return results_.size() == requests_.size(); });
    }

    std::vector<double> getLatencies() {
        return latencies_;
    }

    void createRequestSync(const cv::Mat& image)
    {
        InferReqWrap::Ptr request;

        {
            unique_lock<mutex> lock(mutex_);
            cv_.wait(lock, [this]{ return results_.size() < max_size_; });

            request = net_->createRequest(image, bind(&InferRequestsQueue::putResult, this,
                                                      std::placeholders::_1));
            requests_[request->id()] = request;
            lock.unlock();
        }

        request->infer();
    }

    void createRequestAsync(const cv::Mat& image)
    {
        InferReqWrap::Ptr request;

        {
            unique_lock<mutex> lock(mutex_);
            cv_.wait(lock, [this]{ return requests_.size() < max_size_; });

            request = net_->createRequest(image, bind(&InferRequestsQueue::putResult, this,
                                                      std::placeholders::_1));
            requests_[request->id()] = request;
            lock.unlock();
        }

        request->startAsync();
    }

    RequestResult getResult() {
        unique_lock<mutex> lock(mutex_);
        cv_.wait(lock, [this]{ return results_.size() > 0; });

        RequestResult result = results_.front();
        results_.pop();
        requests_.extract(result.id);

        cv_.notify_all();
        return result;
    }

    inline bool isEmpty() const
    {
        unique_lock<mutex> lock(mutex_);
        return requests_.empty();
    }

    inline bool isFull() const
    {
        unique_lock<mutex> lock(mutex_);
        return requests_.size() >= max_size_;
    }

    inline size_t size() const
    {
        unique_lock<mutex> lock(mutex_);
        return requests_.size();
    }

    inline bool hasResults() const
    {
        unique_lock<mutex> lock(mutex_);
        return !results_.empty();
    }

private:
    void putResult(const RequestResult& result)
    {
        unique_lock<mutex> lock(mutex_);
        latencies_.push_back(result.latency);
        results_.push(result);
        cv_.notify_all();
    }

private:
    map<size_t, InferReqWrap::Ptr> requests_;

    Network* net_;
    size_t max_size_;

    queue<RequestResult> results_;

    mutable mutex mutex_;
    condition_variable cv_;
    vector<double> latencies_;
};


