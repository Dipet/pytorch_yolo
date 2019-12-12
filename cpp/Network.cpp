#include "Network.h"

#include <vpu/vpu_plugin_config.hpp>

#include <exception>


size_t getTensorWidth(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    const auto& dims = desc.getDims();
    const auto& size = dims.size();
    if ((size >= 2) &&
        (layout == InferenceEngine::Layout::NCHW  ||
         layout == InferenceEngine::Layout::NHWC  ||
         layout == InferenceEngine::Layout::NCDHW ||
         layout == InferenceEngine::Layout::NDHWC ||
         layout == InferenceEngine::Layout::OIHW  ||
         layout == InferenceEngine::Layout::CHW   ||
         layout == InferenceEngine::Layout::HW)) {
        // Regardless of layout, dimensions are stored in fixed order
        return dims.back();
    }
    THROW_IE_EXCEPTION << "Tensor does not have width dimension";
}

size_t getTensorHeight(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    const auto& dims = desc.getDims();
    const auto& size = dims.size();
    if ((size >= 2) &&
        (layout == InferenceEngine::Layout::NCHW  ||
         layout == InferenceEngine::Layout::NHWC  ||
         layout == InferenceEngine::Layout::NCDHW ||
         layout == InferenceEngine::Layout::NDHWC ||
         layout == InferenceEngine::Layout::OIHW  ||
         layout == InferenceEngine::Layout::CHW   ||
         layout == InferenceEngine::Layout::HW)) {
        // Regardless of layout, dimensions are stored in fixed order
        return dims.at(size - 2);
    }
    THROW_IE_EXCEPTION << "Tensor does not have height dimension";
}

size_t getTensorChannels(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    if (layout == InferenceEngine::Layout::NCHW  ||
        layout == InferenceEngine::Layout::NHWC  ||
        layout == InferenceEngine::Layout::NCDHW ||
        layout == InferenceEngine::Layout::NDHWC ||
        layout == InferenceEngine::Layout::C     ||
        layout == InferenceEngine::Layout::CHW   ||
        layout == InferenceEngine::Layout::NC    ||
        layout == InferenceEngine::Layout::CN) {
        // Regardless of layout, dimensions are stored in fixed order
        const auto& dims = desc.getDims();
        switch (desc.getLayoutByDims(dims)) {
            case InferenceEngine::Layout::C:     return dims.at(0);
            case InferenceEngine::Layout::NC:    return dims.at(1);
            case InferenceEngine::Layout::CHW:   return dims.at(0);
            case InferenceEngine::Layout::NCHW:  return dims.at(1);
            case InferenceEngine::Layout::NCDHW: return dims.at(1);
            case InferenceEngine::Layout::SCALAR:   // [[fallthrough]]
            case InferenceEngine::Layout::BLOCKED:  // [[fallthrough]]
            default:
                THROW_IE_EXCEPTION << "Tensor does not have channels dimension";
        }
    }
    THROW_IE_EXCEPTION << "Tensor does not have channels dimension";
}

cv::Mat resize(const cv::Mat& img, size_t width, size_t height)
{
    if(img.cols == int(width) || img.rows == int(height)) {
        return img;
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(int(width), int(height)));

    return resized;
}

std::shared_ptr<uint8_t> imageData(const cv::Mat& img)
{
    size_t size = size_t(img.size().width * img.size().height * img.channels());
    shared_ptr<uint8_t> data;
    data.reset(new uint8_t[size], default_delete<uint8_t[]>());
    for (size_t id = 0; id < size; ++id) {
        data.get()[id] = img.data[id];
    }
    return data;
}


void fillBlobImage(Blob::Ptr& inputBlob,
                  const cv::Mat& image,
                  const InputInfo& info) {
    // TODO Поддержка батчей.
    auto inputBlobData = inputBlob->buffer().as<uint8_t*>();
    const TensorDesc& inputBlobDesc = inputBlob->getTensorDesc();

    TensorDesc desc = info.getTensorDesc();
    std::shared_ptr<uint8_t> image_data = imageData(resize(image, getTensorWidth(desc), getTensorHeight(desc)));
    if (!image_data) {
        return;
    }

    const size_t numChannels = getTensorChannels(inputBlobDesc);
    const size_t imageSize = getTensorWidth(inputBlobDesc) * getTensorHeight(inputBlobDesc);

    for (size_t pid = 0; pid < imageSize; pid++) {
        for (size_t ch = 0; ch < numChannels; ++ch) {
            size_t index = 0;
            inputBlobData[index] = image_data.get()[pid * numChannels + ch];
        }
    }
}

Network::Network(const filesystem::path& xml_path, const string& device)
    :
      xml_path_(xml_path), bin_path_(xml_path),  device_(device)
{
    bin_path_.replace_extension(".bin");

    readNetwork();
    setDeviceConfig();
    loadToDevice();
}

void Network::readNetwork()
{
    CNNNetReader net_builder;
    net_builder.ReadNetwork(xml_path_);
    net_builder.ReadWeights(bin_path_);

    network_ = net_builder.getNetwork();
    input_info_ = InputsDataMap(network_.getInputsInfo());
    if (input_info_.empty()) {
        throw logic_error("no inputs info is provided");
    }
    if(input_info_.size() > 1) {
        throw logic_error("need only one input. Got: " + to_string(input_info_.size()));
    }

    output_info_ = network_.getOutputsInfo();
    if (output_info_.empty()) {
        throw std::logic_error("no outputs info is provided");
    }

    for (auto& item : input_info_) {
        item.second->setPrecision(Precision::U8);
    }
}

void Network::setDeviceConfig()
{
    if (device_ != "MYRIAD") {
        throw logic_error("Only MYRIAD supports. Got: " + device_);
    }
    inference_engine_.SetConfig({{ CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE) },
                                { VPU_CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING) }}, device_);
}

void Network::loadToDevice()
{
    map<string, string> config = {{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(NO) }};
    exe_network_ = inference_engine_.LoadNetwork(network_, device_, {});
}

InferReqWrap::Ptr Network::createRequest(const cv::Mat& image, InferCallbackFunction callback_func)
{
    static size_t id = 0;

    auto infer_request = make_shared<InferReqWrap>(exe_network_, id++, input_info_, output_info_, callback_func);

    for (const InputsDataMap::value_type& item : input_info_) {
        // TODO поддержка нескольких входов.
        Blob::Ptr input_blob = infer_request->getBlob(item.first);
        fillBlobImage(input_blob, image, *item.second);
        break;
    }

    return infer_request;
}
