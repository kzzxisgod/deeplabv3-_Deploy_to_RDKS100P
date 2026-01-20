#include <iostream>
#include <vector>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

///////////////////////////////////// 保存与可视化 //////////////////////////////////////////
/**
 * @brief 将分割掩码转化为彩色图像
 * @param mask 预测结果 (单通道灰度，值为类别ID)
 * @param width 图像宽度
 * @param height 图像高度
 * @param filename 保存的路径
 */
void save_segmentation_result(const std::vector<uint8_t>& mask, int width, int height, const std::string& filename) {
    if (mask.empty()) return;
    // 将 mask 向量转为 cv::Mat，将vector 数据映射为OpenCV的单通道矩阵(CV_8UC1)
    cv::Mat mask_mat(height, width, CV_8UC1, (void*)mask.data());
    // 存放最终彩色结果的矩阵
    cv::Mat color_mat;
    // 1. mask_mat * 15: 将较小的类别索引（如 0, 1, 2）放大，以便在伪彩色映射中产生明显的颜色区分
    // 2. applyColorMap: 将灰度值映射为 COLORMAP_JET 调色板颜色（蓝-绿-红渐变）
    cv::applyColorMap(mask_mat * 15, color_mat, cv::COLORMAP_JET); 
    // 将处理后的彩色图像写入文件
    cv::imwrite(filename, color_mat);
    std::cout << "Successfully saved segmentation result to " << filename << std::endl;
}

/**
 * 将分割结果叠加到原图上
 * @param src 原图 (BGR)
 * @param mask 预测结果 (单通道灰度，值为类别ID)
 * @param filename 保存的文件名
 */
void blend_segmentation(cv::Mat& src, const std::vector<uint8_t>& mask, const std::string& filename) {
    int h = src.rows; // 矩阵行数，对应图像的高度
    int w = src.cols; // 矩阵列数，对应图像的宽度

    // 1. 将 mask 向量转为 cv::Mat
    cv::Mat mask_mat(h, w, CV_8UC1, (void*)mask.data());
    // 2. 将 ID 映射为彩色图 (使用 COLORMAP_JET)
    cv::Mat color_mask;
    cv::applyColorMap(mask_mat * 15, color_mask, cv::COLORMAP_JET);
    // 3. 图像融合：dst = src * 0.6 + color_mask * 0.4 + 0
    cv::Mat blended;
    cv::addWeighted(src, 0.6, color_mask, 0.4, 0, blended);
    // 4. 保存并显示
    cv::imwrite(filename, blended);
    std::cout << "Blended result saved to: " << filename << std::endl;
}

///////////////////////////////////// 数据处理 /////////////////////////////////////
/**
 * @brief 准备分量分离的 NV12 输入数据，将图像的 Y 分量和 UV 分量分别填充到不同的张量中
 * @param image_path 本地图像路径
 * @param inputs     输入张量数组（inputs[0]存放Y分量，inputs[1]存放UV分量）
 * @return int       成功返回 0，读取失败或张量数量不足返回 -1
 */
int prepare_nv12_input(const std::string& image_path, std::vector<hbDNNTensor>& inputs) {
    // 1. 检查输入张量数量，分离模式下至少需要两个张量（Y 和 UV）
    if (inputs.size() < 2) return -1;
    // 2. 获取模型期望的输入尺寸信息（以 Y 分量张量为准）
    int h = inputs[0].properties.validShape.dimensionSize[1]; // 图像高度
    int w = inputs[0].properties.validShape.dimensionSize[2]; // 图像宽度
    // 3. 使用 OpenCV 读取原始图像
    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) return -1;
    // 4. 将图像缩放到模型要求的尺寸
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(w, h));
    // 5. 将图像从 BGR 转换为 YUV420P (I420) 格式
    // I420 内存布局为：YYYY... UUUU... VVVV...
    cv::Mat yuv420p;
    cv::cvtColor(resized, yuv420p, cv::COLOR_BGR2YUV_I420);
    // ---------------------- 填充 Input[0] (Y 分量) ----------------------
    // 获取 Y 分量在系统中的虚拟内存地址
    uint8_t* y_src = yuv420p.data;
    uint8_t* y_dest = reinterpret_cast<uint8_t*>(inputs[0].sysMem.virAddr);
    // 获取 Y 分量的内存跨度 (Stride)
    int y_stride = static_cast<int>(inputs[0].properties.stride[1]);
    // 6. 逐行拷贝 Y 分量（亮度信息）
    for (int i = 0; i < h; ++i) {
        // 考虑内存对齐，将有效像素拷贝到带有 stride 的目标内存
        std::memcpy(y_dest + i * y_stride, y_src + i * w, w);
    }
    // 刷新 Y 分量缓存，确保数据同步到硬件
    hbUCPMemFlush(&inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    // ---------------------- 填充 Input[1] (UV 分量) ----------------------
    // 7. 定位 I420 格式中 U 和 V 分量的起始位置
    uint8_t* u_src = yuv420p.data + (h * w);         // U 分量起点
    uint8_t* v_src = u_src + (h * w / 4);             // V 分量起点
    uint8_t* uv_dest = reinterpret_cast<uint8_t*>(inputs[1].sysMem.virAddr);
    // 获取 UV 分量的内存跨度 (Stride)
    int uv_stride = static_cast<int>(inputs[1].properties.stride[1]);
    // 8. 将 U/V 分量重新交织为 NV12 要求的 UVUV... 结构并填充
    for (int i = 0; i < h / 2; ++i) {
        uint8_t* row_dest = uv_dest + i * uv_stride; // 目标行指针
        for (int j = 0; j < w / 2; ++j) {
            // NV12 格式：在同一平面内，UV 像素交替排列
            row_dest[j * 2] = u_src[i * (w / 2) + j];     // 偶数下标存 U
            row_dest[j * 2 + 1] = v_src[i * (w / 2) + j]; // 奇数下标存 V
        }
    }
    // 刷新 UV 分量缓存，确保硬件可访问最新数据
    hbUCPMemFlush(&inputs[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    return 0;
}

////////////////////////////////////// main函数 ///////////////////////////////////////
int main(int argc, char **argv) {
    // hbDNNPackedHandle_t：指向打包的多个模型。
    hbDNNPackedHandle_t packed_dnn_handle;
    // 指定模型文件路径
    const char* model_file_name = "/home/sunrise/Desktop/deeplabv3-_Deploy_to_RDKS100P/model/deeplabv3plus_efficientnetm2_1024x2048_nv12.hbm";
    const char* image_file_name = "/home/sunrise/Desktop/deeplabv3-_Deploy_to_RDKS100P/tem/test.png";
    const char* save_mask_path = "/home/sunrise/Desktop/deeplabv3-_Deploy_to_RDKS100P/output/output_mask.png";
    const char* save_blended_path = "/home/sunrise/Desktop/deeplabv3-_Deploy_to_RDKS100P/output/output_blended.png";
    /*
    从文件完成对dnnPackedHandle的创建和初始化。调用方法可以跨函数、跨线程使用返回的dnnPackedHandle
    * hbDNNPackedHandle_t *dnnPackedHandle：指向多个模型
    * char const **modelFileNames：模型文件路径
    * int32_t modelFileCount：模型文件数量
    * return：0 表示API成功
    int32_t hbDNNInitializeFromFiles(hbDNNPackedHandle_t *dnnPackedHandle,
                                 char const **modelFileNames,
                                 int32_t modelFileCount);
    */
    int ret = hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
    if (ret != 0) return -1;

    const char **model_name_list;
    int model_count = 0;
    /*
    获取dnnPackedHandle中包含的模型名称列表和个数
    * char const ***modelNameList：模型名称列表
    * int32_t *modelNameCount：模型名称数量
    * hbDNNPackedHandle_t dnnPackedHandle：指向多个模型
    * return：0 表示API成功
    int32_t hbDNNGetModelNameList(char const ***modelNameList, 
                              int32_t *modelNameCount,
                              hbDNNPackedHandle_t dnnPackedHandle);
    */
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    // hbDNNHandle_t：指向单一模型
    hbDNNHandle_t dnn_handle;
    /*
    从dnnPackedHandle所指向模型列表中获取一个模型的句柄，调用方可以跨函数、跨线程使用返回的dnnHandle
    * hbDNNHandle_t *dnnHandle：指向一个模型
    * hbDNNPackedHandle_t dnnPackedHandle：指向多个模型
    * char const *modelName：模型名称
    * return：0 表示API成功
    int32_t hbDNNGetModelHandle(hbDNNHandle_t *dnnHandle,
                            hbDNNPackedHandle_t dnnPackedHandle,
                            char const *modelName);
     */
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    int input_count = 0;
    /*
    获取dnnHandle所指向模型的输入tensor数量
    * int32_t *inputCount：输入tensor数量
    * hbDNNHandle_t dnnHandle：指向一个模型
    * return：0 表示API成功
    int32_t hbDNNGetInputCount(int32_t *inputCount, 
                           hbDNNHandle_t dnnHandle);
    */
    hbDNNGetInputCount(&input_count, dnn_handle);
    // hbDNNTensor：输入tensor，用于存放输入输出的信息
    std::vector<hbDNNTensor> input(input_count);
    /*
    获取dnnHandle所指向模型的输入tensor属性
    * hbDNNTensorProperties *properties：输入tensor属性
    * hbDNNHandle_t dnnHandle：指向一个模型
    * int32_t inputIndex：输入tensor索引
    * return：0 表示API成功
    int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties *properties,
                                      hbDNNHandle_t dnnHandle,
                                      int32_t inputIndex);
    */
    for (int i = 0; i < input_count; i++) {
        hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
        auto &props = input[i].properties;

        if (props.stride[0] < 0) {
            // std::cout << "Fixing input[" << i << "] stride..." << std::endl;
            if (i == 0) { // Y分量
                // Input[0]: 1x1024x2048x1 (NV12)
                props.stride[0] = 2097152; 
                props.stride[1] = 2048;
                props.stride[2] = 1;
                props.stride[3] = 1;
                props.alignedByteSize = 2097152; // 1024 * 2048 * 1.5
            } else if (i == 1) { // UV分量
                // Input[1]: 1x512x1024x2 (NV12)
                // 报错要求: stride[0] >= 1048576, stride[1] >= 2048
                props.stride[0] = 1048576; 
                props.stride[1] = 2048;
                props.stride[2] = 2; // 因为最后一个维度是 2
                props.stride[3] = 1;
                props.alignedByteSize = 1048576; // 512 * 2048
            }
        }

        hbUCPMallocCached(&input[i].sysMem, props.alignedByteSize, 0); 
    }
    prepare_nv12_input(image_file_name, input);

    int output_count = 0;
    /*
    获取dnnHandle所指向模型的输出tensor数量
    * int32_t *outputCount：输出tensor数量
    * hbDNNHandle_t dnnHandle：指向一个模型
    * return：0 表示API成功
    int32_t hbDNNGetOutputCount(int32_t *outputCount, 
                            hbDNNHandle_t dnnHandle);
    */
    hbDNNGetOutputCount(&output_count, dnn_handle);
    std::vector<hbDNNTensor> output(output_count);
    /*
    获取dnnHandle所指向模型的输出tensor属性
    * hbDNNTensorProperties *properties：输出tensor的信息
    * hbDNNHandle_t dnnHandle：指向一个模型
    * int32_t outputIndex：输出tensor索引
    * return：0 表示API成功
    int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties *properties,
                                       hbDNNHandle_t dnnHandle, 
                                       int32_t outputIndex);
    */
    for (int i = 0; i < output_count; i++){
        hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        auto &props = output[i].properties;

        if (props.stride[0] < 0) {
            int64_t h = static_cast<int64_t>(props.validShape.dimensionSize[2]);
            int64_t w = static_cast<int64_t>(props.validShape.dimensionSize[3]);
            props.stride[0] = h * w;
            props.stride[1] = h * w;
            props.stride[2] = w;
            props.stride[3] = 1;
            props.alignedByteSize = static_cast<uint64_t>(h * w);
        }
        hbUCPMallocCached(&output[i].sysMem, props.alignedByteSize, 0);
    }

    hbUCPTaskHandle_t task_handle = nullptr;
    ret = hbDNNInferV2(&task_handle, output.data(), input.data(), dnn_handle);
    if (ret != 0 || task_handle == nullptr) return -1;

    hbUCPSchedParam infer_sched_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&infer_sched_param);
    hbUCPSubmitTask(task_handle, &infer_sched_param);
    hbUCPWaitTaskDone(task_handle, 0);

    // 在循环外部先读取一次原图，用于叠加
    cv::Mat original_img = cv::imread(image_file_name);

    for (int i = 0; i < output_count; i++) {
        hbUCPMemFlush(&(output[i].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
        auto &props = output[i].properties;
        
        int h = props.validShape.dimensionSize[2];
        int w = props.validShape.dimensionSize[3];
        int stride = static_cast<int>(props.stride[2]); 

        if (h <= 0 || w <= 0) continue;

        int8_t *raw_ptr = reinterpret_cast<int8_t *>(output[i].sysMem.virAddr);
        std::vector<uint8_t> seg_mask(static_cast<size_t>(h * w));

        for (int row = 0; row < h; ++row) {
            std::memcpy(seg_mask.data() + row * w, raw_ptr + row * stride, static_cast<size_t>(w));
        }

        // 保存纯色分割图
        save_segmentation_result(seg_mask, w, h, save_mask_path);

        // 保存半透明叠加图
        if (!original_img.empty()) {
            cv::Mat resized_src;
            cv::resize(original_img, resized_src, cv::Size(w, h)); // 确保原图尺寸与输出一致
            blend_segmentation(resized_src, seg_mask, save_blended_path);
        }
    }
    // 释放资源
    hbUCPReleaseTask(task_handle);
    for (auto &in : input) hbUCPFree(&in.sysMem);
    for (auto &out : output) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);
    return 0;
}