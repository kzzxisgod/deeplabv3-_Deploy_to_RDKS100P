# deeplabv3-_Deploy_to_RDKS100P
模型转化工具链：天工开物（OE）
参考OE用户手册，完成对deeplabv3+模型的c++板端部署

# 以deeplabv3+为例，记录一次模型部署流程。
1. hb_compile: 将onnx格式模型转化为hbm模型
- 检查onnx模型中有那些算子是BPU不支持的
- 删除特定的节点名称，参考官方给的例程
- 转化的验证集，npy格式数据
- 配置文件 参考官方给出的例子进行修改
- 转化之后会得到一些日志文件和中间模型
- html文件可以查看模型的静态特性
- 动态精度：hrt_model_exec perf --model_file model.hbm
2. hb_model_info：解析hbm和bc编译时的依赖及参数信息、onnx模型基本信息，同时支持对bc可删除节点进行查询
- hb_model_info ${model_file};
- 重点关注输入和输出;
  '''
     input_y  input  [1, 224, 224, 1] UINT8 
     input_uv input  [1, 112, 112, 2] UINT8 
   ​  output   output [1, 1000]        FLOAT32
  '''
- deeplabv3+的输出信息为：
  ''' ​
  2026-01-19 16:22:17,965 INFO input.1_y  input  [1, 1024, 2048, 1] UINT8
  2026-01-19 16:22:17,966 INFO input.1_uv input  [1, 512, 1024, 2]  UINT8
  2026-01-19 16:22:17,966 INFO 705        output [1, 1, 1024, 2048] INT8
  '''
- 它的输出为[1, 1, 1024, 2048]
3. 根据官方的其他模型推理案例的源码来进行测试和修改，主要在于数据的对齐很难搞，输出格式不对不能输出正确的结果。
- 板端部署：统一计算平台(这个是OE工具链里带的，这个工具链的docker镜像是部署在开发机的wsl ubuntu环境)
- 官方示例：RGB输入的ResNet18模型部署

# 模型输入输出：
   	input  [1, 1024, 2048, 1] UINT8
   	input  [1, 512, 1024, 2]  UINT8
   	output [1, 1, 1024, 2048] INT8
最终输出的结果到 seg_mask 中
