# 项目目录

项目全量目录层级介绍如下：

```bash
├─bridge                                           # mbridge在线权重转换
├─checkpoint                                       # 离线权重转换工具
│  ├─common                                        # 离线权重转换工具通用方法
│  ├─sora_model                                    # 多模态生成类模型离线权重转换
│  └─vlm_model                                     # 多模态理解类模型离线权重转换
├─ci                                               # 持续集成模块
├─docs                                             # 项目文档目录
│  ├─en
│  └─zh                                            # 中文文档目录
│      ├─features                                  # 特性说明文档
│      ├─mindspore                                 # mindspore后端迁移文档
│      └─pytorch                                   # pytorch后端迁移文档
├─examples                                         # 所有模型运行脚本和README目录
│  ├─<model_name>                                  # 某个模型的脚本    
│  │  ├─xxx.sh                                     # 启动脚本
│  │  ├─xxx.json/yaml                              # 配置文件
│  │  └─README.md                                  # 运行说明
│  ├─diffsynth                                     # DiffSynth相关模型支持
│  ├─diffusers                                     # Diffusers相关模型支持
│  └─rl                                            # 多模态强化学习相关模型支持 
├─mindspeed_mm                                     # 核心代码目录
│  ├─configs                                       # 配置文件读取和处理代码
│  ├─data                                          # 数据处理代码
│  ├─mindspore                                     # mindspore适配代码
│  ├─models                                        # 模型结构代码
│  ├─optimizer                                     # 优化器代码
│  ├─patchs                                        # patch目录
│  ├─tasks                                         # sft/infer/rl等不同任务的pipeline代码
│  ├─tools                                         # 性能/内存分析工具代码
│  ├─utils                                         # 工具函数/辅助代码目录
│  └─training.py                                   # 训练统一入口
├─evaluate_gen.py                                  # 生成模型评估入口
├─evaluate_vlm.py                                  # 理解类模型评估入口
├─inference_sora.py                                # SORA类模型推理入口
├─inference_vlm.py                                 # VLM类模型推理入口
├─pretrain_omni.py                                 # 全模态模型训练入口
├─pretrain_sora.py                                 # SORA类模型训练入口
├─pretrain_vlm.py                                  # VLM类模型训练入口
├─pretrain_transformers.py                         # Transformers类模型训练入口
├─pyproject.toml                                   # 项目配置和构建文件
├─README.md                                        # 首页文档
├─Third-Party Open Source Software Notice.txt      # 第三方开源软件声明
├─LICENSE                                          # 许可证
├─scripts                                          # 脚本目录
│  ├─install.sh                                    # pytorch环境配置脚本
├─sources                   
│  ├─images                                        # 图片目录
│  └─videos                                        # 视频目录
├─tests                                            # 测试代码目录
│  ├─st                                            # 系统测试用例
│  └─ut                                            # 单元测试用例
├─UserGuide                                        # 用户指南目录
└─verl_plugin                                      # verl适配目录
   ├─verl_npu                                      # verl适配代码
   ├─README.md                                     # verl适配说明文档
   └─setup.py                                      # verl环境安装
```