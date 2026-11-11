# Project Directory

The complete directory hierarchy of the project is introduced as follows:

```bash
├─bridge                                           # mbridge Online Weight Conversion
├─checkpoint                                       # Offline Weight Conversion Tool
│  ├─common                                        # Common Methods of the Offline Weight Conversion Tool
│  ├─sora_model                                    # Offline Weight Conversion for Multimodal Generative Models
│  └─vlm_model                                     # Offline Weight Conversion for Multimodal Understanding Models
├─ci                                               # Continuous Integration Module
├─docs                                             # Project Documentation Directory
│  ├─en
│  └─zh                                            # Chinese Documentation Directory
│      ├─introduction                              # Introduction, Directory Structure, and Supported Models
│      ├─guides                                    # Installation, Practice, Development, Tuning, and Troubleshooting Guides
│      ├─features                                  # Feature Description Documents
│      ├─reference                                 # Configuration and Parameter Description
│      ├─tools                                     # Tool Documentation
│      └─mindspore                                 # MindSpore Backend Documentation
├─examples                                         # Run Scripts and READMEs for All Models
│  ├─<model_name>                                  # Scripts for a Specific Model
│  │  ├─xxx.sh                                     # Startup Script
│  │  ├─xxx.json/yaml                              # Configuration File
│  │  └─README.md                                  # Run Instructions
│  ├─diffsynth                                     # Support for DiffSynth-Related Models
│  ├─diffusers                                     # Support for Diffusers-Related Models
│  └─rl                                            # Support for Multimodal Reinforcement Learning Models
├─mindspeed_mm                                     # Core Code Directory
│  ├─configs                                       # Configuration File Reading and Processing Code
│  ├─data                                          # Data Processing Code
│  ├─models                                        # Model Structure Code
│  ├─optimizer                                     # Optimizer Code
│  ├─patchs                                        # Patch Directory
│  ├─tasks                                         # Pipeline Code for Different Tasks Such as SFT/Infer/RL
│  ├─tools                                         # Performance/Memory Profiling Tool Code
│  ├─utils                                         # Utility Functions/Helper Code Directory
│  └─training.py                                   # Unified Training Entry Point
├─evaluate_gen.py                                  # Generative Model Evaluation Entry Point
├─evaluate_vlm.py                                  # Understanding Model Evaluation Entry Point
├─inference_sora.py                                # SORA Model Inference Entry Point
├─inference_vlm.py                                 # VLM Model Inference Entry Point
├─pretrain_omni.py                                 # Full-Modality Model Training Entry Point
├─pretrain_sora.py                                 # SORA Model Training Entry Point
├─pretrain_vlm.py                                  # VLM Model Training Entry Point
├─pretrain_transformers.py                         # Transformers Model Training Entry Point
├─pyproject.toml                                   # Project Configuration and Build File
├─README.md                                        # Homepage Documentation
├─Third-Party Open Source Software Notice.txt      # Third-Party Open Source Software Notice
├─LICENSE                                          # License
├─scripts                                          # Script Directory
│  ├─install.sh                                    # PyTorch Environment Configuration Script
├─sources                   
│  ├─images                                        # Images Directory
│  └─videos                                        # Videos Directory
├─tests                                            # Test Code Directory
│  ├─st                                            # System Test Cases
│  └─ut                                            # Unit Test Cases
└─verl_plugin                                      # verl Adaptation Directory
   ├─verl_npu                                      # verl Adaptation Code
   ├─README.md                                     # verl Adaptation Documentation
   └─setup.py                                      # verl Environment Installation
```
