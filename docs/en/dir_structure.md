# Project Directory

The complete directory hierarchy of the project is introduced as follows:

```bash
├─bridge                                           # mbridge Online Weight Conversion
├─checkpoint                                       # Offline Weight Conversion Tool
│  ├─common                                        # Offline Weight Conversion Tool Usage
│  ├─sora_model                                    # Offline Weight Conversion for Multimodal Generation Models
│  └─vlm_model                                     # Offline Weight Conversion for Multimodal Understanding Models
├─ci                                               # Continuous Integration Module
├─docs                                             # Project Documentation
│  ├─en
│  └─zh                                            # Chinese Documentation
│      ├─features                                  # Feature Description
│      ├─mindspore                                 # Migration to MindSpore Backend
│      └─pytorch                                   # Migration to PyTorch Backend
├─examples                                         # Model Execution Scripts and README
│  ├─<model_name>                                  # Scripts for a Specific Model
│  │  ├─xxx.sh                                     # Startup Scripts
│  │  ├─xxx.json/yaml                              # Configuration Files
│  │  └─README.md                                  # Run Instructions
│  ├─diffsynth                                     # Support for DiffSynth-Related Models
│  ├─diffusers                                     # Support for Diffusers-Related Models
│  └─rl                                            # Support for Multimodal Reinforcement Learning Models
├─mindspeed_mm                                     # Core Code Directory
│  ├─configs                                       # Configuration File Reading and Code Processing
│  ├─data                                          # Data Processing
│  ├─mindspore                                     # MindSpore Adaptation
│  ├─models                                        # Model Structure
│  ├─optimizer                                     # Optimizer
│  ├─patches                                        # patch Directory
│  ├─tasks                                         # Pipeline Code for Different Tasks such as SFT/Infer/RL
│  ├─tools                                         # Performance/Memory Profiling Tool Code
│  ├─utils                                         # Utility Functions/Help Information
│  └─training.py                                   # Unified Training Entry Point
├─evaluate_gen.py                                  # Entry Point for Generative Model Evaluation
├─evaluate_vlm.py                                  # Entry Point for Understanding Model Evaluation
├─inference_sora.py                                # Entry Point for SORA Model Inference
├─inference_vlm.py                                 # Entry Point for VLM Model Inference
├─pretrain_omni.py                                 # Entry Point for Full-Modality Model Training
├─pretrain_sora.py                                 # Entry Point for SORA Model Training
├─pretrain_vlm.py                                  # Entry Point for VLM Model Training
├─pretrain_transformers.py                         # Entry Point for Transformers Model Training
├─pyproject.toml                                   # Project Configuration and Build Files
├─README.md                                        # Homepage Documentation
├─Third-Party Open Source Software Notice.txt      # Third-Party Open-Source Software Notices
├─LICENSE                                          # License
├─scripts                                          # Script Directory
│  ├─install.sh                                    # PyTorch Environment Configuration Script
├─sources
│  ├─images                                        # Images
│  └─videos                                        # Videos
├─tests                                            # Testing Code
│  ├─st                                            # System Test Cases
│  └─ut                                            # Unit Test Cases
├─UserGuide                                        # User Guide
└─verl_plugin                                      # verl Adaptation
   ├─verl_npu                                      # verl Adapter Code
   ├─README.md                                     # verl Adaptation Documentation
   └─setup.py                                      # verl Environment Setup
```
