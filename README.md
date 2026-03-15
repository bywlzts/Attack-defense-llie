# A Physically-Grounded Attack and Adaptive Defense Framework for Real-World Low-Light Image Enhancement

- *Tongshun Zhang, Pingping Liu, Yuqing Lei, Zixuan Zhong, Qiuzhan Zhou, Zhiyuan Zha*
- *College of Computer Science and Technology, Jilin University*
- *Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education*
- *College of Communication Engineering, Jilin University*

## 🔥 News
- [03/16/2026] The code is released (Based [RetinexMamba](https://github.com/YhuoyuH/RetinexMamba)).

## 1. Abstract
Limited illumination often causes severe physical noise and detail degradation in images. Existing Low-Light Image Enhancement (LLIE)  methods frequently treat the enhancement process as a blind black-box mapping, overlooking the physical noise transformation during imaging, leading to suboptimal performance. To address this, we propose a novel LLIE approach, conceptually formulated as a physics-based attack and display-adaptive defense paradigm. Specifically, on the attack side, we establish a physics-based Degradation Synthesis (PDS) pipeline. Unlike standard data augmentation, PDS explicitly models Image Signal Processor (ISP) inversion to the RAW domain, injects physically plausible photon and read noise, and re-projects the data to the sRGB domain. This generates high-fidelity training pairs with explicitly parameterized degradation vectors, effectively simulating realistic attacks on clean signals. On the defense side, we construct a dual-layer fortified system. A noise predictor estimates degradation parameters from the input sRGB image. These estimates guide a degradation-aware Mixture of Experts (DA-MoE), which dynamically routes features to experts specialized in handling specific noise intensities. Furthermore, we introduce an Adaptive Metric Defense (AMD) mechanism, dynamically calibrating the feature embedding space based on noise severity, ensuring robust representation learning under severe degradation. Extensive experiments demonstrate that our approach offers significant plug-and-play performance enhancement for existing benchmark LLIE methods, effectively suppressing real-world noise while preserving structural fidelity.

## 2. Motivation
