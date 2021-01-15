**基于GoogLenet Inception结构与CNN结合的深度图像去噪模型**

- **Source Code**

  模型通过matlab工具结合caffe框架实现，包含数据增强预处理、卷积层可视化以及模型架构源码，另有tensorflow版本源码可供参考。

- **Architecture**

  <img src=".\Architecture.png" alt="Architecture" style="zoom:40%;" />

  

- **PSNR comparison with baseline models**

  <img src=".\Table.png" alt="psnr" style="zoom:50%;" />

- **Ablation Study with Inception Structure**

  <img src=".\inceptionComp.jpg" alt="inception对比" style="zoom:44%;" />

  <img src=".\inceptionComp2.jpg" alt="inception对比2" style="zoom:44%;" />

- **Ablation Study with BN/Residual Learning**

  <img src=".\AblationStudy.jpg" alt="三种策略对比" style="zoom:50%;" />
