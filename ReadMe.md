**基于GoogLenet Inception结构与CNN结合的深度图像去噪模型**

- **Source Code**

  模型通过matlab工具结合caffe框架实现，包含数据增强预处理、卷积层可视化以及模型架构源码，另有tensorflow版本源码可供参考。

- **Architecture**

  <img src="C:\Users\wei\Desktop\Presentation\资讯科技研究方法\搜狗截图20201118125325.png" alt="搜狗截图20201118125325" style="zoom:50%;" />

  

- **PSNR comparison with baseline models**

  ![搜狗截图20210116023748](C:\Users\wei\Desktop\搜狗截图20210116023748.png)

- **Ablation Study with Inception Structure**

  <img src="C:\Users\wei\Desktop\Presentation\资讯科技研究方法\inception对比.jpg" alt="inception对比" style="zoom:44%;" />

  <img src="C:\Users\wei\Desktop\Presentation\资讯科技研究方法\inception对比2.jpg" alt="inception对比2" style="zoom:44%;" />

- **Ablation Study with BN/Residual Learning**

  <img src="C:\Users\wei\Desktop\Presentation\资讯科技研究方法\三种策略对比.jpg" alt="三种策略对比" style="zoom:50%;" />