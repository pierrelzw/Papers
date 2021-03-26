# DataSet Collection

[TOC]

## 0. wikipedia: Public Transportation Datasets

https://www.polymtl.ca/wikitransport/index.php?title=Public_Transportation_Datasets

## 1. 车辆检测(监控)

https://github.com/gustavovelascoh/traffic-surveillance-dataset

### 雨雪交通监控

https://www.kaggle.com/aalborguniversity/aau-rainsnow

该数据集由22个视频组成，每个视频约5分钟。使用RGB彩色相机和红外热像仪捕获视频。因此，数据包括超过130,000个RGB热图像对。

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadt7yyfxj31b60ay77n.jpg" alt="image-20210202182039470" style="zoom:40%;" />



###  UA-DETRAC 中国北京和天津监控摄像头检测数据

[http://detrac-db.rit.albany.edu/](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fdetrac-db.rit.albany.edu%2F)

UA-DETRAC是一个具有挑战性的现实世界多目标检测和多目标跟踪基准。数据集由 Cannon EOS 550D摄像头在中国北京和天津24个不同地点拍摄的10个小时的视频组成。视频以每秒25帧的速度录制，分辨率为960540像素。在UA-DETRAC数据集中，有超过14万帧和8250辆车被人工标注，总共标记了121万物体的边界盒。我们还对目标检测和多目标跟踪方面的最新方法进行基准测试，以及本网站中详细介绍的评估指标。

车辆分为四类，即轿车、公共汽车、厢式货车和其他车辆。

天气情况分为四类，即多云、夜间、晴天和雨天。

标注的车辆的尺度定义为其像素面积的平方根。将车辆分为三种规模:小型(0-50像素)、中型(50-150像素)和大型(大于150像素)。遮挡比我们使用车辆包围框被遮挡的比例来定义遮挡的程度。

遮挡程度分为三类: 无遮挡、部分遮挡和重遮挡。具体来说，定义了部分遮挡(如果车辆遮挡率在1%-50%之间)和重遮挡(如果遮挡率大于50%)。

截尾率表示车辆部件在帧外的程度，用于训练样本的选择。

![img](https://img-blog.csdnimg.cn/20201015135433337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70)

效果图：

![img](https://img-blog.csdnimg.cn/20201015075600870.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70)



### 北京大学POSS数据集

http://www.poss.pku.edu.cn/download.html

### VdiAw（1000张，监控、自动驾驶视角下的数据，包含bbox标注）

https://data.mendeley.com/datasets/766ygrbt8y/1

覆盖多样化的场景（urban, highway and freeway) ，天气状况（fog, snow, rain and sandstorms.）

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadt9dmokj31gs0u0131.jpg" alt="image-20210202193421721" style="zoom:30%;" />

### IITM-HeTra : Dataset for Vehicle Detection in Heterogeneous Traffic Scenarios

https://www.kaggle.com/deepak242424/iitmhetra

车辆检测L：2400张印度chennai 交通场景数据

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadtbodcaj30zw0r2gr1.jpg" alt="image-20210202194553497" style="zoom:40%;" />

### CADP 车辆事故检测和预测数据集

https://ankitshah009.github.io/accident_forecasting_traffic_camera

https://docs.google.com/document/d/12F7l4yxNzzUAISZufEd9WFhQKSefVVo_QsPdTsWxZh8/edit

### 2014 The Ko-PER Intersection Dataset

https://www.uni-ulm.de/in/mrm/forschung/datensaetze.html

这个数据集包含：激光点云数据，去畸变图像数据，车辆trajectories，roaduuse？

- raw laserscanner measurements
- undistorted monochrome camera images
- highly accurate reference trajectories of cars
- labeled roas users

<img src="../perception.docs/DataSet_Collection.assets/image-20210204153709451.png" alt="image-20210204153709451" style="zoom:40%;" />

###  2013 GRAM Road-Traffic Monitoring

http://agamenon.tsc.uah.es/Personales/rlopez/data/rtm/

- task : multi-vehicle tracking in real-time
- 标注类别：car, truck, van, and big-truck.

<img src="../perception.docs/DataSet_Collection.assets/image-20210204154740176.png" alt="image-20210204154740176" style="zoom:50%;" />

###  Stanford Cars Dataset

[http://ai.stanford.edu/~jkrause/cars/car_dataset.html](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fai.stanford.edu%2F~jkrause%2Fcars%2Fcar_dataset.html)

Cars数据集包含196类汽车的16,185张图像，bbox标注。将数据分成8144张训练图像和8041张测试图像。按制造商、型号、年份划分，例如2012年特斯拉Model S或2012年宝马M3来标记。

 **3D Object Representations for Fine-Grained Categorization**
    Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
    *4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013* **(3dRR-13)**. Sydney, Australia. Dec. 8, 2013.
    [[pdf\]](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fai.stanford.edu%2F~jkrause%2Fpapers%2F3drr13.pdf)  [[BibTex\]](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fai.stanford.edu%2F~jkrause%2Fpapers%2F3drr13.bib)  [[slides\]](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fai.stanford.edu%2F~jkrause%2Fpapers%2F3drr_talk.pdf)

<img src="https://img-blog.csdnimg.cn/20201015140718280.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:50%;" />

## 2. 车辆检测&分割(自动驾驶)

### [BDD100K ](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Fbdd-data.berkeley.edu%2Fwad-2020.html)自动驾驶数据集

[https://bdd-data.berkeley.edu/](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Fbdd-data.berkeley.edu%2F)

视频数据： 超过1,100小时的100000个高清视频序列在一天中许多不同的时间，天气条件，和驾驶场景驾驶经验。视频序列还包括GPS位置、IMU数据和时间戳。

道路目标检测：2D边框框注释了100,000张图片，用于公交、交通灯、交通标志、人、自行车、卡车、摩托车、小汽车、火车和骑手。

实例分割：超过10,000张具有像素级和丰富实例级注释的不同图像。

引擎区域：从10万张图片中学习复杂的可驾驶决策。

车道标记：10万张图片上多类型的车道标注，用于引导驾驶。

如图：

![img](https://img-blog.csdnimg.cn/20201015075842612.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20201015140026286.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70)

### [自动驾驶汽车的语义分割](https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge#dataB.tar.gz)

–作为Lyft Udacity Challenge的一部分创建，此数据集包含5,000张图像和相应的语义分割标签。

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadtb7wi1j318g0l4jvg.jpg" alt="image-20210202191330509" style="zoom:30%;" />

### [TME高速公路数据集](http://cmp.felk.cvut.cz/data/motorway/)

–由28个视频片段组成，总计27分钟的视频，该数据集包括30,000多个带有车辆注释的帧。

<img src="http://cmp.felk.cvut.cz/data/motorway/images/facingSunLS.jpg" alt="Facing Sun example" style="zoom:50%;" />





### [LISA红绿灯数据集](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset)

–尽管此数据集不专注于车辆，但它仍然是用于训练自动车辆算法的非常有用的图像数据集。LISA交通信号灯数据集包括夜间和白天的视频，总计43,0007帧，其中包括带注释的113,888个交通信号灯。该数据集的重点是交通信号灯。但是，几乎所有车架中都装有交通信号灯和车辆。

### MIT DriveSeg Dataset

[https://agelab.mit.edu/driveseg](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Fagelab.mit.edu%2Fdriveseg)

到目前为止，提供给研究社区的自动驾驶数据主要由静态的、单一的图像组成，这些图像可以通过使用边界框来识别和跟踪道路内和周围的常见物体，比如自行车、行人或交通灯。相比之下，DriveSeg包含了更精确的、像素级的这些常见道路物体的表示，但通过连续视频驾驶场景的镜头。这种类型的全场景分割可以特别有助于识别更多的无定形物体，如道路建设和植被，它们并不总是有这样明确和统一的形状。数据集由两部分组成：

DriveSeg(手动)：一种面向前帧逐帧像素级语义标记数据集，该数据集是从一辆在连续日光下通过拥挤的城市街道行驶的移动车辆中捕获的。

视频数据： 2分47秒(5000帧)1080P (1920x1080) 30帧/秒

类定义(12)：车辆、行人、道路、人行道、自行车、摩托车、建筑、地形(水平植被)、植被(垂直植被)、杆子、交通灯和交通标志

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadtasdn0j315m0nctdb.jpg" alt="image-20210202191735283" style="zoom:30%;" />

### KITT

· 数据集链接：[http://www.cvlibs.net/datasets/kitti/](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttp%3A%2F%2Fwww.cvlibs.net%2Fdatasets%2Fkitti%2F)

· 论文链接： [http://www.webmail.cvlibs.net/publications/Geiger2012CVPR.pdf](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttp%3A%2F%2Fwww.webmail.cvlibs.net%2Fpublications%2FGeiger2012CVPR.pdf)

精确的地面真相由Velodyne激光扫描仪和GPS定位系统提供。我们的数据集是通过在中型城市卡尔斯鲁厄(Karlsruhe)、乡村地区和高速公路上行驶来获取的。每张图像可看到多达15辆汽车和30个行人。除了以原始格式提供所有数据外，我们还为每个任务提取基准。对于我们的每一个基准，我们也提供了一个评估指标和这个评估网站。初步实验表明，在现有基准中排名靠前的方法，如Middlebury方法，在脱离实验室进入现实世界后，表现低于平均水平。我们的目标是减少这种偏见，并通过向社会提供具有新困难的现实基准来补充现有基准。

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadt8gdayj30r00a4whg.jpg" alt="image-20210202192239084" style="zoom:50%;" />

### CityScapes

· 数据集链接：[https://www.cityscapes-dataset.com/](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Fwww.cityscapes-dataset.com%2F)

· 论文链接：[https://arxiv.org/pdf/1604.01685.pdf](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Farxiv.org%2Fpdf%2F1604.01685.pdf)

提供了一个新的大规模数据集，其中包含了50个不同城市的街道场景中记录的不同的立体视频序列，有5000帧的高质量像素级注释，还有更大的一组2万帧的弱注释。因此，该数据集比以前类似的尝试要大一个数量级。有关注释类的详细资料及注释示例可在此网页查阅。Cityscapes数据集旨在评估用于语义城市场景理解的主要任务的视觉算法的性能:像素级、实例级和全光学语义标记;支持旨在开发大量(弱)注释数据的研究，例如用于训练深度神经网络。

 <img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadt9tsqsj30r80aamyz.jpg" alt="image-20210202192218695" style="zoom:50%;" />

### [Comma.ai 's Driving Dataset](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttp%3A%2F%2Fcomma.ai%2F)

· 数据集链接：[https://github.com/commaai/research](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Fgithub.com%2Fcommaai%2Fresearch)

· 论文链接：[https://arxiv.org/pdf/1608.01230.pdf](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Farxiv.org%2Fpdf%2F1608.01230.pdf)

目的是低成本的自动驾驶方案，目前是通过手机改装来做自动驾驶，开源的数据包含7小时15分钟分为11段的公路行驶的行车记录仪视频数据，每帧像素为160x320。主要应用方向：图像识别；

### Udacity 's Driving Dataset

· 数据集链接：[https://github.com/udacity/self-driving-car/tree/master/datasets](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Fgithub.com%2Fudacity%2Fself-driving-car%2Ftree%2Fmaster%2Fdatasets)

· 论文链接：未找到

Udacity的自动驾驶数据集，使用Point Grey研究型摄像机拍摄的1920x1200分辨率的图片，采集到的数据分为两个数据集：第一个包括在白天情况下在加利福尼亚州山景城和邻近城市采集的数据，数据集包含9,423帧中超过65,000个标注对象，标注方式结合了机器和人工。标签为：汽车、卡车、行人；第二个数据集与前者大体上相似，除了增加交通信号灯的标注内容，数据集数量上也增加到15,000帧，标注方式完全采用人工。数据集内容除了有车辆拍摄的图像，还包含车辆本身的属性和参数信息，例如经纬度、制动器、油门、转向度、转速。主要应用方向：目标检测，自动驾驶；

### D²-City 大规模行车视频数据集

· 数据集链接：[https://outreach.didichuxing.com/d2city/](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F159684396%2Fedit%23DatasetDescription)

#### 背景

D²-City 是一个大规模行车视频数据集，提供了超过一万段行车记录仪记录的前视视频数据。所有视频均以高清（720P）或超高清（1080P）分辨率录制。我们为其中的约一千段视频提供了包括目标框位置、目标类别和追踪ID信息的逐帧标注，涵盖了共12类行车和道路相关的目标类别。我们为一部分其余的视频提供了关键帧的框标注。

和现有类似数据集相比，D²-City 的数据采集自中国多个城市，涵盖了不同的天气、道路、交通状况，尤其是极复杂和多样性的交通场景。我们希望通过该数据集能够鼓励和帮助自动驾驶相关领域研究取得新进展。

#### 数据集介绍

D²-City 数据集采集自运行在中国五个城市的滴滴运营车辆。所提供的原始数据均存储为帧率25fps、时长30秒的短视频。后续我们将会提供对该数据集的训练、验证和测试集的划分与统计。

我们为其中约一千段视频提供了12类目标的边界框和追踪ID标注信息，对其他的视频，我们提供关键帧的框标注。类别信息详见下表。

![img](https://img-blog.csdnimg.cn/20201015142805716.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70)

#### 评估任务

基于本数据集，我们将提供一项评估任务（和BDD合作）作为NeurIPS 2019 ML4AD挑战赛的赛事。任务和评估的详情请参见竞赛网站相关页面。

赛事：D²-City & BDD100K 目标检测迁移学习挑战赛 在目标检测迁移学习挑战赛中，参赛者需要利用采集自美国的BDD100K数据，训练目标检测模型用于采集自中国的D²-City数据。数据集中可能包含稀有或有挑战性的状况下采集的数据，如光线不足、雨雾天气、道路拥堵等，参赛者需要提供在各状况下准确的目标检测结果。

### ApolloScape

· 数据集链接：[http://apolloscape.auto/inpainting.html](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fapolloscape.auto%2Finpainting.html)

关于ApolloScape数据集
轨迹数据集，三维感知激光雷达目标检测和跟踪数据集，包括约100K图像帧，80k激光雷达点云和1000km城市交通轨迹。数据集由不同的条件和交通密度，其中包括许多具有挑战性的场景，车辆，自行车，和行人之间移动。

数据集包括以下几个方面的研究：

![img](https://img-blog.csdnimg.cn/20201015143443740.png)

Scene Parsing

<img src="https://img-blog.csdnimg.cn/20201015143512433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:80%;" />

3D Car Instance

<img src="https://img-blog.csdnimg.cn/20201015143536671.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:70%;" />

Lane Segmentation

<img src="https://img-blog.csdnimg.cn/20201015143557798.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:80%;" />

Self Localization

<img src="https://img-blog.csdnimg.cn/20201015143621844.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:43%;" />

Trajectory

<img src="https://img-blog.csdnimg.cn/20201015143643850.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:80%;" />

Stereo

<img src="https://img-blog.csdnimg.cn/20201015143713181.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:50%;" />

Inpainting

![img](https://img-blog.csdnimg.cn/20201015143734233.png)

### Oxford RobotCar：

· 数据集链接：[https://www.cityscapes-dataset.com/](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Fwww.cityscapes-dataset.com%2F)

· 论文链接： [http://robotcar-dataset.robots.ox.ac.uk/images/robotcar_ijrr.pdf](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttp%3A%2F%2Frobotcar-dataset.robots.ox.ac.uk%2Fimages%2Frobotcar_ijrr.pdf)

牛津大学的项目，数据是对牛津的一部分连续的道路进行了上百次数据采集，收集到了多种天气、行人和交通情况下的数据，也有建筑和道路施工时的数据，长达1000小时以上（论文名称就写着 '1year, 1000km'）。要注意的是，所下载的Oxford RobotCar Dataset的数据不包含label文件。因此对数据进行使用前，具体需要查看论文内容。主要应用方向：自动驾驶视觉场景分析；

### nuScenes

数据集链接：[https://www.nuscenes.org/](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Fwww.nuscenes.org%2F)

nuScenes数据集是一个具有3d对象标注的大规模自主驾驶数据集。它特点:

完整的传感器套件(1 x激光雷达、5 x雷达、6 x相机,IMU, GPS)

1000 scenes of 20s each、1400000相机图像、390000激光雷达扫描

两个不同的城市:波士顿和新加坡

左派和右手交通详细的地图信息

1.4M 3D 边界盒手工注释等，23个对象类

属性可见性、活动和姿势

新: 1.1B 激光雷达点手工注释为32类

新: 探索nuScenes在SiaSearch免费使用非商业用途

### [waymo]( https://waymo.com/open/data/)







## 3. 车辆重识别

#### OpenData V11.0-车辆重识别数据集 VRID

[http://www.openits.cn/opendata4/748.jhtml](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fwww.openits.cn%2Fopendata4%2F748.jhtml)

**数据集说明：**

开放的车辆重识别的数据来自某城市卡口车辆图像，由326个高清摄像头拍摄，时间覆盖日间14天，分辨率从400×424到990×1134不等。数据集中包含最常见的10种车辆款式，共10000张图像，如表1所列。为了模拟同款车辆对车辆重识别的影响，每个车辆款式里各有100个不同的车辆ID，即100个不同的车辆。在同一车辆款式里的100个车辆ID，它们的外观近乎相同，差异大部分只在于车窗部分的个性化标识，如年检标志等。此外，每个车辆ID包含有10张图像，这10张图像拍摄于不同的道路卡口，光照、尺度以及姿态均不尽相同，相应的同一车辆也可能会具有不同的外观。

车辆重识别数据集的车辆字段属性如表2所示，其中车辆品牌表示车辆品牌信息，车牌号码用于数据库里同一车辆的关联，车窗位置代表图像里的车窗所在区域的坐标，车身颜色表示的是图像里的车辆颜色信息。这些信息使得数据库不仅能用于车辆重识别研究，也可用于车辆品牌精细识别，车辆颜色识别等研究。

数据集里10种车辆款式

<img src="https://img-blog.csdnimg.cn/20201015140852590.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQ1NDY4Mjg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom:60%;" />

数据库属性示意表

<img src="https://img-blog.csdnimg.cn/20201015140859535.png" alt="img" style="zoom:70%;" />

## 4. 车辆分类

### [尼泊尔车辆](https://github.com/sdevkota007/vehicles-nepal-dataset)

由加德满都街头拍摄的总共30部交通视频组成，该数据集包含从这些视频中裁剪的4,800辆车辆的图像。在4800张图像中，有1811张为两轮车，而2989张为四轮车。

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadt8xqx7j304v05m0sr.jpg" alt="(989)" style="zoom:50%;" />

### GTI车辆图像数据库

 –此数据集包括3,425个道路上车辆的后角图像，以及3,900个没有车辆的道路图像。

https://www.gti.ssr.upm.es/data/Vehicle_database.html

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnadtaaac2j30ys07g3zz.jpg" alt="image-20210202182226889" style="zoom:50%;" />

### 综合汽车（CompCars）数据集

[http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fmmlab.ie.cuhk.edu.hk%2Fdatasets%2Fcomp_cars%2Findex.html)

该数据集在 CVPR 2015论文中给出，Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. **A Large-Scale Car Dataset for Fine-Grained Categorization and Verification**, *In Computer Vision and Pattern Recognition (CVPR), 2015*. [PDF](https://www.oschina.net/action/GoToLink?url=http%3A%2F%2Fmmlab.ie.cuhk.edu.hk%2Fdatasets%2Fcomp_cars%2FCompCars.pdf)。

综合汽车(CompCars)数据集包含来自两种场景的数据，包括来自web-nature和监视-nature的图像。

web-nature数据包含163辆汽车和1,716个汽车模型。总共有136,726张图像捕捉整个汽车，27,618张图像捕捉汽车部件。完整的汽车图像被标记为边界框和视点。每个车型都有五个属性，包括最大速度、排水量、车门数量、座椅数量和车型。

监视-自然数据包含了5万张前视图捕捉到的汽车图像。

该数据集已经为以下计算机视觉任务做好了准备:细粒度分类、属性预测、汽车模型验证。

本文中介绍的这些任务的训练/测试子集都包含在数据集中。研究人员也欢迎使用它来完成其他任务，如图像排序、多任务学习和3D重建。

<img src="https://img-blog.csdnimg.cn/img_convert/ce3cf95dab7b83c74fdd2af0c1f55a85.png" alt="img" style="zoom:50%;" />



### BIT车辆数据集 (服务器暂时不可用)

来自北京智能信息技术实验室的数据集包含9,850幅车辆图像。这些图像按车辆类型分为以下六类：公共汽车，小型客车，小型货车，轿车，SUV和卡车。

http://iitlab.bit.edu.cn/mcislab/vehicledb/



### N-CARS数据集

[https://www.prophesee.ai/dataset-n-cars/](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Fwww.prophesee.ai%2Fdataset-n-cars%2F)

N-CARS数据集是一个用于汽车**分类**的大型基于事件的真实世界数据集。

它由12,336个汽车样本和11,693个非汽车样本(背景)组成。这些数据是通过安装在一辆汽车挡风玻璃后的ATIS摄像机记录下来的。这些数据是从不同的驾驶过程中提取的。数据集被分割为7940个car和7482个背景训练样本，4396个 car 和4211个背景测试样本。每个示例持续100毫秒。

## 5. 交通标志数据集

1）[KUL Belgium Traffic Sign Dataset](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttp%3A%2F%2Fwww.vision.ee.ethz.ch%2F~timofter%2Ftraffic_signs%2F)，比利时的一个交通标志数据集。

2）[German Traffic Sign](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttp%3A%2F%2Fbenchmark.ini.rub.de%2F%3Fsection%3Dgtsrb%26subsection%3Ddataset)，德国交通标注数据集。

3）[STSD](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Fwww.cvl.isy.liu.se%2Fresearch%2Fdatasets%2Ftraffic-signs-dataset%2F)，超过20 000张带有20％标签的图像，包含3488个交通标志。

4）[LISA](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttp%3A%2F%2Fcvrr.ucsd.edu%2FLISA%2Flisa-traffic-sign-dataset.html)，超过6610帧上的7855条标注。

5）[Tsinghua-Tencent 100K](https://www.oschina.net/action/GoToLink?url=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Fcg.cs.tsinghua.edu.cn%2Ftraffic-sign%2F) ，腾讯和清华合作的数据集，100000张图片，包含30000个交通标志实例。



## 6. 行人检测

行人检测( Pedestrian Detection) 是计算机视觉领域内应用比较广泛和比较热门的算法，一般会与行人跟踪，行人重识别等技术进行结合，来对区域内的行人进行检测识别跟踪，广泛应用于安防，零售等领域。由于行人的外观易受穿着、尺度、遮挡、姿态和视角等影响，行人检测也具有一定的挑战性。本文我们收集了行人检测常用的一些数据集，方便大家来学习和研究行人检测算法。所有数据集均为网上公开数据集，文末附有下载链接。



### **1.[MIT-CBCL Pedestrian Database（MIT行人数据库）](https://link.zhihu.com/?target=http%3A//cbcl.mit.edu/software-datasets/PedestrianData.html)**

该数据库为较早公开的行人数据库，共924张行人图片（ppm格式，宽高为64x128），肩到脚的距离约80象素。该数据库只含正面和背面两个视角，无负样本，未区分训练集和测试集。Dalal等采用“HOG+SVM”，在该数据库上的检测准确率接近100%。





### **2.[USC Pedestrian Detection Test Set（USC行人数据库）](https://link.zhihu.com/?target=http%3A//iris.usc.edu/Vision-Users/OldUsers/bowu/DatasetWebpage/dataset.html)**

<img src="https://pic3.zhimg.com/80/v2-8e95688eb763a32de754a22d2ca0239e_1440w.jpg" alt="img" style="zoom:30%;" />

该数据库包含三组数据集（USC-A、USC-B和USC-C），以XML格式提供标注信息。USC-A[Wu, 2005]的图片来自于网络，共205张图片，313个站立的行人，行人间不存在相互遮挡，拍摄角度为正面或者背面；USC-B的图片主要来自于CAVIAR视频库，包括各种视角的行人，行人之间有的相互遮挡，共54张图片，271个行人；USC-C有100张图片来自网络的图片，232个行人（多角度），行人之间无相互遮挡。

### **3.[Caltech Pedestrian Detection Benchmark（Caltech行人数据库）](https://link.zhihu.com/?target=http%3A//www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)**

<img src="https://pic4.zhimg.com/80/v2-0e584569580ac9029ad99a5e7bb935cb_1440w.jpg" alt="img" style="zoom:50%;" />

该数据库是目前规模较大的行人数据库，采用车载摄像头拍摄，约10个小时左右，视频的分辨率为640x480，30帧/秒。标注了约250,000帧（约137分钟），350000个矩形框，2300个行人，另外还对矩形框之间的时间对应关系及其遮挡的情况进行标注。

### **4.[Daimler Pedestrian Detection Benchmark (戴姆勒行人检测标准数据库)](https://link.zhihu.com/?target=http%3A//www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Pedestrian_Segmentatio/daimler_pedestrian_segmentatio.html)**

<img src="https://pic2.zhimg.com/80/v2-182c2b2927b1868fcbce38bc88454c15_1440w.jpg" alt="img" style="zoom:40%;" />

该数据库采用车载摄像机获取，分为检测和分类两个数据集。检测数据集的训练样本集有正样本大小为18×36和48×96的图片各15560（3915×4）张，行人的最小高度为72个象素；负样本6744张（大小为640×480或360×288）。测试集为一段27分钟左右的视频（分辨率为640×480），共21790张图片，包含56492个行人。分类数据库有三个训练集和两个测试集，每个数据集有4800张行人图片，5000张非行人图片，大小均为18×36，另外还有3个辅助的非行人图片集，各1200张图片。



### **5.[DukeMTMC-reID](https://link.zhihu.com/?target=https%3A//github.com/layumi/DukeMTMC-reID_evaluation)**

<img src="https://pic2.zhimg.com/80/v2-dfb5e51195cd0cd1be1b41ca0fd272d5_1440w.jpg" alt="img" style="zoom:13%;" />

DukeMTMC-reID 为 DukeMTMC数据集的行人重识别子集。原始数据集包含了85分钟的高分辨率视频，采集自8个不同的摄像头。并且提供了人工标注的bounding box.

### **6.[INRIA Person Dataset（INRIA行人数据库）](https://link.zhihu.com/?target=http%3A//pascal.inrialpes.fr/data/human/)**

<img src="https://pic1.zhimg.com/80/v2-2774950c6e0158ebd8ea227668a2c3e0_1440w.jpg" alt="img" style="zoom:50%;" />

该数据库是目前使用最多的静态行人检测数据库，提供原始图片及相应的标注文件。训练集有正样本614张（包含2416个行人），负样本1218张；测试集有正样本288张（包含1126个行人），负样本453张。图片中人体大部分为站立姿势且高度大于100个象素，部分标注可能不正确。图片主要来源于GRAZ-01、个人照片及google，因此图片的清晰度较高。在XP操作系统下部分训练或者测试图片无法看清楚，但可用OpenCV正常读取和显示。



### **7.[BIWI Walking Pedestrians dataset](https://link.zhihu.com/?target=http%3A//www.vision.ee.ethz.ch/en/datasets/)**

![img](https://pic2.zhimg.com/80/v2-7b2ea71beb22e570cc0cd3ecc57793d9_1440w.jpg)

该数据集中所有图片均是采用鸟瞰视角，对繁忙场景下散步行走的路人进行的记录。

### **8.[Central Pedestrian Crossing Sequences](https://link.zhihu.com/?target=http%3A//www.vision.ee.ethz.ch/en/datasets/)**

![img](https://pic2.zhimg.com/80/v2-7c74661be832230339804fd3e76bbdc5_1440w.jpg)

这是在ICCV'07论文中使用的三个行人穿越序列。每个序列都带有跟踪对象的地面实况框图注释和相机校准。每四帧进行一次标定。

### **9.[Dataset used in our ICCV '07 paper Depth and Appearance for Mobile Scene Analysis](https://link.zhihu.com/?target=https%3A//data.vision.ee.ethz.ch/cvl/aess/iccv2007/)**

![img](https://pic2.zhimg.com/80/v2-9fceab9c0a768bfdf7401c22ede59531_1440w.jpg)

该数据集出自于Eth Zurich（苏黎世联邦理工学院）。当中记录了12298个行人的样本。

### **10.[Human detection and tracking using RGB-D camera](https://link.zhihu.com/?target=http%3A//www.cv.fudan.edu.cn/humandetection.htm)**

![img](https://pic4.zhimg.com/80/v2-7b4b9a055dd5c4caa7247cd77571e613_1440w.jpg)

该数据集出自复旦大学计算机视觉实验室，当中的所有图片均采自于一家服装店中。





### **11.[CUHK Occlusion Dataset](https://link.zhihu.com/?target=http%3A//mmlab.%3C/b%3Eie.cuhk.edu.hk/datasets/cuhk_occlusion/index.html)**

![img](https://pic4.zhimg.com/80/v2-0dbc5091544db0f9ffdeee4b3c0d7b5b_1440w.jpg)

该数据集出自于香港中文大学，可应用于行为分析和行人检测。包含了1063张行人图片。





### **12.[CUHK Person Re-identification Datasets](https://link.zhihu.com/?target=http%3A//www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)**

![img](https://pic2.zhimg.com/80/v2-f4d8a7faa3d6d3ca7e15e34fd26c5f29_1440w.jpg)

这也是出自于香港中文大学的数据集，使用了两个（不相交的）视角，对971个行人进行了记录。每个行人在每个视角中均进行了两次取样。

### **13.[CUHK Square Dataset](https://link.zhihu.com/?target=http%3A//mmlab.ie.cuhk.edu.hk/datasets/cuhk_square/index.html)**

<img src="https://pic3.zhimg.com/80/v2-4158aa6037a8c882d2f848c7ec434916_1440w.jpg" alt="img" style="zoom:30%;" />

港中大的广场数据集。包含了一段长达60分钟的交通视频序列。（大小为720×576）

**行人检测数据集打包下载链接：**[https://pan.baidu.com/s/1o8aanoQ](https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1o8aanoQ)

**密码：**xkka

