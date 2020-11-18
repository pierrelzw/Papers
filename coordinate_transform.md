## HadMap车道线投影（雷摄一体自研）

说明：<img src="/Users/lizhiwei/Library/Containers/com.tencent.WeWorkMac/Data/Library/Application Support/WXWork/Data/1688850523220766/Cache/Image/2020-11/企业微信截图_16050725904470.png" alt="企业微信截图_16050725904470" style="zoom:30%;" />

目标：把参考中心坐标系 $ref$ 转换为相机坐标系 $cam$
输入：相机外参 $^{cam}R_{ref}$、$^{cam}T_{ref}$，参考中心坐标系坐标$^{ref}T_{obj}$, 即obj相对ref的translation，

输出：
$$
^{cam}T_{obj} = ^{cam}R_{ref} * ^{ref}T_{obj} + ^{cam}T_{ref}
$$






## HadMap车道线投影（L3）

- object ：从hadmap查询车辆当前位置附近车道线坐标，投影到车辆的相机图像上。
- input：
  - intrinsic: intrinsic_cam60front
  - extrinsic: $^{cam}R_{imu} , ^{imu}R_{V}$
  - Localization : 
    - positon：$ llh_{V}(WGS84)$$
    - orientation:  $ ^{G}R_{V}$

  其中：imu=hppu(L3) ，G=global, V=vehicle_chassis

1. 查询车道新得到 $llh_{lane}$
2. Coordinate Transformation
   - WGS84ToENU_Vehicle
     - Input :  $ llh_{lane}, llh_{V}$
     - Output:  $Coord\_ENU\_Vehicle_{lane}$ : $^{obj}T_{V_{ENU}}$
   - ENU_Vehicle2LocalVehicle : $^{obj}T_{V} =\  ^GR_{V} * ^{obj}T_{V_{ENU}}$
     - 
   - LocalVehicle2Camera
     - $^{cam}R_{V} =\  ^{cam}R_{imu} *\  ^{imu}R_{V}$
     - $^{V}R_{cam} =\  ^{cam}R_V^T$
     - $^{obj}T_{cam}=\ ^{V}R_{cam} *\ ^{obj}T_{V} +\  ^{V}T_{cam} $   
3. $^{obj}T_{cam}$ to 2D image coordinate

