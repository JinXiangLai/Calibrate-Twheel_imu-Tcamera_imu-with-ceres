## 一. 位置估计准确度分析
### 1. 像素靠近图像边缘且不添加噪声
#### a. 运动非退化时：
		条件数：

		标准差：
#### b. 运动退化时：

		条件数

		标准差
### 2. 像素靠近图像中心且不添加噪声
#### a. 运动非退化时：
		条件数

		标准差

#### b. 运动退化时：
		条件数

		标准差

### 3. 像素靠近图像边缘且添加噪声
#### a. 运动非退化时：
		条件数：

		标准差
#### b. 运动退化时：
		条件数

		标准差

### 4. 像素靠近图像中心且添加噪声
#### a. 运动非退化时：
		条件数：

		标准差

#### b. 运动退化时：
		条件数

		标准差

### 5. 结论
### a. 目的
	希望能够快速给出一个初始位置估计，要求不能太不准。
	
	条件数越小，意味着解受噪声影响较小，解越稳定，但无法保证解的准确度，同时也意味着需要更多的运动;

	标准差越小，意味着解越准，但若设置得太小，可能错过初期虽然标准差大，但是解已经接近真实值的情况，导致收敛时间增加;

	如果能只依赖于条件数判断，尽管std可能较大，但是解精度足够了，那么就能够加快给出一个初始估计的时间，一个方法是减小条件数阈值

		同时增大std阈值


### b. 条件数指标

### c. 标准差指标

### 二、位置先验的可行处理方法：
1. 保留每一个Pw_i的估计及其Cov_i，然后起加入优化函数，但这其实增加了计算量

2. 计算maen Pw作为先验，然后再维护一个累计信息矩阵H，这样就能在滑窗中保持计算量不变，相当于对方法1.的近似

3. 不使用滑窗，也不要求实时，只需要在std足够小时不再更新Pw


### 三、位置退化的识别
1. 互反条件数会很小，意味着系统量测不足，结果存在很大的不确定性，一般体现是Z轴的奇异值很小

2. 验证当前观测使得std变大，并保留该观测时，无法使得后续std变小

3. 当处于退化场景时，Z位置的值本就受噪声影响严重，即便初始值较准，其标准差也很大而不可信，因此处于

	退化场景时只能使用对地高或者其余先验信息，因其退化场景恰好也是对地高高精度的场景

### 四、滑窗优化可行性
1. 根据雅可比剔除Z轴信息量最小的，保留信息量最大的前N个帧

### 五、逆深度估计
1. 如下log为什么第2次计算出的深度值那么不对为 27.9853，那个并不是退化运动啊：
	Default parameters:
			radius=3.000000, depth=50.000000, distX=12.000000
	Current parameters:
			radius=3.000000, depth=50.000000, distX=12.000000
	Pws:
	10 22
	11 11
	50 50

	last Pwc & Rwc in rpy:
	lastPwc: 0 0 0
	lastRwc:  0 -0  0
	Please input 1th Pc1_c2: 0.1 0 0
	Get Pc1_c2: 0.1   0   0
	Please input 1th Rw_c: 0 0 0
	Get rpy: 0 0 0
	current accumulateBaseline: 0.1
	l2s.size: 1
	res: 40

	last Pwc & Rwc in rpy:
	lastPwc: 0.1   0   0
	lastRwc:  0 -0  0
	Please input 2th Pc1_c2: 0.3 0 0
	Get Pc1_c2: 0.3   0   0
	Please input 2th Rw_c: 0 0 0
	Get rpy: 0 0 0
	current accumulateBaseline: 0.4
	l2s.size: 2
	res: 40
	src depth_: 40 estimate depth: 40
	Inverse depth info:
	idepth range: [0.000226378, 0.025, 0.0497736], corresponding depth: [4417.38, 40, 20.091
	Depth info:
	depth s range: [-59.9911, 40, 139.991]

	last Pwc & Rwc in rpy:
	lastPwc: 0.4   0   0
	lastRwc:  0 -0  0
	Please input 3th Pc1_c2: 0.5 0.3 0
	Get Pc1_c2: 0.5 0.3   0
	Please input 3th Rw_c: 1 2 6
	Get rpy: 1 2 6
	current accumulateBaseline: 0.983095
	l2s.size: 3
	res: 27.9853
	src depth_: 40 estimate depth: 27.9853
	Inverse depth info:
	idepth range: [0.0233568, 0.0338189, 0.044281], corresponding depth: [42.8141, 29.5693, 22.583
	Depth info:
	depth s range: [14.9612, 28.1957, 41.4302]


2. 更大异常的输出：
	last Pwc & Rwc in rpy:
	lastPwc:   4.88114  0.323841 -0.108188
	lastRwc: 1 2 0
	Please input 10th Pc1_c2: 0.1 0 0
	Get Pc1_c2: 0.1   0   0
	Please input 10th Rw_c: 10 20 3
	Get rpy: 10 20  3
	current accumulateBaseline: 5
	l2s.size: 10
	res: 9.47925       0
	src depth_: 39.2249 estimate depth: 9.47925
	Inverse depth info:
	idepth range: [0.0291809, 0.0308303, 0.0324797], corresponding depth: [34.269, 32.4356, 30.7884]
	Depth info:
	depth s range: [10.0008, 10.5992, 11.1976]

3. 根据输入打印相关结果：
	last Pwc & Rwc in rpy:
	lastPwc: 0 0 0
	lastRwc:  0 -0  0
	Please input 1th Pc1_c2: 0.5 0 0
	Get Pc1_c2: 0.5   0   0
	Please input 1th Rw_c: 10 20 6
	Get rpy: 10 20  6
	current accumulateBaseline: 0.5
	l2s.size: 1
	Pn1:     0.2 0.21875 Pn2:    -0.15 0.384375 P12:         0.5 6.93889e-18           0 A:0.161963 B:0.196081
	res: 1.21065       0
	
