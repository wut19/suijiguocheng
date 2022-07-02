import numpy as np
import GPy
import matplotlib.pyplot as plt


x2 = np.arange(400,410,1).reshape(10,1)
x1 = np.arange(0,400,1).reshape(400,1)
x3 = np.arange(395,410,1).reshape(15,1)

# 读取数据集
data1 = np.loadtxt(open("data1(NVIDIA).csv","rb"),delimiter="\",\"",skiprows=1,usecols=[4])
# data2 = np.loadtxt(open("data2(TESLA).csv","rb"),delimiter="\",\"",skiprows=1,usecols=[4])
# data3 = np.loadtxt(open("data3(MSFT).csv","rb"),delimiter="\",\"",skiprows=1,usecols=[4])
# data4 = np.loadtxt(open("data4(AAPL).csv","rb"),delimiter="\",\"",skiprows=1,usecols=[4])
# data5 = np.loadtxt(open("data5(AMDI).csv","rb"),delimiter="\",\"",skiprows=1,usecols=[4])

#print(data1.shape,data2.shape,data3.shape,data4.shape,data5.shape)

# 分割数据集，获取训练集和测试集
train_data1 = data1[10:410]
test_data1 = data1[0:10]
train_data1 = train_data1[::-1].reshape(400,1)
test_data1 = test_data1[::-1].reshape(10,1)

# train_data2 = data2[10:410]
# test_data2 = data2[0:10]
# train_data3 = data3[10:410]
# test_data3 = data3[0:10]
# train_data4 = data4[10:410]
# test_data4 = data4[0:10]
# train_data5 = data5[10:410]
# test_data5 = data5[0:10]

# 数据处理，将测试数据均值设置为0，并将方差归一
train_data1_mean = train_data1.mean()
train_data1 = train_data1 - train_data1_mean
train_data1_sigma = np.sqrt(np.sum(np.square(train_data1))/400)
train_data1 = train_data1/train_data1_sigma
test_data1 = test_data1 - train_data1_mean
test_data1 = test_data1/train_data1_sigma

# 创建回归模型
# SE核
kernel1_SE = GPy.kern.RBF(input_dim=1, variance=2)
model1_1 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_SE, noise_var=1) # SE核模型
# EX(Exponential)核
kernel1_EX = GPy.kern.src.stationary.Exponential(input_dim=1)
model1_2 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_EX)  # EQ核模型
# Matern32核
kernel1_MT = GPy.kern.src.stationary.Matern32(input_dim=1)
model1_3 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_MT)  # MT核模型
# RQ(RatQuad) 核
kernel1_RQ = GPy.kern.src.stationary.RatQuad(input_dim=1)
model1_4 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_RQ)  # RQ核模型


# 优化模型
model1_1.optimize(messages=True)
model1_2.optimize(messages=True)
model1_3.optimize(messages=True)
model1_4.optimize(messages=True)

print(model1_1)
print(model1_2)
print(model1_3)
print(model1_4)
# print(model1_RC1)
# print(model1_RC2)
# print(model1_RC3)
# print(model1_RC4)
# print(model1_RC5)
# print(model1_RC6)

# 预测
predict_mean1_1, predict_var1_1 = model1_1.predict(x2)
predict_mean1_2, predict_var1_2 = model1_2.predict(x2)
predict_mean1_3, predict_var1_3 = model1_3.predict(x2)
predict_mean1_4, predict_var1_4 = model1_4.predict(x2)

# predict_mean1_1 = predict_mean1_1 + train_data1_mean

# 数据展示，展示原始股票波动，便于和预测结果进行对比
plt.plot(x2,test_data1)
plt.plot(x1,train_data1)
# y = (data1[0:15] - train_data1_mean)/train_data1_sigma
# plt.plot(x3,y[::-1],'r')

# 画出结果
# SE核
model1_1.plot(plot_limits=(395,410))
y = (data1[0:15] - train_data1_mean)/train_data1_sigma
plt.plot(x3,y[::-1],'r')
# EX核
model1_2.plot(plot_limits=(395,410))
y = (data1[0:15] - train_data1_mean)/train_data1_sigma
plt.plot(x3,y[::-1],'r')
# MT核
model1_3.plot(plot_limits=(395,410))
y = (data1[0:15] - train_data1_mean)/train_data1_sigma
plt.plot(x3,y[::-1],'r')
# RQ核
model1_4.plot(plot_limits=(395,410))
y = (data1[0:15] - train_data1_mean)/train_data1_sigma
plt.plot(x3,y[::-1],'r')


# y_major_locator=MultipleLocator(10)
# ax=plt.gca()
# ax.yaxis.set_major_locator(y_major_locator)


# 评价指标:MSE,MAE,MAPE,NOO(number of outliers,deviation>2*SD)
# SE核
MSE1_1 = np.linalg.norm(predict_mean1_1-test_data1,ord=2)**2/10
MAE1_1 = np.linalg.norm(predict_mean1_1-test_data1,ord=1)/10
MAPE1_1 = np.mean(np.abs(predict_mean1_1-test_data1)/test_data1)*100
NOO1_1 = np.sum(np.abs(predict_mean1_1-test_data1)>2*np.sqrt(predict_var1_1))
print("MSE1_1:", MSE1_1, " MAE1_1:", MAE1_1, " MAPE1_1:", MAPE1_1, " NOO1_1:", NOO1_1, " likelihood", model1_1.log_likelihood())
# EX核
MSE1_2 = np.linalg.norm(predict_mean1_2-test_data1,ord=2)**2/10
MAE1_2 = np.linalg.norm(predict_mean1_2-test_data1,ord=1)/10
MAPE1_2 = np.mean(np.abs(predict_mean1_2-test_data1)/test_data1)*100
NOO1_2 = np.sum(np.abs(predict_mean1_2-test_data1)>2*np.sqrt(predict_var1_2))
print("MSE1_2:", MSE1_2, " MAE1_2:", MAE1_2, " MAPE1_2:", MAPE1_2, " NOO1_2:", NOO1_2, " likelihood", model1_2.log_likelihood())
# MT核
MSE1_3 = np.linalg.norm(predict_mean1_3-test_data1,ord=2)**2/10
MAE1_3 = np.linalg.norm(predict_mean1_3-test_data1,ord=1)/10
MAPE1_3 = np.mean(np.abs(predict_mean1_3-test_data1)/test_data1)*100
NOO1_3 = np.sum(np.abs(predict_mean1_3-test_data1)>2*np.sqrt(predict_var1_3))
print("MSE1_3:", MSE1_3, " MAE1_3:", MAE1_3, " MAPE1_3:", MAPE1_3, " NOO1_3:", NOO1_3, " likelihood", model1_3.log_likelihood())
# RQ核
MSE1_4 = np.linalg.norm(predict_mean1_4-test_data1,ord=2)**2/10
MAE1_4 = np.linalg.norm(predict_mean1_4-test_data1,ord=1)/10
MAPE1_4 = np.mean(np.abs(predict_mean1_4-test_data1)/test_data1)*100
NOO1_4 = np.sum(np.abs(predict_mean1_4-test_data1)>2*np.sqrt(predict_var1_4))
print("MSE1_4:", MSE1_4, " MAE1_4:", MAE1_4, " MAPE1_4:", MAPE1_4, " NOO1_4:", NOO1_4, " likelihood", model1_4.log_likelihood())

# EX核似然值最大，用它来形成复合核
# EX+SE
kernel1_RC1 = kernel1_EX + kernel1_SE
model1_RC1 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_RC1)
# EX+MT
kernel1_RC2 = kernel1_EX + kernel1_MT
model1_RC2 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_RC2)
# EX+RQ
kernel1_RC3 = kernel1_EX + kernel1_RQ
model1_RC3 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_RC3)
# EX*SE
kernel1_RC4 = kernel1_EX * kernel1_SE
model1_RC4 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_RC4)
# EX*MT
kernel1_RC5 = kernel1_EX * kernel1_MT
model1_RC5 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_RC5)
# EX*RQ
kernel1_RC6 = kernel1_EX * kernel1_RQ
model1_RC6 = GPy.models.GPRegression(X=x1, Y=train_data1, kernel=kernel1_RC6)

model1_RC1.optimize(messages=True)
model1_RC2.optimize(messages=True)
model1_RC3.optimize(messages=True)
model1_RC4.optimize(messages=True)
model1_RC5.optimize(messages=True)
model1_RC6.optimize(messages=True)

# 第一次进行核组合后，EX+RQ的似然值最大，进行对其预测
predict_mean1_RC , predict_var1_RC = model1_RC3.predict(x2)

# EX+RQ
model1_RC3.plot(plot_limits=(395,410))
y = (data1[0:15] - train_data1_mean)/train_data1_sigma
plt.plot(x3,y[::-1],'r')

# EX+RQ核
MSE1_RC = np.linalg.norm(predict_mean1_RC-test_data1,ord=2)**2/10
MAE1_RC = np.linalg.norm(predict_mean1_RC-test_data1,ord=1)/10
MAPE1_RC = np.mean(np.abs(predict_mean1_RC-test_data1)/test_data1)*100
NOO1_RC = np.sum(np.abs(predict_mean1_RC-test_data1)>2*np.sqrt(predict_var1_RC))
print("MSE1_RC:", MSE1_RC, " MAE1_RC:", MAE1_RC, " MAPE1_RC:", MAPE1_RC, " NOO1_RC:", NOO1_RC, " likelihood", model1_RC3.log_likelihood())

print("RC1_likelihood:",model1_RC1.log_likelihood())
print("RC2_likelihood:",model1_RC2.log_likelihood())
print("RC3_likelihood:",model1_RC3.log_likelihood())
print("RC4_likelihood:",model1_RC4.log_likelihood())
print("RC5_likelihood:",model1_RC5.log_likelihood())
print("RC6_likelihood:",model1_RC6.log_likelihood())


plt.show()