import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython import display
import scipy.io as sio
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, recall_score
import spectral
git config user.email "1931827153@qq.com"
git config user.name "Dreamvai"

git remote add origin https://github.com/Dreamvai/LG-GCN.git

dataname = 'IP' ##IP,SV,PU
spc = 30
if dataname == 'PU':
    block = [5, 10, 15, 20, 25, 30, 35, 40]
    c = 103
    classx = 9
if dataname == 'SV':
    block = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    c = 204
    classx = 16
if dataname == 'IP':
    block = [5, 10, 15, 20, 25]
    c = 200
    classx = 16
#block = []
high_contrast_colors = [
            '#000000',  # 黑色
            '#ff0000',  # 红色
            '#00ff00',  # 绿色
            '#0000ff',  # 蓝色
            '#ffff00',  # 黄色
            '#ff00ff',  # 品红色
            '#00ffff',  # 青色
            '#c86400',  # 橙褐色
            '#00c864',  # 春绿色
            '#6400c8',  # 紫色
            '#c80064',  # 玫瑰红色
            '#64c800',  # 草绿色
            '#0064c8',  # 天蓝色
            '#964b4b',  # 棕褐色
            '#4b964b',  # 橄榄绿色
            '#4b4b96',  # 钴蓝色
            '#ff6464',  # 浅红色
            '#64ff64',  # 浅绿色
            '#6464ff',  # 浅蓝色
            '#ff964b',  # 桔黄色
            '#4bff96',  # 浅青色
            '#964bff',  # 紫罗兰色
            '#323232',  # 暗灰色
            '#646464',  # 中灰色
            '#969696',  # 银色
            '#c8c8c8',  # 亮灰色
            '#fafafa',  # 非常浅的灰色
            '#640000',  # 暗红色
            '#c80000',  # 鲜红色
            '#006400',  # 暗绿色
            '#00c800',  # 鲜绿色
            '#000064',  # 暗蓝色
            '#0000c8',  # 鲜蓝色
            '#646400',  # 橄榄色
            '#c8c800',  # 金色
            '#640064',  # 暗紫色
            '#c800c8',  # 鲜紫色
            '#006464',  # 暗青色
            '#00c8c8',  # 鲜青色

        ]
def remove_zeros(x, y):
    # 确保x和y的长度相同
    if len(x) != len(y):
        return "x和y的长度不一致，请提供长度相同的数组。"

    # 使用列表推导式创建新的列表，只包含非零元素
    x_filtered = [xi for xi, yi in zip(x, y) if yi != 0]
    y_filtered = [yi for yi in y if yi != 0]

    return x_filtered, y_filtered

def set_zeros_in_x(x, y):
    # 遍历y中的每个元素
    for i in range(len(y)):
        for j in range(len(y[i])):
            # 如果y中的元素为0，则将x中对应的元素设置为0
            if y[i][j] == 0:
                x[i][j] = 0
    return x

def loadData(name, i = 0, block = 100, spc = 10,  model = 'all'):
    data_path = os.path.join(os.getcwd(), 'mnt')
    if name == 'IP':
        if model == 'all':
            data = sio.loadmat(f'X://Project//myproject//data//prediction//Indian_pines//{spc}//PCA{c}//{block}_overall_skip_2_SGConv_l1_clip//{i}.mat')['pred']
        else:
            data = sio.loadmat(f'X://Project//myproject//data//prediction//Indian_pines//{spc}//PCA{c}//integratedLearning_l1_clip//{i}.mat')['pred']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        if model == 'all':
            data = sio.loadmat(f'X://Project//myproject//data//prediction//Salinas//{spc}//PCA{c}//{block}_overall_skip_2_SGConv_l1_clip//{i}.mat')['pred']
        else:
            data = sio.loadmat(f'X://Project//myproject//data//prediction//Salinas//{spc}//PCA{c}//integratedLearning_l1_clip//{i}.mat')['pred']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        if model == 'all':
            data = sio.loadmat(f'X://Project//myproject//data//prediction//PaviaU//{spc}//PCA{c}//{block}_overall_skip_2_SGConv_l1_clip//{i}.mat')['pred']
        else:
            data = sio.loadmat(f'X://Project//myproject//data//prediction//PaviaU//{spc}//PCA{c}//integratedLearning_l1_clip//{i}.mat')['pred']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']

    else:
        print("NO DATASET")
        exit()


    return data, labels


for b in block:
    all_oa = []
    all_aa = []
    all_kappa = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    class6 = []
    class7 = []
    class8 = []
    class9 = []
    class10 = []
    class11 = []
    class12 = []
    class13 = []
    class14 = []
    class15 = []
    class16 = []
    for i in range(10):
        x, y = loadData(dataname, i=i, block=b, spc=spc)
        x = x + 1
        # print(x)
        # print(y)

        # 调用函数并打印结果
        x = set_zeros_in_x(x, y)
        save_root = '可视化/{}/非集成学习/{}/'.format(dataname, b)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        # 创建一个自定义的颜色映射
        cmap = ListedColormap(high_contrast_colors[:classx])
        plt.figure(figsize=(8, 8))  # 可以调整图片大小
        plt.imshow(x.astype(int), cmap=cmap, interpolation='nearest')
        plt.axis('off')  # 关闭坐标轴
        display.clear_output(wait=True)
        # 保存图片为矢量图格式，没有坐标轴
        plt.savefig(f'可视化/{dataname}/非集成学习/{b}/第{i}次' + " 预测.svg", format='svg', bbox_inches='tight', pad_inches=0)
        plt.close()
        #plt.show()
        #spectral.save_rgb(f'可视化/{dataname}/非集成学习/{b}/第{i}次' + " 预测.jpg", x.astype(int), colors=spectral.spy_colors)
        # spectral.save_rgb(f'{i}' + " 原始.jpg", y.astype(int), colors=spectral.spy_colors)
        x = x.flatten()
        y = y.flatten()
        # print(x)
        # print(y)

        # 调用函数
        x, y = remove_zeros(x, y)

        x = np.array(x)
        y = np.array(y)
        # print(x.shape)
        # print(y.shape)

        # 计算整体精度
        overall_accuracy = accuracy_score(y, x)

        # 初始化一个字典来存储每个类别的精度
        class_accuracies = {}
        # 遍历每个类别
        for label in set(y):
            # 获取具有当前标签的样本的索引
            indices = [i for i, true_label in enumerate(y) if true_label == label]
            # 计算当前类别的精度
            class_accuracy = accuracy_score([y[i] for i in indices], [x[i] for i in indices])
            # 将类别精度存储在字典中
            class_accuracies[label] = class_accuracy

        # 打印整体精度和每个类别的精度
        #print(f"整体精度：{overall_accuracy:.4f}")
        for label, accuracy in class_accuracies.items():
            if label == 1:
                class1 = np.append(class1, accuracy)
            if label == 2:
                class2 = np.append(class2, accuracy)
            if label == 3:
                class3 = np.append(class3, accuracy)
            if label == 4:
                class4 = np.append(class4, accuracy)
            if label == 5:
                class5 = np.append(class5, accuracy)
            if label == 6:
                class6 = np.append(class6, accuracy)
            if label == 7:
                class7 = np.append(class7, accuracy)
            if label == 8:
                class8 = np.append(class8, accuracy)
            if label == 9:
                class9 = np.append(class9, accuracy)
            if label == 10:
                class10 = np.append(class10, accuracy)
            if label == 11:
                class11 = np.append(class11, accuracy)
            if label == 12:
                class12 = np.append(class12, accuracy)
            if label == 13:
                class13 = np.append(class13, accuracy)
            if label == 14:
                class14 = np.append(class14, accuracy)
            if label == 15:
                class15 = np.append(class15, accuracy)
            if label == 16:
                class16 = np.append(class16, accuracy)
            #print(f"类别 {label} 的精度：{accuracy:.4f}")
        #print(class_accuracies)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y, x)
        # 计算OA（Overall Accuracy）
        oa = accuracy_score(y, x)
        all_oa = np.append(all_oa, oa)
        # 计算Kappa系数
        kappa = cohen_kappa_score(y, x)
        all_kappa = np.append(all_kappa, kappa)

        # 计算每个类别的召回率
        recall_per_class = recall_score(y, x, average=None)
        # 计算AA
        aa = recall_per_class.mean()
        all_aa = np.append(all_aa, aa)

        # print(f"第{i}次的OA: {oa}")
        # print(f"第{i}次Kappa: {kappa}")
        # print(f"第{i}次AA: {aa}")
        # print('\n')
    mean_1 = np.mean(class1)
    std_1 = np.std(class1)
    print(f"{dataname}数据集上 Block={b}时 class1: {mean_1}±{std_1}")

    mean_2 = np.mean(class2)
    std_2 = np.std(class2)
    print(f"{dataname}数据集上 Block={b}时 class2: {mean_2}±{std_2}")

    mean_3 = np.mean(class3)
    std_3 = np.std(class3)
    print(f"{dataname}数据集上 Block={b}时 class3: {mean_3}±{std_3}")

    mean_4 = np.mean(class4)
    std_4 = np.std(class4)
    print(f"{dataname}数据集上 Block={b}时 class4: {mean_4}±{std_4}")

    mean_5 = np.mean(class5)
    std_5 = np.std(class5)
    print(f"{dataname}数据集上 Block={b}时 class5: {mean_5}±{std_5}")

    mean_6 = np.mean(class6)
    std_6 = np.std(class6)
    print(f"{dataname}数据集上 Block={b}时 class6: {mean_6}±{std_6}")

    mean_7 = np.mean(class7)
    std_7 = np.std(class7)
    print(f"{dataname}数据集上 Block={b}时 class7: {mean_7}±{std_7}")

    mean_8 = np.mean(class8)
    std_8 = np.std(class8)
    print(f"{dataname}数据集上 Block={b}时 class8: {mean_8}±{std_8}")

    mean_9 = np.mean(class9)
    std_9 = np.std(class9)
    print(f"{dataname}数据集上 Block={b}时 class9: {mean_9}±{std_9}")

    mean_10 = np.mean(class10)
    std_10 = np.std(class10)
    print(f"{dataname}数据集上 Block={b}时 class10: {mean_10}±{std_10}")

    mean_11 = np.mean(class11)
    std_11 = np.std(class11)
    print(f"{dataname}数据集上 Block={b}时 class11: {mean_11}±{std_11}")

    mean_12 = np.mean(class12)
    std_12 = np.std(class12)
    print(f"{dataname}数据集上 Block={b}时 class12: {mean_12}±{std_12}")

    mean_13 = np.mean(class13)
    std_13 = np.std(class13)
    print(f"{dataname}数据集上 Block={b}时 class13: {mean_13}±{std_13}")

    mean_14 = np.mean(class14)
    std_14 = np.std(class14)
    print(f"{dataname}数据集上 Block={b}时 class14: {mean_14}±{std_14}")

    mean_15 = np.mean(class15)
    std_15 = np.std(class15)
    print(f"{dataname}数据集上 Block={b}时 class15: {mean_15}±{std_15}")

    mean_16 = np.mean(class16)
    std_16 = np.std(class16)
    print(f"{dataname}数据集上 Block={b}时 class16: {mean_16}±{std_16}")



    mean_oa = np.mean(all_oa)
    std_oa = np.std(all_oa)

    mean_aa = np.mean(all_aa)
    std_aa = np.std(all_aa)

    mean_kappa = np.mean(all_kappa)
    std_kappa = np.std(all_kappa)

    print('OA列表', all_oa)
    print('AA列表', all_aa)
    print('Kappa列表', all_kappa)
    print(f"{dataname}数据集上 Block={b}时 OA: {mean_oa}±{std_oa}")
    print(f"{dataname}数据集上 Block={b}时 AA: {mean_aa}±{std_aa}")
    print(f"{dataname}数据集上 Block={b}时 KAPPA: {mean_kappa}±{std_kappa}")
    print('\n')

all_oa = []
all_aa = []
all_kappa = []
class1 = []
class2 = []
class3 = []
class4 = []
class5 = []
class6 = []
class7 = []
class8 = []
class9 = []
class10 = []
class11 = []
class12 = []
class13 = []
class14 = []
class15 = []
class16 = []

for i in range(10):
    x, y = loadData(dataname, i=i, spc=spc, model='no')
    x = x + 1
    # print(x)
    # print(y)

    # 调用函数并打印结果
    x = set_zeros_in_x(x, y)
    # 创建一个自定义的颜色映射
    cmap = ListedColormap(high_contrast_colors[:classx])
    plt.figure(figsize=(8, 8))  # 可以调整图片大小
    plt.imshow(x.astype(int), cmap=cmap, interpolation='nearest')
    plt.axis('off')  # 关闭坐标轴
    display.clear_output(wait=True)
    # 保存图片为矢量图格式，没有坐标轴
    plt.savefig(f'可视化/{dataname}/集成学习/第{i}次' + " 预测.svg", format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()
    #spectral.save_rgb(f'可视化/{dataname}/集成学习/第{i}次' + " 预测.jpg", x.astype(int), colors=spectral.spy_colors)
    #spectral.save_rgb(f'可视化/{dataname}/非集成学习/第{i}次' + " 预测.jpg", x.astype(int), colors=spectral.spy_colors)
    # spectral.save_rgb(f'{i}' + " 原始.jpg", y.astype(int), colors=spectral.spy_colors)

    x = x.flatten()
    y = y.flatten()
    # print(x)
    # print(y)

    # 调用函数
    x, y = remove_zeros(x, y)

    x = np.array(x)
    y = np.array(y)
    # print(x.shape)
    # print(y.shape)

    # 计算整体精度
    overall_accuracy = accuracy_score(y, x)

    # 初始化一个字典来存储每个类别的精度
    class_accuracies = {}
    # 遍历每个类别
    for label in set(y):
        # 获取具有当前标签的样本的索引
        indices = [i for i, true_label in enumerate(y) if true_label == label]
        # 计算当前类别的精度
        class_accuracy = accuracy_score([y[i] for i in indices], [x[i] for i in indices])
        # 将类别精度存储在字典中
        class_accuracies[label] = class_accuracy

    # 打印整体精度和每个类别的精度
    # print(f"整体精度：{overall_accuracy:.4f}")
    for label, accuracy in class_accuracies.items():
        if label == 1:
            class1 = np.append(class1, accuracy)
        if label == 2:
            class2 = np.append(class2, accuracy)
        if label == 3:
            class3 = np.append(class3, accuracy)
        if label == 4:
            class4 = np.append(class4, accuracy)
        if label == 5:
            class5 = np.append(class5, accuracy)
        if label == 6:
            class6 = np.append(class6, accuracy)
        if label == 7:
            class7 = np.append(class7, accuracy)
        if label == 8:
            class8 = np.append(class8, accuracy)
        if label == 9:
            class9 = np.append(class9, accuracy)
        if label == 10:
            class10 = np.append(class10, accuracy)
        if label == 11:
            class11 = np.append(class11, accuracy)
        if label == 12:
            class12 = np.append(class12, accuracy)
        if label == 13:
            class13 = np.append(class13, accuracy)
        if label == 14:
            class14 = np.append(class14, accuracy)
        if label == 15:
            class15 = np.append(class15, accuracy)
        if label == 16:
            class16 = np.append(class16, accuracy)
        # print(f"类别 {label} 的精度：{accuracy:.4f}")
    # print(class_accuracies)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y, x)
    # 计算OA（Overall Accuracy）
    oa = accuracy_score(y, x)
    all_oa = np.append(all_oa, oa)
    # 计算Kappa系数
    kappa = cohen_kappa_score(y, x)
    all_kappa = np.append(all_kappa, kappa)

    # 计算每个类别的召回率
    recall_per_class = recall_score(y, x, average=None)
    # 计算AA
    aa = recall_per_class.mean()
    all_aa = np.append(all_aa, aa)

    # print(f"第{i}次的OA: {oa}")
    # print(f"第{i}次Kappa: {kappa}")
    # print(f"第{i}次AA: {aa}")
    # print('\n')
mean_1 = np.mean(class1)
std_1 = np.std(class1)
print(f"{dataname}数据集上 class1: {mean_1}±{std_1}")

mean_2 = np.mean(class2)
std_2 = np.std(class2)
print(f"{dataname}数据集上  class2: {mean_2}±{std_2}")

mean_3 = np.mean(class3)
std_3 = np.std(class3)
print(f"{dataname}数据集上  class3: {mean_3}±{std_3}")

mean_4 = np.mean(class4)
std_4 = np.std(class4)
print(f"{dataname}数据集上  class4: {mean_4}±{std_4}")

mean_5 = np.mean(class5)
std_5 = np.std(class5)
print(f"{dataname}数据集上  class5: {mean_5}±{std_5}")

mean_6 = np.mean(class6)
std_6 = np.std(class6)
print(f"{dataname}数据集上  class6: {mean_6}±{std_6}")

mean_7 = np.mean(class7)
std_7 = np.std(class7)
print(f"{dataname}数据集上  class7: {mean_7}±{std_7}")

mean_8 = np.mean(class8)
std_8 = np.std(class8)
print(f"{dataname}数据集上  class8: {mean_8}±{std_8}")

mean_9 = np.mean(class9)
std_9 = np.std(class9)
print(f"{dataname}数据集上  class9: {mean_9}±{std_9}")

mean_10 = np.mean(class10)
std_10 = np.std(class10)
print(f"{dataname}数据集上  class10: {mean_10}±{std_10}")

mean_11 = np.mean(class11)
std_11 = np.std(class11)
print(f"{dataname}数据集上  class11: {mean_11}±{std_11}")

mean_12 = np.mean(class12)
std_12 = np.std(class12)
print(f"{dataname}数据集上  class12: {mean_12}±{std_12}")

mean_13 = np.mean(class13)
std_13 = np.std(class13)
print(f"{dataname}数据集上 class13: {mean_13}±{std_13}")

mean_14 = np.mean(class14)
std_14 = np.std(class14)
print(f"{dataname}数据集上  class14: {mean_14}±{std_14}")

mean_15 = np.mean(class15)
std_15 = np.std(class15)
print(f"{dataname}数据集上  class15: {mean_15}±{std_15}")

mean_16 = np.mean(class16)
std_16 = np.std(class16)
print(f"{dataname}数据集上  class16: {mean_16}±{std_16}")


mean_oa = np.mean(all_oa)
std_oa = np.std(all_oa)

mean_aa = np.mean(all_aa)
std_aa = np.std(all_aa)

mean_kappa = np.mean(all_kappa)
std_kappa = np.std(all_kappa)

print(all_oa)
print('AA列表', all_aa)
print('Kappa列表', all_kappa)
print(f"{dataname}数据集上 集成学习时 OA: {mean_oa}±{std_oa}")
print(f"{dataname}数据集上 集成学习时 AA: {mean_aa}±{std_aa}")
print(f"{dataname}数据集上 集成学习时 KAPPA: {mean_kappa}±{std_kappa}")
print('\n')













