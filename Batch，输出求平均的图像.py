'''
Description: 
Version: 1.0
Author: tangliwen
Date: 2022-09-08 10:40:04
LastEditors: tangliwen
LastEditTime: 2022-10-26 17:02:11
'''
import numpy as np
import cv2 as cv
import pandas as pd
from  matplotlib import pyplot as plt
import warnings
import spectral
import preprocessBatchSpectral
import processhyperspectralcls
import sys
from PIL import Image
import glob
from sklearn.feature_selection import VarianceThreshold
warnings.filterwarnings('ignore') #  忽略弹出的warnings信息

def BatchProcess(dataFolder):
    """_summary_
        批量处理数据
    Args:
        datafolder (_type_): 放数据的总文件夹
    """    
    folders = glob.glob(dataFolder + "/*")
    datas = []
    result_df = pd.DataFrame()
    for folder in folders:
        # datarawFile = glob.glob(folder + "/capture/[!DARKREF,!WHITEREF]*.raw")       
        phs = preprocessBatchSpectral.ProcessHyperSpectral()
        folder = folder.replace('\\', '/')
        # 数据提取
        path = folder + "/capture/"
        filename = path.split('/')[-3]  #改成对每个文件名操作
        pa = phs.ReadFromRAW(path)
        pnum=0
        i=0
        df1 = phs.ExtractAndCalcSpectral(pa,filename,pnum,i)#把结果保存到一个excel里面
        result_df = pd.concat([result_df, df1], axis=0, ignore_index=True)

    result_df.to_excel('output3.xlsx', index=False)

    pass

if __name__ == '__main__':
    # Process()F:/pumpkin/2_1_1_emptyname_2023-02-27_02-25-03/capture/
    BatchProcess("E:/pumpkin/NO_2")
    pass
    


class ProcessHyperSpectral:
    """_summary_
        用于处理高光谱数据的类
    Returns:
        _type_: _description_
    """    
    # 属性
    lines = 0
    samples = 0
    bands = 0
    datatype = 0
    data_nparr = 0
    def ReadFromRAW(self,capturePath):
        """_summary_
            用于从RAW中读取数据，生成图像，并裁剪
        Args:
            capturePath (_type_): the raw capture folder path
        """        
        # 单个数据的文件夹
        # dataFolderPath = "E:/研究生/光谱/data/HyperSpectralImage/陕西2000/shanxi_canhunyang_3_emptyname_2022-07-29_07-46-07/capture/"
        dataFolderPath = capturePath
        # 读取目录下HDR文件
        listdata = dataFolderPath.split('/')
        f = filter(lambda x:x.find("empty") >= 0,listdata)
        l = list(f)
        l = l[0].split('/')
        hdrPath = dataFolderPath + l[0] + ".hdr"
        rawPath = dataFolderPath + l[0] + ".raw"
        hdrHeader = spectral.envi.read_envi_header(hdrPath)

        # 根据头文件内容读取初始大小
        self.lines = int(hdrHeader['lines'])
        self.samples = int(hdrHeader['samples'])
        self.bands = int(hdrHeader['bands'])
        self.datatype = hdrHeader['data type']

        # 经过周折终于从https://eufat.github.io/2019/02/19/hyperspectral-image-preprocessing-with-python.html发现了正确的读取解决方法
        data_ref = spectral.envi.open(dataFolderPath + l[0] + ".hdr", dataFolderPath + l[0] + ".raw")
        white_ref = spectral.envi.open(dataFolderPath + "WHITEREF_" + l[0] + ".hdr", dataFolderPath + "WHITEREF_" + l[0] + ".raw")
        dark_ref = spectral.envi.open(dataFolderPath + "DARKREF_" + l[0] + ".hdr", dataFolderPath + "DARKREF_" + l[0] + ".raw")

        self.white_nparr = np.array(white_ref.load())
        self.dark_nparr = np.array(dark_ref.load())
        self.data_nparr = np.array(data_ref.load())


        # 图像裁剪
        self.data_nparr = self.data_nparr[10:600,50:550,:]
        self.dark_nparr = self.dark_nparr[10:600,50:550,:]
        self.white_nparr = self.white_nparr[10:600,50:550,:]

        self.corrected_nparr = np.divide(
            np.subtract(self.data_nparr, self.dark_nparr),
            np.subtract(self.white_nparr, self.dark_nparr)
            )

        # im = im[10:370,50:580]
        # 选择一个波段的图片导出
        # for i in range(1):
        for i in range(224):
            im = Image.fromarray(self.corrected_nparr[:,:,i])
            im = im.convert('L')  # 这样才能转为灰度图,如果是彩色图则改L为‘RGB’
            im.save('hyperdata/outcorrdata/outcorrdata_'+ str(i) + '.jpg')
            im2 = Image.fromarray(self.data_nparr[:,:,i])
            im2 = im2.convert('L')  # 这样才能转为灰度图,如果是彩色图则改L为‘RGB’
            im2.save('hyperdata/outdata/outband_'+ str(i) + '.jpg')


        return self.data_nparr
    
    
    def ExtractAndCalcSpectral(self,corrected_nparr,name,pnum,i):
        """_summary_
            用于提取图像中的种子，同时计算他们的平均光谱
        Returns:
            _type_: _description_
        """        
        # 寻找边缘

        im = cv.imread('hyperdata/outdata/outband_1.jpg',0)
        imgray=im
        ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # ret, thresh = cv.threshold(imgray, 150, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # # 查找连通域
        retval, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)
        imgray = cv.imread('hyperdata/outdata/outband_1.jpg',1)
        cv.imwrite("imgray.jpg",imgray)

        im = cv.imread('hyperdata/outdata/outband_1.jpg',0)
        imgray=im
        ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_OTSU)
        # 查找连通域
        retval, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)
        for i, stat in enumerate(stats):
            #绘制连通区域
            cv.rectangle(thresh, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (255, 0, 0), 1)
            #按照连通区域的索引来打上标签
            cv.putText(thresh, str(i), (stat[0], stat[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv.imwrite(sys.path[0]+ "/hyperdata/outresult/" +"BA_"+ name + '.jpg',thresh)
        im = cv.imread('hyperdata/outdata/outband_2.jpg',0)


        # 比种子数量稍微多一点
        df = pd.DataFrame()
        print(i)
        
        countArr = np.zeros(1000)
        for x in range(corrected_nparr.shape[0]):
            for y in range(corrected_nparr.shape[1]):
                if labels[x,y] != 0 and corrected_nparr[x,y,].__contains__(np.Inf) == False:
                    if df.columns.__contains__(labels[x,y]):               
                        df[labels[x,y]] = df[labels[x,y]] + corrected_nparr[x,y,:]
                    else:
                        df[labels[x,y]] = corrected_nparr[x,y,:] 
                    countArr[labels[x,y]] = countArr[labels[x,y]] + 1
        
        pnum=np.argmax(countArr)
###################导出图片
        pumpkin_data=corrected_nparr[stats[pnum][1]:stats[pnum][1]+stats[pnum][3],stats[pnum][0]:stats[pnum][0]+stats[pnum][2],0]
        
        cv.imwrite( 'hyperdata/img/'+name+ '.jpg',pumpkin_data)

        # df[pnum] = round((df[pnum]/ countArr[pnum]).astype('float'),2)
        # dfmax=df[pnum]
        # print(pnum)
        # dfmax=dfmax.transpose()
        # # selector = VarianceThreshold(threshold=1)
        # # df_corrdata = selector.fit_transform(df)
        # dftest = pd.DataFrame(dfmax)
        # # dftest = dftest.drop(dftest[dftest[0] == 0].index)
        # # dftest = pd.DataFrame(np.transpose(dftest))
        # print(name)
        # print(dftest.shape)
        # plt.plot(dftest)
        # # plt.savefig(sys.path[0]+ "/hyperdata/outresult/" + name + '.jpg')
        # plt.clf()
        # # 导出数据
        # # dftest.to_excel(sys.path[0] + "/hyperdata/outresult/" + name + ".xlsx")
        # dftest=dftest.transpose()
        # dftest.insert(loc=0,column='filename', value=name[:6])
        # print(dftest.shape)
        return df

        """_summary_
            数据预处理，提取连通域，计算平均光谱，采用光谱预处理算法得到结果
        Args:
            data_nparr (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # 寻找边缘
        im = cv.imread('outband_0.jpg',0)
        imgray = im
        ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_OTSU)

        # 查找连通域
        retval, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)
        for i, stat in enumerate(stats):
        # 绘制连通区域
            cv.rectangle(thresh, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (255, 0, 0), 1)
        # 按照连通区域的索引来打上标签
            cv.putText(thresh, str(i+1), (stat[0], stat[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        plt.imshow(thresh)
        
        # 计算平均光谱
        countArr = np.zeros(320)       
        df = pd.DataFrame()
        for x in range(data_nparr.shape[0]):
            for y in range(data_nparr.shape[1]):
                if labels[x,y] != 0:
                    if df.columns.__contains__(labels[x,y]):
                        df[labels[x,y]] = df[labels[x,y]] + data_nparr[x,y,:]
                    else:
                        df[labels[x,y]] = data_nparr[x,y,:]
                    countArr[labels[x,y]] = countArr[labels[x,y]] + 1

        for col in df.columns:
            if countArr[col] < 300 and  countArr[col] > 50:
                df[col] = round((df[col]/ countArr[col]).astype('float'),2)
            else:
                df[col] = 0
        # 绘图，这里删了最后一个连通域
        df = df.iloc[:,:-1]

        dftest = pd.DataFrame(np.transpose(df))
        dftest = dftest.drop(dftest[dftest[0] == 0].index)
        dftest = pd.DataFrame(np.transpose(dftest))
        print(dftest.shape)
        plt.plot(dftest)
        plt.savefig(name + '.jpg')

        # 导出数据
        dftest.to_excel(name + ".xlsx")
        return df