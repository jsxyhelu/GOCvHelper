////////////////////////////////////////////////////////////////////////////////
////名称：GOCVHelper_2020_04_20.h
////功能：图像处理和MFC增强
////作者：jsxyhelu(1755311380@qq.com http://jsxyhelu.cnblogs.com)
////组织：GREENOPEN
////日期：2019-04-20
//调用算法库请在Opencv正确配置的环境下。
//并在项目-属性-配置属性-c/c++-预处理器-预处理器定义加入 _CRT_SECURE_NO_WARNINGS

#include "pch.h"

#include <io.h>
#include <iostream>
#include <fstream>
#include <opencv.hpp>

#define  DIRECTION_X 0
#define  DIRECTION_Y 1
#define  VP  vector<Point>  //用VP符号代替 vector<point>

using namespace std;
using namespace cv;

//多尺度图像增强
cv::Mat multiScaleSharpen(cv::Mat Src, int Radius);
//读取图片并转灰度
Mat imread2gray(string path);
//带有上下限的threshold
Mat threshold2(Mat src,int minvalue,int maxvalue);
//孔洞填充
Mat fillHoles(Mat src);
float getWhiteRate(Mat src);
Mat getInnerHoles(Mat src);
//顶帽去光差
Mat moveLightDiff(Mat src,int radius = 40);//radius为模板半径
//图像细化
void thin(const Mat &src, Mat &dst, const int iterations=100);//Mat 需要为DEPTH_8U型二值图 
//使得rect区域半透明
Mat translucence(Mat src,Rect rect,int idepth = 90);
//使得rect区域马赛克
Mat mosaic(Mat src,Rect rect,int W = 18,int H = 18);
//基于颜色直方图的距离计算
double GetHsVDistance(Mat src_base,Mat src_test1);
// 正片叠底
void Multiply(Mat& src1, Mat& src2, Mat& dst);
// 颜色加深
void Color_Burn(Mat& src1, Mat& src2, Mat& dst);
// 线性增强
void Linear_Burn(Mat& src1, Mat& src2, Mat& dst);
// 饱和度提高
Mat EnhanceSaturation(Mat temp);
//----------------------------------------------------------------------------------------------------------------------------------------//
//点乘法 
Mat EWM(Mat m1,Mat m2);
//图像局部对比度增强算法
Mat ACE(Mat src,int C = 4,int n=20,int MaxCG = 5);
//LN算法
Mat LocalNormalization(Mat float_gray,float sigma1,float sigma2);
//----------------------------------------------------------------------------------------------------------------------------------------//
//寻找最大的轮廓
VP FindBigestContour(Mat src);
//寻找第n个大轮廓
VP FindnthContour(Mat src,int ith );
//寻找并绘制出彩色联通区域
vector<VP> connection2(Mat src,Mat& draw);
vector<VP> connection2(Mat src);
//根据轮廓的面积大小进行选择
vector<VP>  selectShapeArea(Mat src,Mat& draw,vector<VP> contours,int minvalue,int maxvalue);
vector<VP>  selectShapeArea(vector<VP> contours,int minvalue,int maxvalue);
float calculateCircularity(VP contour);
vector<VP> selectShapeCircularity(vector<VP> contours,float minvalue,float maxvalue);
vector<VP> selectShapeCircularity(Mat src,Mat& draw,vector<VP> contours,float minvalue,float maxvalue);
//返回两点之间的距离
float getDistance(Point2f f1,Point2f f2);
//返回点到直线（线段）的距离
float GetPointLineDistance(Point2f pointInput,Point2f pa,Point2f pb,Point2f& pointOut);
//根据pca方法，返回轮廓的角度
double getOrientation(vector<Point> &pts, Mat &img);
//根据中线将轮廓分为2个部分
//参数：pts 轮廓；pa pb 中线线段端点；p1 p2 分为两边后最远2点；lenght1,length2 对应距离；img 用于绘图
//返回 是否分割成功
bool SplitContoursByMiddleLine(vector<Point> &pts,Mat &img,Point pa,Point pb,Point& p1,float& length1,Point& p2,float& length2);
//获得真实的长宽,返回值为false的话代表识别不成功
bool getRealWidthHeight(vector<Point> &pts,vector<Point> &resultPts, Mat &img,float& flong,float& fshort);
//投影到x或Y轴上,上波形为vup,下波形为vdown,gap为误差间隔
void projection2(Mat src,vector<int>& vup,vector<int>& vdown,int direction = DIRECTION_X,int gap = 10);
//轮廓柔和
bool SmoothEdgeSingleChannel( Mat mInput,Mat &mOutput, double amount, double radius, uchar Threshold) ;
//----------------------------------------------------------------------------------------------------------------------------------------//
//递归读取目录下全部文件
void getFiles(string path, vector<string>& files,string flag ="r"/*如果不想递归这里不写r就可以*/);
//递归读取目录下全部图片
void getFiles(string path, vector<Mat>& files,string flag = "r");
//递归读取目录下全部图片和名称
void getFiles(string path, vector<pair<Mat,string>>& files,string flag="r");
//删除目录下的全部文件
void deleteFiles(string path,string flag = "r");
//创建或续写目录下的csv文件,填写“文件位置-分类”对
int writeCsv(const string& filename,const vector<pair<string,string>>srcVect,char separator=';');
//读取目录下的csv文件,获得“文件位置-分类”对
vector<pair<string,string>> readCsv(const string& filename, char separator = ';') ;
//----------------------------------------------------------------------------------------------------------------------------------------//
//C++的spilt函数
void SplitString(const string& s, vector<string>& v, const string& c);
//! 通过文件夹名称获取文件名，不包括后缀
void getFileName(const string& filepath, string& name,string& lastname);
void getFileName(const string& filepath, string& name);
