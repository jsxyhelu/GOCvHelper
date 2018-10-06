//////////////////////////////////////////////////////////////////////////////
//名称：GOCVHelper0.7b.cpp
//功能：图像处理和MFC增强
//作者：jsxyhelu(1755311380@qq.com http://jsxyhelu.cnblogs.com)
//组织：GREENOPEN
//日期：2016-09-04
/////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include <io.h>
#include <odbcinst.h>
#include <afxdb.h>
#include "GoCvHelper.h"
#include "opencv/cv.h"
#include "atlstr.h"

RNG  rng(12345);
//2016年1月26日GoCvHelper添加string 相关操作函数到其他操作中
//2016年1月28日10:45:22 GOCVHelper基于颜色直方图的CBIR到图像操作中去
//2016年8月12日08:27:03 添加关于excel操作相关函数
namespace GO{

#pragma region 图像操作
	//读取灰度或彩色图片到灰度
	Mat imread2gray(string path){
		Mat src = imread(path);
		Mat srcClone = src.clone();
		if (CV_8UC3 == srcClone.type() )
			cvtColor(srcClone,srcClone,CV_BGR2GRAY);
		return srcClone;
	}

	//带有上下限的threshold
	Mat threshold2(Mat src,int minvalue,int maxvalue){
		Mat thresh1;
		Mat thresh2;
		Mat dst;
		threshold(src,thresh1,minvalue,255, THRESH_BINARY);
		threshold(src,thresh2,maxvalue,255,THRESH_BINARY_INV);
		dst = thresh1 & thresh2;
		return dst;
	}

	//自适应门限的canny算法 
#pragma region canny2
	Mat canny2(Mat src){
		Mat imagetmp = src.clone();
		double low_thresh = 0.0;  
		double high_thresh = 0.0;  
		AdaptiveFindThreshold(imagetmp,&low_thresh,&high_thresh);
		Canny(imagetmp,imagetmp,low_thresh,high_thresh);   
		return imagetmp;}
	void AdaptiveFindThreshold( Mat src,double *low,double *high,int aperture_size){
		const int cn = src.channels();
		Mat dx(src.rows,src.cols,CV_16SC(cn));
		Mat dy(src.rows,src.cols,CV_16SC(cn));
		Sobel(src,dx,CV_16S,1,0,aperture_size,1,0,BORDER_REPLICATE);
		Sobel(src,dy,CV_16S,0,1,aperture_size,1,0,BORDER_REPLICATE);
		CvMat _dx = dx;
		CvMat _dy = dy;
		_AdaptiveFindThreshold(&_dx, &_dy, low, high); }  
	void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high){                                                                                
		CvSize size;                                                             
		IplImage *imge=0;                                                        
		int i,j;                                                                 
		CvHistogram *hist;                                                       
		int hist_size = 255;                                                     
		float range_0[]={0,256};                                                 
		float* ranges[] = { range_0 };                                           
		double PercentOfPixelsNotEdges = 0.7;                                    
		size = cvGetSize(dx);                                                    
		imge = cvCreateImage(size, IPL_DEPTH_32F, 1);                            
		// 计算边缘的强度, 并存于图像中                                          
		float maxv = 0;                                                          
		for(i = 0; i < size.height; i++ ){                                                                        
			const short* _dx = (short*)(dx->data.ptr + dx->step*i);          
			const short* _dy = (short*)(dy->data.ptr + dy->step*i);          
			float* _image = (float *)(imge->imageData + imge->widthStep*i);  
			for(j = 0; j < size.width; j++){                                                                
				_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));          
				maxv = maxv < _image[j] ? _image[j]: maxv;}}                                                                        
		if(maxv == 0){                                                           
			*high = 0;                                                       
			*low = 0;                                                        
			cvReleaseImage( &imge );                                         
			return;}                                                                        
		// 计算直方图                                                            
		range_0[1] = maxv;                                                       
		hist_size = (int)(hist_size > maxv ? maxv:hist_size);                    
		hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);            
		cvCalcHist( &imge, hist, 0, NULL );                                      
		int total = (int)(size.height * size.width * PercentOfPixelsNotEdges);   
		float sum=0;                                                             
		int icount = hist->mat.dim[0].size;                                     
		float *h = (float*)cvPtr1D( hist->bins, 0 );                             
		for(i = 0; i < icount; i++){                                                                        
			sum += h[i];                                                     
			if( sum > total )                                                
				break; }                                                                        
		// 计算高低门限                                                          
		*high = (i+1) * maxv / hist_size ;                                       
		*low = *high * 0.4;                                                      
		cvReleaseImage( &imge );                                                 
		cvReleaseHist(&hist); }     
#pragma endregion canny2

	//填充孔洞
#pragma region fillholes
	Mat fillHoles(Mat src){
		Mat dst = getInnerHoles(src);
		threshold(dst,dst,0,255,THRESH_BINARY_INV);
		dst = src + dst;
		return dst;
	}
	//获得图像中白色的比率
	float getWhiteRate(Mat src){
		int iWhiteSum = 0;
		for (int x =0;x<src.rows;x++){
			for (int y=0;y<src.cols;y++){
				if (src.at<uchar>(x,y) != 0)
					iWhiteSum = iWhiteSum +1;
			}
		}
		return (float)iWhiteSum/(float)(src.rows*src.cols);
	}
	//获得内部孔洞图像
	Mat getInnerHoles(Mat src){ 
		Mat clone = src.clone();
		srand((unsigned)time(NULL));  // 生成时间种子
		float fPreRate = getWhiteRate(clone);
		float fAftRate = 0;
		do {
			clone = src.clone();
			// x y 对于 cols rows
			floodFill(clone,Point((int)rand()%src.cols,(int)rand()%src.rows),Scalar(255));
			fAftRate = getWhiteRate(clone);
		} while ( fAftRate < 0.6);
		return clone;
	}
#pragma endregion fillHoles

	//顶帽去光差,radius为模板半径
	Mat moveLightDiff(Mat src,int radius){
		Mat dst;
		Mat srcclone = src.clone();
		Mat mask = Mat::zeros(radius*2,radius*2,CV_8U);
		circle(mask,Point(radius,radius),radius,Scalar(255),-1);
		//顶帽
		erode(srcclone,srcclone,mask);
		dilate(srcclone,srcclone,mask);
		dst =  src - srcclone;
		return dst;
	}

	//将 DEPTH_8U型二值图像进行细化  经典的Zhang并行快速细化算法
#pragma region thin
	void thin(const Mat &src, Mat &dst, const int iterations){
		const int height =src.rows -1;
		const int width  =src.cols -1;
		//拷贝一个数组给另一个数组
		if(src.data != dst.data)
			src.copyTo(dst);
		int n = 0,i = 0,j = 0;
		Mat tmpImg;
		uchar *pU, *pC, *pD;
		bool isFinished =FALSE;
		for(n=0; n<iterations; n++){
			dst.copyTo(tmpImg); 
			isFinished =FALSE;   //一次 先行后列扫描 开始
			//扫描过程一 开始
			for(i=1; i<height;  i++) {
				pU = tmpImg.ptr<uchar>(i-1);
				pC = tmpImg.ptr<uchar>(i);
				pD = tmpImg.ptr<uchar>(i+1);
				for(int j=1; j<width; j++){
					if(pC[j] > 0){
						int ap=0;
						int p2 = (pU[j] >0);
						int p3 = (pU[j+1] >0);
						if (p2==0 && p3==1)
							ap++;
						int p4 = (pC[j+1] >0);
						if(p3==0 && p4==1)
							ap++;
						int p5 = (pD[j+1] >0);
						if(p4==0 && p5==1)
							ap++;
						int p6 = (pD[j] >0);
						if(p5==0 && p6==1)
							ap++;
						int p7 = (pD[j-1] >0);
						if(p6==0 && p7==1)
							ap++;
						int p8 = (pC[j-1] >0);
						if(p7==0 && p8==1)
							ap++;
						int p9 = (pU[j-1] >0);
						if(p8==0 && p9==1)
							ap++;
						if(p9==0 && p2==1)
							ap++;
						if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7){
							if(ap==1){
								if((p2*p4*p6==0)&&(p4*p6*p8==0)){                           
									dst.ptr<uchar>(i)[j]=0;
									isFinished =TRUE;                            
								}
							}
						}                    
					}

				} //扫描过程一 结束
				dst.copyTo(tmpImg); 
				//扫描过程二 开始
				for(i=1; i<height;  i++){
					pU = tmpImg.ptr<uchar>(i-1);
					pC = tmpImg.ptr<uchar>(i);
					pD = tmpImg.ptr<uchar>(i+1);
					for(int j=1; j<width; j++){
						if(pC[j] > 0){
							int ap=0;
							int p2 = (pU[j] >0);
							int p3 = (pU[j+1] >0);
							if (p2==0 && p3==1)
								ap++;
							int p4 = (pC[j+1] >0);
							if(p3==0 && p4==1)
								ap++;
							int p5 = (pD[j+1] >0);
							if(p4==0 && p5==1)
								ap++;
							int p6 = (pD[j] >0);
							if(p5==0 && p6==1)
								ap++;
							int p7 = (pD[j-1] >0);
							if(p6==0 && p7==1)
								ap++;
							int p8 = (pC[j-1] >0);
							if(p7==0 && p8==1)
								ap++;
							int p9 = (pU[j-1] >0);
							if(p8==0 && p9==1)
								ap++;
							if(p9==0 && p2==1)
								ap++;
							if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7){
								if(ap==1){
									if((p2*p4*p8==0)&&(p2*p6*p8==0)){                           
										dst.ptr<uchar>(i)[j]=0;
										isFinished =TRUE;                            
									}
								}
							}                    
						}
					}
				} //一次 先行后列扫描完成          
				//如果在扫描过程中没有删除点，则提前退出
				if(isFinished ==FALSE)
					break; 
			}
		}
	}
#pragma endregion thin

	//使得rect区域半透明
	Mat translucence(Mat src,Rect rect,int idepth){
		Mat dst = src.clone();
		Mat roi = dst(rect);
		roi += cv::Scalar(idepth,idepth,idepth);
		return dst;
	}

	//使得rect区域打上马赛克
	Mat mosaic(Mat src,Rect rect,int W,int H){
		Mat dst = src.clone();
		Mat roi = dst(rect);
		for (int i=W; i<roi.cols; i+=W) {
			for (int j=H; j<roi.rows; j+=H) {
				uchar s=roi.at<uchar>(j-H/2,(i-W/2)*3);
				uchar s1=roi.at<uchar>(j-H/2,(i-W/2)*3+1);
				uchar s2=roi.at<uchar>(j-H/2,(i-W/2)*3+2);
				for (int ii=i-W; ii<=i; ii++) {
					for (int jj=j-H; jj<=j; jj++) {
						roi.at<uchar>(jj,ii*3+0)=s;
						roi.at<uchar>(jj,ii*3+1)=s1;
						roi.at<uchar>(jj,ii*3+2)=s2;
					}
				}
			}
		}
		return dst;}
//基于颜色直方图的距离计算
double GetHsVDistance(Mat src_base,Mat src_test1){
	Mat   hsv_base;
	Mat   hsv_test1;
	///  Convert  to  HSV
	cvtColor(  src_base,  hsv_base,  COLOR_BGR2HSV  );
	cvtColor(  src_test1,  hsv_test1,  COLOR_BGR2HSV  );
	///  Using  50  bins  for  hue  and  60  for  saturation
	int  h_bins  =  50;  int  s_bins  =  60;
	int  histSize[]  =  {  h_bins,  s_bins  };
	//  hue  varies  from  0  to  179,  saturation  from  0  to  255
	float  h_ranges[]  =  {  0,  180  };
	float  s_ranges[]  =  {  0,  256  };
	const  float*  ranges[]  =  {  h_ranges,  s_ranges  };
	//  Use  the  o-th  and  1-st  channels
	int  channels[]  =  {  0,  1  };
	///  Histograms
	MatND  hist_base;
	MatND  hist_test1;
	///  Calculate  the  histograms  for  the  HSV  images
	calcHist(  &hsv_base,  1,  channels,  Mat(),  hist_base,  2,  histSize,  ranges,  true,  false  );
	normalize(  hist_base,  hist_base,  0,  1,  NORM_MINMAX,  -1,  Mat()  );
	calcHist(  &hsv_test1,  1,  channels,  Mat(),  hist_test1,  2,  histSize,  ranges,  true,  false  );
	normalize(  hist_test1,  hist_test1,  0,  1,  NORM_MINMAX,  -1,  Mat()  );
	///  Apply  the  histogram  comparison  methods
	double  base_test1  =  compareHist(  hist_base,  hist_test1,  0  );
	return base_test1;
}
#pragma endregion 图像操作

#pragma region 轮廓操作
	//寻找最大的轮廓
	VP FindBigestContour(Mat src){    
		int imax = 0; //代表最大轮廓的序号
		int imaxcontour = -1; //代表最大轮廓的大小
		std::vector<std::vector<cv::Point>>contours;    
		findContours(src,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
		for (int i=0;i<contours.size();i++){
			int itmp =  contourArea(contours[i]);//这里采用的是轮廓大小
			if (imaxcontour < itmp ){
				imax = i;
				imaxcontour = itmp;
			}
		}
		return contours[imax];
	}

	//寻找并绘制出彩色联通区域
	vector<VP> connection2(Mat src,Mat& draw){    
		draw = Mat::zeros(src.rows,src.cols,CV_8UC3);
		vector<VP>contours;    
		findContours(src.clone(),contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
		//由于给大的区域着色会覆盖小的区域，所以首先进行排序操作
		//冒泡排序，由小到大排序
		VP vptmp;
		for(int i=1;i<contours.size();i++){
			for(int j=contours.size()-1;j>=i;j--){
				if(contours[j].size()<contours[j-1].size()){	
					vptmp = contours[j-1];
					contours[j-1] = contours[j];
					contours[j] = vptmp;
				}
			}
		}
		//打印结果
		for (int i=contours.size()-1;i>=0;i--){
			Scalar  color  = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
			drawContours(draw,contours,i,color,-1);
		}
		return contours;
	}
	vector<VP> connection2(Mat src){
		Mat draw;
		return connection2(src,draw);
	}

	//根据轮廓的面积大小进行选择
	vector<VP>  selectShapeArea(Mat src,Mat& draw,vector<VP> contours,int minvalue,int maxvalue){
		vector<VP> result_contours;
		draw = Mat::zeros(src.rows,src.cols,CV_8UC3);
		for (int i=0;i<contours.size();i++){ 
			double countour_area = contourArea(contours[i]);
			if (countour_area >minvalue && countour_area<maxvalue)
				result_contours.push_back(contours[i]);
		}
		for (int i=0;i<result_contours.size();i++){
			int iRandB = rng.uniform(0,255);
			int iRandG = rng.uniform(0,255);
			int iRandR = rng.uniform(0,255);
			Scalar  color  = Scalar(iRandB,iRandG,iRandR);
			drawContours(draw,result_contours,i,color,-1);
			char cbuf[100];sprintf_s(cbuf,"%d",i+1);
			//寻找最小覆盖圆,求出圆心。使用反色打印轮廓项目
			float radius;
			cv::Point2f center;
			cv::minEnclosingCircle(result_contours[i],center,radius);
			putText(draw,cbuf,center, FONT_HERSHEY_PLAIN ,5,Scalar(255-iRandB,255-iRandG,255-iRandR),5);
		}
		return result_contours;
	}
	vector<VP>  selectShapeArea(vector<VP> contours,int minvalue,int maxvalue)
	{
		vector<VP> result_contours;
		for (int i=0;i<contours.size();i++){ 
			double countour_area = contourArea(contours[i]);
			if (countour_area >minvalue && countour_area<maxvalue)
				result_contours.push_back(contours[i]);
		}
		return result_contours;
	}

	//根据轮廓的圆的特性进行选择
	vector<VP> selectShapeCircularity(Mat src,Mat& draw,vector<VP> contours,float minvalue,float maxvalue){
		vector<VP> result_contours;
		draw = Mat::zeros(src.rows,src.cols,CV_8UC3);
		for (int i=0;i<contours.size();i++){
			float fcompare = calculateCircularity(contours[i]);
			if (fcompare >=minvalue && fcompare <=maxvalue)
				result_contours.push_back(contours[i]);
		}
		for (int i=0;i<result_contours.size();i++){
			Scalar  color  = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
			drawContours(draw,result_contours,i,color,-1);
		}
		return result_contours;
	}
	vector<VP> selectShapeCircularity(vector<VP> contours,float minvalue,float maxvalue){
		vector<VP> result_contours;
		for (int i=0;i<contours.size();i++){
			float fcompare = calculateCircularity(contours[i]);
			if (fcompare >=minvalue && fcompare <=maxvalue)
				result_contours.push_back(contours[i]);
		}
		return result_contours;
	}
	//计算轮廓的圆的特性
	float calculateCircularity(VP contour){
		Point2f center;
		float radius = 0;
		minEnclosingCircle((Mat)contour,center,radius);
		//以最小外接圆半径作为数学期望，计算轮廓上各点到圆心距离的标准差
		float fsum = 0;
		float fcompare = 0;
		for (int i=0;i<contour.size();i++){   
			Point2f ptmp = contour[i];
			float fdistenct = sqrt((float)((ptmp.x - center.x)*(ptmp.x - center.x)+(ptmp.y - center.y)*(ptmp.y-center.y)));
			float fdiff = abs(fdistenct - radius);
			fsum = fsum + fdiff;
		}
		fcompare = fsum/(float)contour.size();
		return fcompare;
	}

	//返回两点之间的距离
	float getDistance(Point2f f1,Point2f f2)
	{
		return sqrt((float)(f1.x - f2.x)*(f1.x - f2.x) + (f1.y -f2.y)*(f1.y- f2.y));
	}
#pragma endregion 轮廓操作

#pragma region 投影操作
	//投影到x或Y轴上,上波形为vup,下波形为vdown,gap为误差间隔
	void projection2(Mat src,vector<int>& vup,vector<int>& vdown,int direction,int gap){
		Mat tmp = src.clone();
		vector<int> vdate;
		if (DIRECTION_X == direction){
			for (int i=0;i<tmp.cols;i++){
				Mat data = tmp.col(i);
				int itmp = countNonZero(data);
				vdate.push_back(itmp);
			}
		}else{
			for (int i=0;i<tmp.rows;i++){
				Mat data = tmp.row(i);
				int itmp = countNonZero(data);
				vdate.push_back(itmp);
			}
		}
		//整形,去除长度小于gap的零的空洞
		if (vdate.size()<=gap)
			return;
		for (int i=0;i<vdate.size()-gap;i++){
			if (vdate[i]>0 && vdate[i+gap]>0){
				for (int j=i;j<i+gap;j++){
					vdate[j] = 1;
				}
				i = i+gap-1;
			}
		}
		//记录上下沿
		for (int i=1;i<vdate.size();i++){
			if (vdate[i-1] == 0 && vdate[i]>0)
				vup.push_back(i);
			if (vdate[i-1]>0 && vdate[i] == 0)
				vdown.push_back(i);
		}
	}

#pragma endregion 投影操作

#pragma region 文件操作
	//递归读取目录下全部文件
	void getFiles(string path, vector<string>& files,string flag){
		//文件句柄
		long   hFile   =   0;
		//文件信息
		struct _finddata_t fileinfo;
		string p;
		if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1){
			do{
				//如果是目录,迭代之,如果不是,加入列表
				if((fileinfo.attrib &  _A_SUBDIR)){
					if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0 && flag=="r")
						getFiles( p.assign(path).append("\\").append(fileinfo.name), files,flag );
				}
				else{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
				}
			}while(_findnext(hFile, &fileinfo)  == 0);
			_findclose(hFile);
		}
	}
	//递归读取目录下全部图片
	void getFiles(string path, vector<Mat>& files,string flag){
		vector<string> fileNames;
		getFiles(path,fileNames,flag);
		for (int i=0;i<fileNames.size();i++){
			Mat tmp = imread(fileNames[i]);
			if (tmp.rows>0)//如果是图片
				files.push_back(tmp);
		}
	}
	//递归读取目录下全部图片和名称
	void getFiles(string path, vector<pair<Mat,string>>& files,string flag){
		vector<string> fileNames;
		getFiles(path,fileNames,flag);
		for (int i=0;i<fileNames.size();i++){
			Mat tmp = imread(fileNames[i]);
			if (tmp.rows>0){
				pair<Mat,string> apir;
				apir.first = tmp;
				apir.second = fileNames[i];
				files.push_back(apir);
			}
		}
	}
	////删除目录下的全部文件
	void deleteFiles(string path,string flag){
		//文件句柄
		long   hFile   =   0;
		//文件信息
		struct _finddata_t fileinfo;
		string p;
		if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1){
			do{
				//如果是目录,迭代之,如果不是,加入列表
				if((fileinfo.attrib &  _A_SUBDIR)){
					if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0 && flag=="r")
						deleteFiles(p.assign(path).append("\\").append(fileinfo.name).c_str(),flag );
				}
				else{
					deleteFiles(p.assign(path).append("\\").append(fileinfo.name).c_str());
				}
			}while(_findnext(hFile, &fileinfo)  == 0);
			_findclose(hFile);
		}
	}
	//创建或续写目录下的csv文件,填写“文件位置-分类”对
	int writeCsv(const string& filename,const Vector<pair<string,string>>srcVect,char separator ){
		ofstream file(filename.c_str(),ofstream::app);
		if (!file)
			return 0;
		for (int i=0;i<srcVect.size();i++){
			file<<srcVect[i].first<<separator<<srcVect[i].second<<endl;
		}
		return srcVect.size();
	}
	//读取目录下的csv文件,获得“文件位置-分类”对
	vector<pair<string,string>> readCsv(const string& filename, char separator) {
		pair<string,string> apair;
		string line, path, classlabel;
		vector<pair<string,string>> retVect;
		ifstream file(filename.c_str(), ifstream::in);
		if (!file) 
			return retVect;
		while (getline(file, line)) {
			stringstream liness(line);
			getline(liness, path, separator);
			getline(liness, classlabel);
			if(!path.empty() && !classlabel.empty()) {
				apair.first = path;
				apair.second = classlabel;
				retVect.push_back(apair);
			}

		}
		return retVect;
	}

	 CString  GetInitString( CString Name1 ,CString Name2){
		char c[100] ;
		memset( c ,0 ,100) ;
		CString csCfgFilePath;
		GetModuleFileName(NULL, csCfgFilePath.GetBufferSetLength(MAX_PATH+1), MAX_PATH); 
		csCfgFilePath.ReleaseBuffer(); 
		int nPos = csCfgFilePath.ReverseFind ('\\');
		csCfgFilePath = csCfgFilePath.Left (nPos);
		csCfgFilePath += "\\Config" ;
		BOOL br = GetPrivateProfileString(Name1,Name2 ,"0",c, 100 , csCfgFilePath) ;
		CString rstr ;
		rstr .Format("%s" , c) ;
		return rstr ;
	}

	 void WriteInitString( CString Name1 ,CString Name2 ,CString strvalue){
		CString csCfgFilePath;
		GetModuleFileName(NULL, csCfgFilePath.GetBufferSetLength(MAX_PATH+1), MAX_PATH); 
		csCfgFilePath.ReleaseBuffer(); 
		int nPos = csCfgFilePath.ReverseFind ('\\');
		csCfgFilePath = csCfgFilePath.Left (nPos);
		csCfgFilePath += "\\Config" ;
		BOOL br = WritePrivateProfileString(Name1 ,Name2 ,strvalue ,csCfgFilePath) ;
		if ( !br)
			TRACE("savewrong") ;
	}

	//获得当前目录路径
	static CString GetLocalPath(){
		CString csCfgFilePath;
		GetModuleFileName(NULL, csCfgFilePath.GetBufferSetLength(MAX_PATH+1), MAX_PATH); 
		csCfgFilePath.ReleaseBuffer(); 
		int nPos = csCfgFilePath.ReverseFind ('\\');
		csCfgFilePath = csCfgFilePath.Left (nPos);
		return csCfgFilePath;
	}

	//获得.exe路径
	static CString GetExePath()
	{
		CString strPath;
		GetModuleFileName(NULL,strPath.GetBufferSetLength(MAX_PATH+1),MAX_PATH);
		strPath.ReleaseBuffer();
		return strPath;
	}

	//开机自动运行
	static BOOL SetAutoRun(CString strPath,bool flag)
	{
		CString str;
		HKEY hRegKey;
		BOOL bResult;
		str=_T("Software\\Microsoft\\Windows\\CurrentVersion\\Run");
		if(RegOpenKey(HKEY_LOCAL_MACHINE, str, &hRegKey) != ERROR_SUCCESS) 
			bResult=FALSE;
		else
		{
			_splitpath(strPath.GetBuffer(0),NULL,NULL,str.GetBufferSetLength(MAX_PATH+1),NULL);
			strPath.ReleaseBuffer();
			str.ReleaseBuffer();//str是键的名字
			if (flag){
				if(::RegSetValueEx( hRegKey,str,0,REG_SZ,(CONST BYTE *)strPath.GetBuffer(0),strPath.GetLength() ) != ERROR_SUCCESS)
					bResult=FALSE;
				else
					bResult=TRUE;
			}else{
				if(	::RegDeleteValue(hRegKey,str) != ERROR_SUCCESS)
					bResult=FALSE;
				else
					bResult=TRUE;
			}
			strPath.ReleaseBuffer();
		}
		return bResult;
	}		


#pragma endregion 文件操作

#pragma region 其他操作
	//string替换
	void string_replace(string & strBig, const string & strsrc, const string &strdst)
	{
		string::size_type pos=0;
		string::size_type srclen=strsrc.size();
		string::size_type dstlen=strdst.size();
		while( (pos=strBig.find(strsrc, pos)) != string::npos)
		{
			strBig.replace(pos, srclen, strdst);
			pos += dstlen;
		}
	}

	//C++的spilt函数
	void SplitString(const string& s, vector<string>& v, const string& c){
		std::string::size_type pos1, pos2;
		pos2 = s.find(c);
		pos1 = 0;
		while(std::string::npos != pos2){
			v.push_back(s.substr(pos1, pos2-pos1));
			pos1 = pos2 + c.size();
			pos2 = s.find(c, pos1);
		}
		if(pos1 != s.length())
			v.push_back(s.substr(pos1));
	}
	//! 通过文件夹名称获取文件名，不包括后缀
	void getFileName(const string& filepath, string& name,string& lastname){
		vector<string> spilt_path;
		SplitString(filepath, spilt_path, "\\");
		int spiltsize = spilt_path.size();
		string filename = "";
		if (spiltsize != 0){
			filename = spilt_path[spiltsize-1];
			vector<string> spilt_name;
			SplitString(filename, spilt_name, ".");
			int name_size = spilt_name.size();
			if (name_size != 0)
				name = spilt_name[0];
			lastname = spilt_name[name_size-1];
		}
	}
#pragma endregion 其他操作

#pragma region excel操作
	//////////////////////////////////////////////////////////////////////////////
	//名称：GetExcelDriver
	//功能：获取ODBC中Excel驱动
	//作者：徐景周(jingzhou_xu@163.net)
	//组织：未来工作室(Future Studio)
	//日期：2002.9.1
	/////////////////////////////////////////////////////////////////////////////
	CString GetExcelDriver()
	{
		char szBuf[2001];
		WORD cbBufMax = 2000;
		WORD cbBufOut;
		char *pszBuf = szBuf;
		CString sDriver;

		// 获取已安装驱动的名称(涵数在odbcinst.h里)
		if (!SQLGetInstalledDrivers(szBuf, cbBufMax, &cbBufOut))
			return "";

		// 检索已安装的驱动是否有Excel...
		do
		{
			if (strstr(pszBuf, "Excel") != 0)
			{
				//发现 !
				sDriver = CString(pszBuf);
				break;
			}
			pszBuf = strchr(pszBuf, '\0') + 1;
		}
		while (pszBuf[1] != '\0');

		return sDriver;
	}

	///////////////////////////////////////////////////////////////////////////////
	//	BOOL MakeSurePathExists( CString &Path,bool FilenameIncluded)
	//	参数：
	//		Path				路径
	//		FilenameIncluded	路径是否包含文件名
	//	返回值:
	//		文件是否存在
	//	说明:
	//		判断Path文件(FilenameIncluded=true)是否存在,存在返回TURE，不存在返回FALSE
	//		自动创建目录
	//
	///////////////////////////////////////////////////////////////////////////////
	BOOL MakeSurePathExists( CString &Path,bool FilenameIncluded)
	{
		int Pos=0;
		while((Pos=Path.Find('\\',Pos+1))!=-1)
			CreateDirectory(Path.Left(Pos),NULL);
		if(!FilenameIncluded)
			CreateDirectory(Path,NULL);
		//	return ((!FilenameIncluded)?!_access(Path,0):
		//	!_access(Path.Left(Path.ReverseFind('\\')),0));

		return !_access(Path,0);
	}

	//获得默认的文件名
	BOOL GetDefaultXlsFileName(CString& sExcelFile)
	{
		///默认文件名：yyyymmddhhmmss.xls
		CString timeStr;
		CTime day;
		day=CTime::GetCurrentTime();
		int filenameday,filenamemonth,filenameyear,filehour,filemin,filesec;
		filenameday=day.GetDay();//dd
		filenamemonth=day.GetMonth();//mm月份
		filenameyear=day.GetYear();//yyyy
		filehour=day.GetHour();//hh
		filemin=day.GetMinute();//mm分钟
		filesec=day.GetSecond();//ss
		timeStr.Format("%04d%02d%02d%02d%02d%02d",filenameyear,filenamemonth,filenameday,filehour,filemin,filesec);
		sExcelFile =  timeStr + ".xls"; //获取随机时间的文件名称
		//打开选择路径窗口
		CString pathName; 
		CString defaultDir = _T("C:\\outtest");
		CString fileName=sExcelFile;
		CString szFilters= _T("xls(*.xls)");
		CFileDialog dlg(FALSE,defaultDir,fileName,OFN_HIDEREADONLY|OFN_READONLY,szFilters,NULL);
		if(dlg.DoModal()==IDOK){
			//获得保存位置
			pathName = dlg.GetPathName();
		}

		sExcelFile = pathName;
		return TRUE;
	}

	///////////////////////////////////////////////////////////////////////////////
	//	void GetExcelDriver(CListCtrl* pList, CString strTitle)
	//	参数：
	//		pList		需要导出的List控件指针
	//		strTitle	导出的数据表标题
	//	说明:
	//		导出CListCtrl控件的全部数据到Excel文件。Excel文件名由用户通过“另存为”
	//		对话框输入指定。创建名为strTitle的工作表，将List控件内的所有数据（包括
	//		列名和数据项）以文本的形式保存到Excel工作表中。保持行列关系。
	//	
	//	edit by [r]@dotlive.cnblogs.com
	//  2016年8月12日 修改为可以保存多个表的模式
	///////////////////////////////////////////////////////////////////////////////
	CString ExportListToExcel(CString  sExcelFile,CListCtrl* pList, CString strTitle)
	{
		CString warningStr;
		if (pList->GetItemCount ()>0) {	
			CDatabase database;
			
			
			CString sSql;
			CString tableName = strTitle;

			// 检索是否安装有Excel驱动 "Microsoft Excel Driver (*.xls)" 
			CString sDriver;
			sDriver = GetExcelDriver();
			if (sDriver.IsEmpty())
			{
				// 没有发现Excel驱动
				AfxMessageBox("没有安装Excel!\n请先安装Excel软件才能使用导出功能!");
				return NULL;
			}

			///默认文件名
		/*	CString sExcelFile; 
			if (!GetDefaultXlsFileName(sExcelFile))
				return NULL;*/

			// 创建进行存取的字符串
			sSql.Format("DRIVER={%s};DSN='';FIRSTROWHASNAMES=1;READONLY=FALSE;CREATE_DB=\"%s\";DBQ=%s",sDriver, sExcelFile, sExcelFile);

			// 创建数据库 (既Excel表格文件)
			if( database.OpenEx(sSql,CDatabase::noOdbcDialog) )
			{
				// 创建表结构
				int i;
				LVCOLUMN columnData;
				CString columnName;
				int columnNum = 0;
				CString strH;
				CString strV;

				sSql = "";
				strH = "";
				columnData.mask = LVCF_TEXT;
				columnData.cchTextMax =100;
				columnData.pszText = columnName.GetBuffer (100);
				for(i=0;pList->GetColumn(i,&columnData);i++)
				{
					if (i!=0)
					{
						sSql = sSql + ", " ;
						strH = strH + ", " ;
					}
					sSql = sSql + " " + columnData.pszText +" TEXT";
					strH = strH + " " + columnData.pszText +" ";
				}
				columnName.ReleaseBuffer ();
				columnNum = i;

				sSql = "CREATE TABLE " + tableName + " ( " + sSql +  " ) ";
				database.ExecuteSQL(sSql);


				// 插入数据项
				int nItemIndex;
				for (nItemIndex=0;nItemIndex<pList->GetItemCount ();nItemIndex++){
					strV = "";
					for(i=0;i<columnNum;i++)
					{
						if (i!=0)
						{
							strV = strV + ", " ;
						}
						strV = strV + " '" + pList->GetItemText(nItemIndex,i) +"' ";
					}

					sSql = "INSERT INTO "+ tableName 
						+" ("+ strH + ")"
						+" VALUES("+ strV + ")";
					database.ExecuteSQL(sSql);
				}

			}      

			// 关闭数据库
			database.Close();
			return sExcelFile;
		}
	}
	//2个datesheet的模式
	CString ExportListToExcel(CListCtrl* pList, CString strTitle,CListCtrl* pList2,CString strTitle2)
	{
		CString warningStr;
		if (pList->GetItemCount ()>0) {	
			CDatabase database;
			CString sDriver;
			CString sExcelFile; 
			CString sSql;
			CString tableName = strTitle;
			CString tableName2 = strTitle2;

			// 检索是否安装有Excel驱动 "Microsoft Excel Driver (*.xls)" 
			sDriver = GetExcelDriver();
			if (sDriver.IsEmpty())
			{
				// 没有发现Excel驱动
				AfxMessageBox("没有安装Excel!\n请先安装Excel软件才能使用导出功能!");
				return NULL;
			}

			///默认文件名
			if (!GetDefaultXlsFileName(sExcelFile))
				return NULL;

			// 创建进行存取的字符串
			sSql.Format("DRIVER={%s};DSN='';FIRSTROWHASNAMES=1;READONLY=FALSE;CREATE_DB=\"%s\";DBQ=%s",sDriver, sExcelFile, sExcelFile);

			// 创建数据库 (既Excel表格文件)
			if( database.OpenEx(sSql,CDatabase::noOdbcDialog) )
			{
				// 创建表结构1
				int i;
				LVCOLUMN columnData;
				LVCOLUMN columnData2;
				CString columnName;
				CString columnName2;
				int columnNum = 0;
				CString strH;
				CString strV;

				sSql = "";
				strH = "";
				columnData.mask = LVCF_TEXT;
				columnData.cchTextMax =100;
				columnData.pszText = columnName.GetBuffer (100);
				columnData2.mask = LVCF_TEXT;
				columnData2.cchTextMax =100;
				columnData2.pszText = columnName2.GetBuffer (100);
				// 插入数据项1
				for(i=0;pList->GetColumn(i,&columnData);i++)
				{
					if (i!=0)
					{
						sSql = sSql + ", " ;
						strH = strH + ", " ;
					}
					sSql = sSql + " " + columnData.pszText +" TEXT";
					strH = strH + " " + columnData.pszText +" ";
				}
				columnName.ReleaseBuffer ();
				columnNum = i;

				sSql = "CREATE TABLE " + tableName + " ( " + sSql +  " ) ";
				database.ExecuteSQL(sSql);

				
				int nItemIndex;
				for (nItemIndex=0;nItemIndex<pList->GetItemCount();nItemIndex++){
					strV = "";
					for(i=0;i<columnNum;i++)
					{
						if (i!=0)
						{
							strV = strV + ", " ;
						}
						strV = strV + " '" + pList->GetItemText(nItemIndex,i) +"' ";
					}

					sSql = "INSERT INTO "+ tableName 
						+" ("+ strH + ")"
						+" VALUES("+ strV + ")";
					database.ExecuteSQL(sSql);
				}
				//插入数据项2
				sSql = "";
				strH="";
				int columnNum2 = 0;
			
				for(int i=0;pList2->GetColumn(i,&columnData2);i++)
				{
					if (i!=0)
					{
						sSql = sSql + ", " ;
						strH = strH + ", " ;
					}
					sSql = sSql + " " + columnData2.pszText +" TEXT";
					strH = strH + " " + columnData2.pszText +" ";
				}
				columnName2.ReleaseBuffer ();
				columnNum2 = i;

				sSql = "CREATE TABLE " + tableName2 + " ( " + sSql +  " ) ";
				database.ExecuteSQL(sSql);
			 
				for (nItemIndex=0;nItemIndex<pList2->GetItemCount ();nItemIndex++){
					strV = "";
					for(i=0;i<columnNum2;i++)
					{
						if (i!=0)
						{
							strV = strV + ", " ;
						}
						strV = strV + " '" + pList2->GetItemText(nItemIndex,i) +"' ";
					}

					sSql = "INSERT INTO "+ tableName2 
						+" ("+ strH + ")"
						+" VALUES("+ strV + ")";
					database.ExecuteSQL(sSql);
				}
			}      
			// 关闭数据库
			database.Close();

			warningStr.Format("导出文件保存于%s!",sExcelFile);
			AfxMessageBox(warningStr);
			return sExcelFile;
		}
	}
#pragma endregion

}