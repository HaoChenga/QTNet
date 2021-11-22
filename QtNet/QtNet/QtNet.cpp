#include "QtNet.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include <QJsonObject> 
#include <QJsonArray>
#include <QString>
#include <QByteArray>
#include <QJsonDocument>
#include <Qobject.h>

#include <map>
#include <algorithm>
#include <iterator>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>

#define CV_IMWRITE_JPEG_QUALITY 1
#define  SEVER_AIMARK "http://122.224.127.158:6060/facial_analysis" 
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
QtNet::QtNet(QWidget *parent)
    : QMainWindow(parent),
	ui(new Ui::QtNetClass)

{
    ui->setupUi(this);
}
QtNet::~QtNet()
{
	delete ui;
}
void QtNet::on_selectPhoto_clicked() {
	QString  path = QFileDialog::getOpenFileName(this,
		tr("Open Image"), "../data", tr("Image Files (*.png *.jpg *.bmp)"));
	if (!path.isEmpty())
	{
		strPath = path.toLocal8Bit().toStdString();
		directoryPath = strPath.substr(0, strPath.rfind('/'));
		testPhoto = cv::imread(strPath);
		photoTrans=contrarotatePhoto(testPhoto);
		DisplayMat(testPhoto);
		facePoints.clear();
		//std::cout << strPath << std::endl;
	}
}
void QtNet::on_sendInfor_clicked()
{
	sendPhotoAndGetReply();
}

void QtNet::DisplayMat(cv::Mat image)
{
	cv::Mat rgb;
	QImage img;
	if (image.channels() == 3)
	{
		cv::cvtColor(image, rgb, CV_BGR2RGB);
		img = QImage((const unsigned char*)(rgb.data),
			rgb.cols, rgb.rows, rgb.cols*rgb.channels(),//rgb.cols*rgb.channels()�����滻Ϊimage.step
			QImage::Format_RGB888);
	}
	else
	{
		img = QImage((const unsigned char*)(image.data),
			image.cols, image.rows, rgb.cols*image.channels(),
			QImage::Format_RGB888);
	} 
	ui->displayMat->setPixmap(QPixmap::fromImage(img).scaled(ui->displayMat->size()));//setPixelmap(QPixmap::fromImage(img));
	ui->displayMat->resize(ui->displayMat->pixmap()->size());//resize(ui->label->pixmap()->size());
}

void QtNet::on_drawPoints_clicked()
{
	if (!facePoints.empty())
	{
		std::map<std::string, cv::Point2d>::iterator iter;
		for (iter = facePoints.begin();iter !=facePoints.end();iter++)
		{
			cv::Point2d point = iter->second;
			std::string name = iter->first;
			cv::circle(testPhoto, point, 3, cv::Scalar(0, 0, 255), -1);
			cv::putText(testPhoto, name, point, cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(0,0,255));
		}
		std::string savePath = directoryPath + "/result.jpg";
		cv::imwrite(savePath, testPhoto);
		cv::Mat rgb;
		cv::cvtColor(testPhoto, rgb, CV_BGR2RGB);
		QImage img = QImage((const unsigned char*)(rgb.data),
			rgb.cols, rgb.rows, rgb.cols*rgb.channels(),
			QImage::Format_RGB888);
		ui->displayMat_2->setPixmap(QPixmap::fromImage(img).scaled(ui->displayMat_2->size()));
		ui->displayMat_2->resize(ui->displayMat_2->pixmap()->size());
	}
	else
	{
		QMessageBox::information(this, QString::fromLocal8Bit("��ʾ"), QString::fromLocal8Bit("���������ݽ���ʧ�ܣ��޷����ƣ�"));
	}
}

void QtNet::on_finishMesh_clicked()
{
	PointCloud::Ptr centreFace(new PointCloud);
	std::string plyPath = directoryPath + "/RGBL.ply";
	std::string pcdSavePath = directoryPath + "/centreface.pcd";
	pcl::io::loadPLYFile(plyPath,*centreFace);
	//��������άģ���ϱ������м�������
	PointT minPt, maxPt;
	pcl::getMinMax3D(*centreFace, minPt, maxPt);
	pcl::PointCloud<PointT>::Ptr cloud_filter(new pcl::PointCloud<PointT>);
	pcl::PassThrough<PointT> pass;     //�����˲�������
	pass.setInputCloud(centreFace);                //���ô��˲��ĵ���
	pass.setFilterFieldName("z");             //������Z�᷽���Ͻ����˲�
	pass.setFilterLimits(maxPt.z - 0.08, maxPt.z);    //�����˲���Χ
	pass.setFilterLimitsNegative(false);      //����
	pass.filter(*cloud_filter);               //�˲����洢
	cloud_filter->height = 1;
	cloud_filter->width = cloud_filter->points.size();
	//pcl::io::savePCDFile(pcdSavePath, *cloud_filter);

	std::string depthPath = directoryPath +"/depthC.png";
	cv::Mat depth=cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
	std::string inCamera = "../data/realsen2_cameraData.yml";
	//�����ö�ά�ؼ���תΪ��ά�㼯����ʱ����ά�㻹����������άģ���ϵ���ά��
	std::vector<cv::Point3f> _3dpoints = Compute2dto3d(depth, inCamera, facePoints);
	//�ָ�������ά��������һ����ת����Ϊ���յ���άģ��Ϊ��������ʾ������һ����ת
	pcl::PointCloud<PointT>::Ptr _3dcloud(new pcl::PointCloud<PointT>);
	Eigen::Matrix4f RT;
	RT << 0, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1;
	for (int i = 0; i<_3dpoints.size(); i++)
	{
		Eigen::Vector4f point(_3dpoints[i].x, _3dpoints[i].y, _3dpoints[i].z, 1);
		Eigen::Vector4f result = RT*point;
		pcl::PointXYZRGB tempPoint;
		tempPoint.x = result(0, 0) / result(3, 0);
		tempPoint.y = result(1, 0) / result(3, 0);
		tempPoint.z = result(2, 0) / result(3, 0);// +0.001;
		tempPoint.r = 255;
		tempPoint.g = 0;
		tempPoint.b = 0;
		_3dcloud->points.push_back(tempPoint);
	}
	_3dcloud->width = _3dcloud->points.size();
	_3dcloud->height = 1;
	//���ָ���������ά���������м������򣬴�ʱ����ά��������յ���ά����
	//resultPoints �������յ���ά�㼯
	//_3dcloud�ǽ���ά�㼯��ֵ���˵��ƣ�����ply������ʾ
	pcl::KdTreeFLANN<PointT> kdtreeNearst;
	kdtreeNearst.setInputCloud(cloud_filter);
	int nums = 1;
	std::vector<cv::Point3d> resultPoints;
	//resultPoints.resize(_3dcloud->points.size());
	for (int i = 0; i < _3dcloud->points.size();i++)
	{
		std::vector<int> pointIdxNKNSearch(nums);      //�洢��ѯ���������
		std::vector<float> pointNKNSquaredDistance(nums); //�洢���ڵ��Ӧ����ƽ��
		PointT &_3dpoint = _3dcloud->points[i];
		if ((kdtreeNearst.nearestKSearch(_3dpoint, nums, pointIdxNKNSearch, pointNKNSquaredDistance)) > 0)
		{
			cv::Point3d cvPoint;
			int index = pointIdxNKNSearch[0];
			//���㸳ֵ�����ƣ�����ply
			_3dpoint.x = cloud_filter->points[index].x;
			_3dpoint.y = cloud_filter->points[index].y;
			_3dpoint.z = cloud_filter->points[index].z;
			//���㸳ֵ��cv::Pointe3d����
			cvPoint.x = cloud_filter->points[index].x;
			cvPoint.y = cloud_filter->points[index].y;
			cvPoint.z = cloud_filter->points[index].z;
			resultPoints.push_back(cvPoint);
		}
	}
	std::string savePath = directoryPath + "/3d.ply";
	pcl::io::savePLYFile(savePath, *_3dcloud);
	QMessageBox::information(this, QString::fromLocal8Bit("��ʾ"), QString::fromLocal8Bit("��ά����Ƴɹ����Ѵ洢��data�ļ����У�"));
}

//void QtNet::slot_LoadImage()
//{
//	testPhoto = cv::imread("test.jpg");
//}
std::string base64Encode(const unsigned char * Data, int DataByte)
{
	const char EncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	std::string strEncode;
	unsigned char Tmp[4] = { 0 };
	int lineLength = 0;
	for (int i = 0; i < (int)(DataByte / 3); i++)
	{
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		Tmp[3] = *Data++;
		strEncode += EncodeTable[Tmp[1] >> 2];
		strEncode += EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
		strEncode += EncodeTable[((Tmp[2] << 2) | (Tmp[2] >> 6)) & 0x3F];
		strEncode += EncodeTable[Tmp[3] & 0x3f];
		if (lineLength += 4, lineLength == 76)
		{
			strEncode += '\r\n';
			lineLength = 0;
		}
	}
	int Mod = DataByte % 3;
	if (Mod == 1)
	{
		Tmp[1] = *Data++;
		strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode += EncodeTable[((Tmp[1] & 0x03) << 4)];
		strEncode += "==";
	}
	else if (Mod == 2)
	{
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode += EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
		strEncode += EncodeTable[((Tmp[2] & 0x0F) << 2)];
		strEncode += "==";
	}
	return strEncode;
}
std::string Mat2Base64(const cv::Mat &img, std::string imgType)
{

	std::string img_data;
	std::vector<uchar> VecImg;
	std::vector<int> vecCompression_params;
	vecCompression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	vecCompression_params.push_back(90);
	imgType = "." + imgType;
	cv::imencode(imgType, img, VecImg, vecCompression_params);
	img_data = base64Encode(VecImg.data(), VecImg.size());
	return img_data;
}

bool QtNet::slot_recUploadImage(QNetworkReply *reply)
{
	disconnect(m_Mannger, &QNetworkAccessManager::finished, this, &QtNet::slot_recUploadImage);
	QByteArray data = reply->readAll();
	if (!data.isEmpty())
	{
		std::list<QString> nameList
		{"Tr","G","Sn","Me","Sto","HairR","ExR","EnR","EnL",
		"ExL","HairL","Prn","ZyR","ZyL","FtR","FtL","U1","L1",
		"ChR","ChL","GoR","GoL","LrR","LrL","PuR","PuL","A1R","A1L"};
		QList<QByteArray> dataList = data.split('@');
		if (QString(dataList.first())=="0")
		{
			QMessageBox::information(this, QString::fromLocal8Bit("��ʾ"), QString::fromLocal8Bit("���������ݽ���ʧ�ܣ�"));
			return false;
		}
		if (QString(dataList.first()) == "1")
		{
			QJsonParseError doc_err;
			QJsonDocument doc = QJsonDocument::fromJson(dataList.last(), &doc_err);
			if (doc_err.error == QJsonParseError::NoError)
			{
				if (doc.isArray())
				{
					QJsonArray array = doc.array();
					QJsonObject obj = array.at(0).toObject();
					facePoints.clear();
					for each (QString name in nameList)
					{
						if (obj.contains(name))
						{
							double x = obj.value(name).toObject().value("x").toString().toDouble();
							double y = obj.value(name).toObject().value("y").toString().toDouble();
							//Eigen::Matrix3i transMatrix;
							//���Ĺؼ�������ʱ����תͼ���ϵ����꣬��˻��ý�����ӳ�䵽ԭͼ����ȥ
							cv::Mat transMatrix = (cv::Mat_<double>(3, 3) << 0, -1, 1280,1, 0, 0,0, 0, 1);
							cv::Mat srcPoint = (cv::Mat_<double>(3, 1) << x, y, 1);
							cv::Mat tgtPoint = transMatrix*srcPoint;
							x = tgtPoint.at<double>(0, 0);
							y = tgtPoint.at<double>(1, 0);
							facePoints.insert(std::map<std::string, cv::Point2f>::value_type(name.toLocal8Bit().toStdString(), cv::Point2f(x, y)));
						}
					}
				}
				QMessageBox::information(this, QString::fromLocal8Bit("��ʾ"), QString::fromLocal8Bit("���������ݽ����ɹ���"));
				return true;
			}
		}

		if (QString(dataList.first()) != "1"&&QString(dataList.first())!="0")
		{
			QMessageBox::information(this, QString::fromLocal8Bit("��ʾ"), QString::fromLocal8Bit("���������ݽ���ʧ�ܣ�"));
			return false;
		}
	}
	else
	{
		QMessageBox::information(this, QString::fromLocal8Bit("��ʾ"), QString::fromLocal8Bit("���������ݽ���ʧ�ܣ�"));
		return false;
	}
}
void QtNet::sendPhotoAndGetReply()
{
	std::string str_decode = Mat2Base64(photoTrans, "bmp");
	QJsonObject obj;
	obj.insert("image", QString::fromStdString(str_decode).trimmed());
	QByteArray data = QJsonDocument(obj).toJson();
	QNetworkRequest request;
	QString url = QString(SEVER_AIMARK);
	request.setUrl(QUrl(url));
	request.setRawHeader("Content_Type", "application/json");
	//slot_recUploadImage()
	m_Mannger = new QNetworkAccessManager;
	connect(m_Mannger, &QNetworkAccessManager::finished, this, &QtNet::slot_recUploadImage);
	m_Mannger->post(request, data);
}

cv::Mat QtNet::contrarotatePhoto(const cv::Mat &srcImage)
{
	cv::Mat photoTemp, tgtPhoto;
	transpose(srcImage, photoTemp);
	cv::flip(photoTemp, tgtPhoto, 0);
	return tgtPhoto;
}
cv::Point3d augY(int x,int y,int yMax,const cv::Mat &depth)
{
	for (int i=y;i<yMax;i++)
	{
		ushort _2ddepth  = depth.ptr<ushort>(i)[x];
		if (_2ddepth!=0&& _2ddepth<1000)
		{
			cv::Point3d searchPoint(x, i, _2ddepth);
			return searchPoint;
		}
	}
		return cv::Point3d();

}
cv::Point3d subY(int x, int y, int ymin, const cv::Mat &depth)
{
	for (int i = y; i > ymin; i--)
	{
		ushort _2ddepth = depth.ptr<ushort>(i)[x];
		if (_2ddepth != 0 && _2ddepth<1000)
		{
			cv::Point3d searchPoint(x, i, _2ddepth);
			return searchPoint;
		}
	}
		return cv::Point3d();

}
cv::Point3d augX(int x, int y, int xMax, const cv::Mat &depth)
{
	for (int i = x; i < xMax; i++)
	{
		ushort _2ddepth = depth.ptr<ushort>(y)[i];
		if (_2ddepth != 0 && _2ddepth<1000)
		{
			cv::Point3d searchPoint(i, y, _2ddepth);
			return searchPoint;
		}
	}
		return cv::Point3d();

}
cv::Point3d subX(int x, int y, int xMin, const cv::Mat &depth)
{
	for (int i = x; i > xMin; i--)
	{
		ushort _2ddepth = depth.ptr<ushort>(y)[i];
		if (_2ddepth != 0 && _2ddepth<1000)
		{
			cv::Point3d searchPoint(i, y, _2ddepth);
			return searchPoint;
		}
	}
		return cv::Point3d();
}
//����ά��ӳ�䵽��άģ���ϵ���ά��
std::vector<cv::Point3f> QtNet::Compute2dto3d(const cv::Mat & depth, const std::string & InCamera,  std::map<std::string, cv::Point2d>& _2dpoints)
{
	assert(_2dpoints.size() > 0);
	std::vector<cv::Point3f>_3dpoints;
	cv::Mat cameraMatrix, distCoef;
	cv::FileStorage fs(InCamera, cv::FileStorage::READ);
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoef;
	fs.release();
	float U0 = cameraMatrix.at<double>(0, 2);
	float V0 = cameraMatrix.at<double>(1, 2);
	float fX = cameraMatrix.at<double>(0, 0);
	float fY = cameraMatrix.at<double>(1, 1);
	std::map<std::string, cv::Point2d>::iterator it;
	for (it = _2dpoints.begin(); it!=_2dpoints.end(); it++)
	{
		cv::Point2i _2dpoint = it->second;
		ushort _2ddepth = depth.ptr<ushort>(_2dpoint.y)[_2dpoint.x];
		if (_2ddepth == 0||_2ddepth>=1000)
		{
			std::vector<std::string> outPoints{ "GoR","ZyR","HairR","FtR","GoL","ZyL","HairL","FtL","Tr","Me" };
			std::string pointName = it->first;
			std::vector<std::string>::iterator result = std::find(outPoints.begin(), outPoints.end(), pointName); //��������ʧ�ܵĵ�
			if (result != outPoints.end()) //��ǰ�������׳���ĵ�
			{
				int index = std::distance(outPoints.begin(), result);
				cv::Point3f _3dpoint;
				switch (index)
				{
				case  0:
				case  1:
				case  2:
				case  3:
					 _3dpoint = augY(_2dpoint.x, _2dpoint.y, depth.rows, depth);
					_3dpoint.z = double(_3dpoint.z) / 1000;
					_3dpoint.x = (_3dpoint.x - U0) * (_3dpoint.z) / fX;
					_3dpoint.y = (_3dpoint.y - V0) * (_3dpoint.z) / fY;
					_3dpoints.push_back(_3dpoint);
					break;
				case  4:
				case  5:
				case  6:
				case  7:
					_3dpoint = subY(_2dpoint.x, _2dpoint.y, 0, depth);
					_3dpoint.z = double(_3dpoint.z) / 1000;
					_3dpoint.x = (_3dpoint.x - U0) * (_3dpoint.z) / fX;
					_3dpoint.y = (_3dpoint.y - V0) * (_3dpoint.z) / fY;
					_3dpoints.push_back(_3dpoint);
					break;
				case  8:
					_3dpoint = subX(_2dpoint.x, _2dpoint.y, 0, depth);
					_3dpoint.z = double(_3dpoint.z) / 1000;
					_3dpoint.x = (_3dpoint.x - U0) * (_3dpoint.z) / fX;
					_3dpoint.y = (_3dpoint.y - V0) * (_3dpoint.z) / fY;
					_3dpoints.push_back(_3dpoint);
					break;
				case  9:
					_3dpoint = augX(_2dpoint.x, _2dpoint.y, depth.cols, depth);
					_3dpoint.z = double(_3dpoint.z) / 1000;
					_3dpoint.x = (_3dpoint.x - U0) * (_3dpoint.z) / fX;
					_3dpoint.y = (_3dpoint.y - V0) * (_3dpoint.z) / fY;
					_3dpoints.push_back(_3dpoint);
					break;

				}
			}
			else
			{
				int y = _2dpoint.y;
				int x = _2dpoint.x;
				int radius = 6, count = 0;
				double sum = 0;
				for (int i = y - radius; i < y + radius; i++)
				{
					for (int j = x - radius; j < x + radius; j++)
					{
						if (i > 0 && i < depth.rows && j>0 && j < depth.cols)
						{
							ushort depthTemp = depth.ptr<ushort>(i)[j];
							if (depthTemp != 0)
							{
								sum += depthTemp;
								count++;
							}
						}
					}
				}
				sum /= count;
				_2ddepth = sum;
				cv::Point3f _3dpoint;
				_3dpoint.z = double(_2ddepth) / 1000;
				_3dpoint.x = (_2dpoint.x - U0) * (_3dpoint.z) / fX;
				_3dpoint.y = (_2dpoint.y - V0) * (_3dpoint.z) / fY;
				_3dpoints.push_back(_3dpoint);

			}
			
		}
		else
		{
			cv::Point3f _3dpoint;
			_3dpoint.z = double(_2ddepth) / 1000;
			_3dpoint.x = (_2dpoint.x - U0) * (_3dpoint.z) / fX;
			_3dpoint.y = (_2dpoint.y - V0) * (_3dpoint.z) / fY;
			_3dpoints.push_back(_3dpoint);

		}
	}
	return _3dpoints;
}
