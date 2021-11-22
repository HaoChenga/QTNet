#pragma once
#include <QtWidgets/QMainWindow>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include "ui_QtNet.h"
#include "opencv2/core.hpp"
#include<iostream>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>
#include <map>
class QtNet : public QMainWindow
{
    Q_OBJECT

public:
    QtNet(QWidget *parent = Q_NULLPTR);
	~QtNet();
	std::map<std::string, cv::Point2d> getFacePoints() const {
		if (!facePoints.empty())  return facePoints;
	}
private:
    Ui::QtNetClass *ui;
	cv::Mat testPhoto, photoTrans;//ԭͼ����ʱ����ת֮���ͼ
	QNetworkAccessManager* m_Mannger;
	std::string strPath;
	std::string directoryPath;
	std::map<std::string, cv::Point2d> facePoints;
	//void slot_LoadImage();
	void sendPhotoAndGetReply();
	cv::Mat contrarotatePhoto(const cv::Mat &srcImage);//��������ͼ�����Ǻ�����ģ���Ҫ��ʱ����ת�����������ؼ�����
	std::vector<cv::Point3f> Compute2dto3d
	(const cv::Mat &depth, const std::string &InCamera, std::map<std::string, cv::Point2d> &_2dpoints);
private slots:
	//
bool slot_recUploadImage(QNetworkReply *reply);
void on_selectPhoto_clicked();
void on_sendInfor_clicked();
void DisplayMat(cv::Mat image);
void on_drawPoints_clicked();
void on_finishMesh_clicked();
};
