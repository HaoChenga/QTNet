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
	cv::Mat testPhoto, photoTrans;//原图和逆时针旋转之后的图
	QNetworkAccessManager* m_Mannger;
	std::string strPath;
	std::string directoryPath;
	std::map<std::string, cv::Point2d> facePoints;
	//void slot_LoadImage();
	void sendPhotoAndGetReply();
	cv::Mat contrarotatePhoto(const cv::Mat &srcImage);//相机拍摄的图像人是横过来的，需要逆时针旋转才能正常做关键点检测
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
