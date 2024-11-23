//
// Created by lan on 17-12-26.
//
#include "LSDmatcher.h"

#define PI 3.1415926

using namespace std;

namespace ORB_SLAM2
{
    const int LSDmatcher::TH_HIGH = 80;
    const int LSDmatcher::TH_LOW = 50;
    const int LSDmatcher::HISTO_LENGTH = 30;

    LSDmatcher::LSDmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    double LSDmatcher::computeAngle2D(const cv::Mat &vector1, const cv::Mat &vector2)
    {
        // Compute the angle between two vectors
        double dot_product = vector1.dot(vector2);

        // Find magnitude of line AB and BC
        double magnitudeAB = std::sqrt(vector1.at<double>(0) * vector1.at<double>(0) +
                                       vector1.at<double>(1) * vector1.at<double>(1));

        double magnitudeBC = std::sqrt(vector2.at<double>(0) * vector2.at<double>(0) +
                                       vector2.at<double>(1) * vector2.at<double>(1));

        // Find the cosine of the angle formed
        return abs(dot_product / (magnitudeAB * magnitudeBC));
    }

    int LSDmatcher::SearchByGeomNApearance(Frame &CurrentFrame, const Frame &LastFrame, const float desc_th,vector<int> &matches_12)
    {
        int lmatches = 0;
        //std::vector<int> matches_12;

        ///先使用描述符来匹配两帧之间的线段
        match(LastFrame.mLdesc, CurrentFrame.mLdesc, desc_th, matches_12);

        std::vector<DMatch> geom_matches;
        const double deltaAngle = M_PI / 8.0;
        const double deltaWidth = (CurrentFrame.mnMaxX - CurrentFrame.mnMinX) * 0.1;
        const double deltaHeight = (CurrentFrame.mnMaxY - CurrentFrame.mnMinY) * 0.1;

        double th_angle = 20.0;
        //double th_angle = 10.0;
        double th_rad = th_angle / 180.0 * M_PI;
        double cos_th_angle = std::cos(th_rad);

        const int nmatches_12 = matches_12.size();
        for (int i1 = 0; i1 < nmatches_12; ++i1)
        {
            if (!LastFrame.mvpMapLines[i1])
                continue;
            const int i2 = matches_12[i1];
            if (i2 < 0)
                continue;

            if (CurrentFrame.mvKeylinesUn[i2].startPointX == 0)
                continue;

            // check for orientation
            cv::Mat v_line_current = (cv::Mat_<double>(1, 2)
                                          << CurrentFrame.mvKeylinesUn[i2].ePointInOctaveX - CurrentFrame.mvKeylinesUn[i2].sPointInOctaveX,
                                      CurrentFrame.mvKeylinesUn[i2].ePointInOctaveY - CurrentFrame.mvKeylinesUn[i2].sPointInOctaveY);

            cv::Mat v_line_last = (cv::Mat_<double>(1, 2)
                                       << LastFrame.mvKeylinesUn[i1].ePointInOctaveX - LastFrame.mvKeylinesUn[i1].sPointInOctaveX,
                                   LastFrame.mvKeylinesUn[i1].ePointInOctaveY - LastFrame.mvKeylinesUn[i1].sPointInOctaveY);

            double angle = computeAngle2D(v_line_current, v_line_last);

            if (angle < cos_th_angle)
            {
                matches_12[i1] = -1;
                continue;
            }

            // check for position in image
            const float &sX_curr = CurrentFrame.mvKeylinesUn[i2].sPointInOctaveX;
            const float &sX_last = LastFrame.mvKeylinesUn[i1].sPointInOctaveX;
            const float &sY_curr = CurrentFrame.mvKeylinesUn[i2].sPointInOctaveY;
            const float &sY_last = LastFrame.mvKeylinesUn[i1].sPointInOctaveY;
            const float &eX_curr = CurrentFrame.mvKeylinesUn[i2].ePointInOctaveX;
            const float &eX_last = LastFrame.mvKeylinesUn[i1].ePointInOctaveX;
            const float &eY_curr = CurrentFrame.mvKeylinesUn[i2].ePointInOctaveY;
            const float &eY_last = LastFrame.mvKeylinesUn[i1].ePointInOctaveY;

            if ((fabs(sX_curr - sX_last) > deltaWidth || fabs(sY_curr - sY_last) > deltaHeight) && (fabs(eX_curr - eX_last) > deltaWidth || fabs(eY_curr - eY_last) > deltaHeight))
            {
                matches_12[i1] = -1;
                continue;
            }

            cv::DMatch match;
            match.trainIdx = matches_12[i1];
            match.queryIdx = i1;
            match.distance = 2.0;
            geom_matches.push_back(match);
            CurrentFrame.mvpMapLines[i2] = LastFrame.mvpMapLines[i1];
            ++lmatches;
        }

        return lmatches;
    }

    std::pair<double, double> LSDmatcher::fitLineRANSAC(const std::vector<cv::Point2f >& points, int numIterations, double distanceThreshold) {
        int numPoints = points.size();
        if (numPoints < 2) {
            std::cerr << "Insufficient points for line fitting." << std::endl;
            return std::make_pair(0.0, 0.0);
        }

        std::pair<double, double> bestLine;
        int maxInliers = 0;

        srand(static_cast<unsigned>(time(nullptr)));

        for (int iteration = 0; iteration < numIterations; ++iteration) {
            // 随机选择两个点
            int index1 = rand() % numPoints;
            int index2 = rand() % numPoints;

            while (index2 == index1) {
                index2 = rand() % numPoints;
            }

            cv::Point2f p1 = points[index1];
            cv::Point2f p2 = points[index2];

            // 计算直线的斜率和截距
            double m = (p2.y - p1.y) / (p2.x - p1.x);
            double b = p1.y - m * p1.x;

            // 计算内点数
            int inliers = 0;
            for (const cv::Point2f& point : points) {
                double d = std::abs(point.y - m * point.x - b);
                if (d < distanceThreshold) {
                    inliers++;
                }
            }

            // 更新最佳拟合直线
            if (inliers > maxInliers) {
                maxInliers = inliers;
                bestLine = std::make_pair(m, b);
            }
        }

        return bestLine;
    }

    int LSDmatcher::optical_flow_line(ORB_SLAM2::Frame &CurrentFrame, const ORB_SLAM2::Frame &LastFrame,vector<int> &matchs_12,cv::Mat Rcl)
    {
        ///Rcl 是上一帧到当前帧的一个由ma估计的粗略的旋转   用做上一帧线段上采样点在当前帧上的位置估计
        cv::Mat img_last = LastFrame.ImageGray.clone();
        cv::Mat img_cur = CurrentFrame.ImageGray.clone();
        cv::Mat Kinv;
        bool tag = cv::invert(CurrentFrame.mK,Kinv);

        /// new method
        int len = LastFrame.NL;
        for(int i1=0;i1<len;i1++)
        {
            int i2 = matchs_12[i1];
            ///无匹配
            if(i2==-1)
            {
                ///但在当前帧可能有匹配的线段 但没匹配上
                ///对该线段进行切分 并用光流进行跟踪
                double len = cv::norm(LastFrame.mvKeylinesUn[i1].getStartPoint() - LastFrame.mvKeylinesUn[i1].getEndPoint());
                double numSmp = (double)min((int)len,20);
                vector<cv::Point2f> pt1,pt2,pt2_lk,pts;
                vector<uchar> status;
                vector<float> error;
                for(int j=0;j<=numSmp;j++)
                {
                    cv::Point2d pt = LastFrame.mvKeylinesUn[i1].getStartPoint() * (1 - j / numSmp) +
                                     LastFrame.mvKeylinesUn[i1].getEndPoint() * (j / numSmp);
                    if(pt.x<0||pt.y<0||pt.x>=LastFrame.ImageDepth.cols||pt.y>=LastFrame.ImageGray.rows)continue;
                    pt1.push_back(pt);
                }
                if(pt1.size() < 5)continue;

                ///采样点完成  存在temp中
                ///通过  p2 = K * R21* Kinv * p1  用ma辅助计算采样点在当前帧上的估计位置  作为光流初始值
                for(int j=0;j<pt1.size();j++)
                {
                    cv::Mat p1 = (cv::Mat_<float>(3,1) << pt1[j].x , pt1[j].y , 1.0);
                    cv::Mat mat;
                    mat = CurrentFrame.mK * Rcl * Kinv * p1;
                    cv::Point2f p2(mat.at<float>(0,0),mat.at<float>(1,0));
                    pt2.emplace_back(p2);
                }
                cv::calcOpticalFlowPyrLK(img_cur,img_cur,pt2,pt2_lk,status,error);
                ///跟踪的结果存储在pt2_lk中  对ans进行ransac拟合  剔除离群值
                for(int j=0;j<pt2_lk.size();j++)if(status[j])pts.emplace_back(pt2_lk[j]);
                ///pts存放光流跟踪成功的结果
                std::pair<double,double> fit_line = fitLineRANSAC(pts,10,10);
                //cout<<"fit line 1 y = "<<fit_line.first<<"*x + "<<fit_line.second<<endl;
                if(fit_line.first == 0.0 && fit_line.second == 0.0)
                {
                    ///RANSAC拟合直线失败
                    cout<<"RANSAC拟合直线失败-------"<<endl;
                    matchs_12[i1] = -1;
                    continue;
                }

                ///pts中存在离群值  离群值距离估计出来的直线的距离大于5
                bool inLiner[pts.size()];
                fill(inLiner,inLiner+pts.size(),true);
                for(int i=0;i<pts.size();i++)
                {
                    double dis = pts[i].y - fit_line.first*pts[i].x - fit_line.second;
                    if(dis>5.0)inLiner[i]=false;
                }
                //cout<<"before fit point num = "<<pts.size()<<endl;
                vector<cv::Point2f> temp;
                for(int i=0;i<pts.size();i++)if(inLiner[i])temp.emplace_back(pts[i]);
                //cout<<"after fit point num = "<<temp.size()<<endl;
                ////拟合的已经很好的  但还是加一个去除离群值的步骤  重新拟合直线
                std::pair<double,double> fit_line2 = fitLineRANSAC(temp,10,5);
                //cout<<"fit line 2 y = "<<fit_line2.first<<"*x + "<<fit_line2.second<<endl;
                ///temp中第一个元素就是预测线段的起点 最后一个元素就是预测线段的终点
                ////用拟合的直线与当前帧检测到的直线做匹配  优化描述子的匹配结果
                ///接下来枚举所有当前帧中检测到的线段
                int num = CurrentFrame.NL;
                int bestId = -1;
                double sum_err=100.0;
                for(int i=0;i<num;i++)
                {
                    ///衡量匹配线的三个指标  一是采样点到线段的平均距离 二是预测线段法向量与候选线段的点乘  三是中点之间的距离
                    Eigen::Vector3d line_obs = CurrentFrame.mvKeyLineFunctions[i];
                    double dis_err = 0.0;
                    for(int j=0;j<temp.size();j++)
                    {
                        dis_err += (double)temp[j].x * line_obs[0] +(double)temp[j].y * line_obs[1] +line_obs[2];
                    }
                    if(dis_err!=0.0)dis_err/=temp.size();
                    ///预测线段 终点-起点
                    Eigen::Vector2d l1;
                    cv::Point2f line1 = temp[temp.size()-1] - temp[0];
                    l1 << (double)line1.x , (double)line1.y;
                    l1[0] /= sqrt(l1[0]*l1[0] + l1[1]*l1[1]);
                    l1[1] /= sqrt(l1[0]*l1[0] + l1[1]*l1[1]);
                    ///预测线段的法向量
                    Eigen::Vector2d nor1(l1[1],-1.0*l1[0]);
                    Eigen::Vector2d l2;
                    cv::Point2f line2 = CurrentFrame.mvKeylinesUn[i].getEndPoint() - CurrentFrame.mvKeylinesUn[i].getStartPoint();
                    l2 << (double)line2.x , (double)line2.y;
                    l2[0] /= sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
                    l2[1] /= sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
                    double angle_err = nor1.dot(l2);
                    if(isnan(angle_err))
                    {
                        cout<<"预测线段的法向量与候选线段点乘为nan！！！！！"<<endl;
                        continue;
                    }
                    if(std::abs(angle_err) > 0.1)continue;
                    if(std::abs(dis_err) > 2.5)continue;
                    cv::Point2f mid1 = (temp[temp.size()-1] + temp[0])*0.5;
                    cv::Point2f mid2 = (CurrentFrame.mvKeylinesUn[i].getStartPoint() + CurrentFrame.mvKeylinesUn[i].getEndPoint())*0.5;
                    ////中点距离过大就得剔除
                    if(std::sqrt(std::pow(mid1.x - mid2.x,2) + std::pow(mid1.y - mid2.y,2)) > 2.5)continue;
                    //cout<<"i = "<<i<<" dis_err = "<<dis_err<<" angle_err = "<<angle_err<<endl;
                    if(std::abs(dis_err)+std::abs(angle_err) < sum_err)
                    {
                        sum_err = std::abs(dis_err) + std::abs(angle_err);
                        bestId = i;
                    }
                }
                if(bestId == -1)continue;
                //cout<<"bestId = "<<bestId<<" error = "<<sum_err<<endl;
                ///画出两条匹配的线段
//                cv::Mat img1 = LastFrame.ImageGray.clone();
//                cv::Mat img2 = CurrentFrame.ImageGray.clone();
//                cvtColor(img1,img1,CV_GRAY2BGR);
//                cvtColor(img2,img2,CV_GRAY2BGR);
//                ///当前帧的线段
//                cv::line(img2,CurrentFrame.mvKeylinesUn[bestId].getStartPoint(),CurrentFrame.mvKeylinesUn[bestId].getEndPoint()
//                ,cv::Scalar(255,0,0),1);
//                ///上一帧的线段
//                cv::line(img1,LastFrame.mvKeylinesUn[i1].getStartPoint(),LastFrame.mvKeylinesUn[i1].getEndPoint(),
//                         cv::Scalar(255,0,0),1);
//                ///预测线段  画出采样点
//                for(int i=0;i<temp.size();i++)
//                {
//                    cv::circle(img2,temp[i],1,cv::Scalar(0,0,255),1);
//                }
//                cv::Mat combine_img;
//                cvtColor(combine_img,combine_img,CV_GRAY2BGR);
//                cv::hconcat(img1,img2,combine_img);
//                cv::Point2f a_mid = (LastFrame.mvKeylinesUn[i1].getEndPoint() + LastFrame.mvKeylinesUn[i1].getStartPoint())*0.5;
//                cv::Point2f b_mid = (CurrentFrame.mvKeylinesUn[bestId].getEndPoint() + CurrentFrame.mvKeylinesUn[bestId].getStartPoint())*0.5;
//                b_mid.x += img1.cols;
//                cv::line(combine_img,a_mid,b_mid,cv::Scalar(0,255,0),1);
//                cv::Mat img3 = LastFrame.ImageGray.clone();
//                cv::Mat img4 = CurrentFrame.ImageGray.clone();
//                cvtColor(img3,img3,CV_GRAY2BGR);
//                cvtColor(img4,img4,CV_GRAY2BGR);
//                for(int i=0;i<LastFrame.mvKeylinesUn.size();i++)
//                {
//                    cv::line(img3,LastFrame.mvKeylinesUn[i].getStartPoint(),LastFrame.mvKeylinesUn[i].getEndPoint(),
//                             cv::Scalar(255,0,0),1);
//                }
//                for(int i=0;i<CurrentFrame.mvKeylinesUn.size();i++)
//                {
//                    cv::line(img4,CurrentFrame.mvKeylinesUn[i].getStartPoint(),CurrentFrame.mvKeylinesUn[i].getEndPoint(),
//                             cv::Scalar(255,0,0),1);
//                }
//                cv::Mat combine_img2;
//                cvtColor(combine_img2,combine_img2,CV_GRAY2BGR);
//                cv::hconcat(img3,img4,combine_img2);
//                cv::imshow("wh",combine_img);
//                getchar();
//                cv::destroyWindow("wh");


                matchs_12[i1] = bestId;
            }
            else
            {
                //continue;
                ///有匹配 但匹配不一定是正确的  接下来开始检查lbd匹配的结果是否符合要求
                ///1.两条匹配的线段应该是大致平行的
                Vector2d l1;
                cv::Point2f line1 = LastFrame.mvKeylinesUn[i1].getEndPoint() - LastFrame.mvKeylinesUn[i1].getStartPoint();
                l1 << (double)line1.x , (double)line1.y;
                l1[0] /= sqrt(l1[0]*l1[0] + l1[1]*l1[1]);
                l1[1] /= sqrt(l1[0]*l1[0] + l1[1]*l1[1]);
                Vector2d nor1(l1[1],-1.0*l1[0]);
                Vector2d l2;
                cv::Point2f line2 = CurrentFrame.mvKeylinesUn[i2].getEndPoint() - CurrentFrame.mvKeylinesUn[i2].getStartPoint();
                l2 << (double)line2.x , (double)line2.y;
                l2[0] /= sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
                l2[1] /= sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
                ///如何没有匹配 点乘是nan
                ///点乘按道理来说应该是接近0的一个数字 误匹配会是一个比较大的值  但也有一些特殊的误匹配  值接近0 应该使用其它条件筛选
                double error = nor1.dot(l2);
                double dis = 0.0;
                ///mid_p 是上一帧中一条线段的中点   dis为mid_p到当前帧中匹配的线段的距离
                Vector2d mid_p;
                mid_p << (double)(LastFrame.mvKeylinesUn[i1].startPointX+LastFrame.mvKeylinesUn[i1].endPointX)*0.5,
                        (double)(LastFrame.mvKeylinesUn[i1].startPointY+LastFrame.mvKeylinesUn[i1].endPointY)*0.5;
                dis = mid_p[0]*CurrentFrame.mvKeyLineFunctions[i2][0] + mid_p[1]*CurrentFrame.mvKeyLineFunctions[i2][1] + CurrentFrame.mvKeyLineFunctions[i2][2];
                ///根据距离和向量点乘可以过滤掉大部分的误匹配线段
                if(error > 0.1 || dis > 10.0)
                {
                    ///但在当前帧可能有匹配的线段 但没匹配上
                    ///对该线段进行切分 并用光流进行跟踪
                    double len = cv::norm(LastFrame.mvKeylinesUn[i1].getStartPoint() - LastFrame.mvKeylinesUn[i1].getEndPoint());
                    double numSmp = (double)min((int)len,20);
                    vector<cv::Point2f> pt1,pt2,pt2_lk,pts;
                    vector<uchar> status;
                    vector<float> error;
                    for(int j=0;j<=numSmp;j++)
                    {
                        cv::Point2d pt = LastFrame.mvKeylinesUn[i1].getStartPoint() * (1 - j / numSmp) +
                                         LastFrame.mvKeylinesUn[i1].getEndPoint() * (j / numSmp);
                        if(pt.x<0||pt.y<0||pt.x>=LastFrame.ImageDepth.cols||pt.y>=LastFrame.ImageGray.rows)continue;
                        pt1.push_back(pt);
                    }
                    if(pt1.size() < 5)
                    {
                        matchs_12[i1] = -1;
                    }

                    ///采样点完成  存在temp中
                    ///通过  p2 = K * R21* Kinv * p1  用ma辅助计算采样点在当前帧上的估计位置  作为光流初始值
                    for(int j=0;j<pt1.size();j++)
                    {
                        cv::Mat p1 = (cv::Mat_<float>(3,1) << pt1[j].x , pt1[j].y , 1.0);
                        cv::Mat mat;
                        mat = CurrentFrame.mK * Rcl * Kinv * p1;
                        cv::Point2f p2(mat.at<float>(0,0),mat.at<float>(1,0));
                        pt2.emplace_back(p2);
                    }
                    cv::calcOpticalFlowPyrLK(img_cur,img_cur,pt2,pt2_lk,status,error);
                    ///跟踪的结果存储在pt2_lk中  对ans进行ransac拟合  剔除离群值  构造新的线特征
                    for(int j=0;j<pt2_lk.size();j++)if(status[j])pts.emplace_back(pt2_lk[j]);
                    ///pts存放光流跟踪成功的结果
                    std::pair<double,double> fit_line = fitLineRANSAC(pts,10,10);
                    //cout<<"fit line 1 y = "<<fit_line.first<<"*x + "<<fit_line.second<<endl;
                    if(fit_line.first == 0.0 && fit_line.second == 0.0)
                    {
                        ///RANSAC拟合直线失败
                        cout<<"RANSAC拟合直线失败-------"<<endl;
                        matchs_12[i1] = -1;
                        continue;
                    }

                    ///pts中存在离群值  离群值距离估计出来的直线的距离大于10
                    bool inLiner[pts.size()];
                    fill(inLiner,inLiner+pts.size(),true);
                    for(int i=0;i<pts.size();i++)
                    {
                        double dis = pts[i].y - fit_line.first*pts[i].x - fit_line.second;
                        if(dis>5.0)inLiner[i]=false;
                    }
                    //cout<<"before fit point num = "<<pts.size()<<endl;
                    vector<cv::Point2f> temp;
                    for(int i=0;i<pts.size();i++)if(inLiner[i])temp.emplace_back(pts[i]);
                    //cout<<"after fit point num = "<<temp.size()<<endl;
                    ////拟合的已经很好的  但还是加一个去除离群值的步骤  重新拟合直线
                    std::pair<double,double> fit_line2 = fitLineRANSAC(temp,10,5);
                    //cout<<"fit line 2 y = "<<fit_line2.first<<"*x + "<<fit_line2.second<<endl;
                    ///temp中第一个元素就是预测线段的起点 最后一个元素就是预测线段的终点
                    ////用拟合的直线与当前帧检测到的直线做匹配  优化描述子的匹配结果
                    ///接下来枚举所有当前帧中检测到的线段
                    int num = CurrentFrame.NL;
                    int bestId = -1;
                    double sum_err=100.0;
                    for(int i=0;i<num;i++)
                    {
                        ///衡量匹配线的两个指标  一是采样点到线段的距离 二是预测线段法向量与候选线段的点乘
                        Eigen::Vector3d line_obs = CurrentFrame.mvKeyLineFunctions[i];
                        double dis_err = 0.0;
                        for(int j=0;j<temp.size();j++)
                        {
                            dis_err += (double)temp[j].x * line_obs[0] +(double)temp[j].y * line_obs[1] +line_obs[2];
                        }
                        if(dis_err!=0.0)dis_err/=temp.size();
                        ///预测线段 终点-起点
                        Eigen::Vector2d l1;
                        cv::Point2f line1 = temp[temp.size()-1] - temp[0];
                        l1 << (double)line1.x , (double)line1.y;
                        l1[0] /= sqrt(l1[0]*l1[0] + l1[1]*l1[1]);
                        l1[1] /= sqrt(l1[0]*l1[0] + l1[1]*l1[1]);
                        ///预测线段的法向量
                        Eigen::Vector2d nor1(l1[1],-1.0*l1[0]);
                        Eigen::Vector2d l2;
                        cv::Point2f line2 = CurrentFrame.mvKeylinesUn[i].getEndPoint() - CurrentFrame.mvKeylinesUn[i].getStartPoint();
                        l2 << (double)line2.x , (double)line2.y;
                        l2[0] /= sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
                        l2[1] /= sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
                        double angle_err = nor1.dot(l2);
                        if(isnan(angle_err))
                        {
                            cout<<"预测线段的法向量与候选线段点乘为nan！！！！！"<<endl;
                            matchs_12[i1] = -1;
                            continue;
                        }
                        if(std::abs(angle_err) > 0.1)
                        {
                            matchs_12[i1] = -1;
                            continue;
                        }
                        if(std::abs(dis_err) > 2.5)
                        {
                            matchs_12[i1] = -1;
                            continue;
                        }
                        cv::Point2f mid1 = (temp[temp.size()-1] + temp[0])*0.5;
                        cv::Point2f mid2 = (CurrentFrame.mvKeylinesUn[i].getStartPoint() + CurrentFrame.mvKeylinesUn[i].getEndPoint())*0.5;
                        ////中点距离过大就得剔除
                        if(std::sqrt(std::pow(mid1.x - mid2.x,2) + std::pow(mid1.y - mid2.y,2)) > 2.5)
                        {
                            matchs_12[i1] = -1;
                            continue;
                        }
                        //cout<<"i = "<<i<<" dis_err = "<<dis_err<<" angle_err = "<<angle_err<<endl;
                        if(std::abs(dis_err)+std::abs(angle_err) < sum_err)
                        {
                            sum_err = std::abs(dis_err) + std::abs(angle_err);
                            bestId = i;
                        }
                    }
                    if(bestId == -1)
                    {
                        matchs_12[i1] = -1;
                        continue;
                    }
                    //cout<<"bestId = "<<bestId<<" error = "<<sum_err<<endl;
                    ///画出两条匹配的线段
                    cv::Mat img1 = LastFrame.ImageGray.clone();
                    cv::Mat img2 = CurrentFrame.ImageGray.clone();
                    cvtColor(img1,img1,CV_GRAY2BGR);
                    cvtColor(img2,img2,CV_GRAY2BGR);
                    ///当前帧的线段
                    cv::line(img2,CurrentFrame.mvKeylinesUn[bestId].getStartPoint(),CurrentFrame.mvKeylinesUn[bestId].getEndPoint()
                            ,cv::Scalar(0,0,255),1);
                    ///当前帧匹配错误的线段
                    cv::line(img2,CurrentFrame.mvKeylinesUn[i2].getStartPoint(),CurrentFrame.mvKeylinesUn[i2].getEndPoint(),
                             cv::Scalar(0,165,255),1);
                    ///上一帧的线段
                    cv::line(img1,LastFrame.mvKeylinesUn[i1].getStartPoint(),LastFrame.mvKeylinesUn[i1].getEndPoint(),
                             cv::Scalar(255,0,0),1);
                    ///预测线段  画出采样点
//                    for(int i=0;i<temp.size();i++)
//                    {
//                        cv::circle(img2,temp[i],1,cv::Scalar(0,0,255),1);
//                    }
                    cv::Mat combine_img;
                    cvtColor(combine_img,combine_img,CV_GRAY2BGR);
                    cv::hconcat(img1,img2,combine_img);
                    cv::Point2f a_mid = (LastFrame.mvKeylinesUn[i1].getEndPoint() + LastFrame.mvKeylinesUn[i1].getStartPoint())*0.5;
                    cv::Point2f b_mid = (CurrentFrame.mvKeylinesUn[i2].getEndPoint() + CurrentFrame.mvKeylinesUn[i2].getStartPoint())*0.5;
                    b_mid.x += img1.cols;
                    cv::line(combine_img,a_mid,b_mid,cv::Scalar(0,255,0),1);
//                    cv::imshow("wh",combine_img);
//                    //cv::imshow("ly",combine_img2);
//                    getchar();
//                    //cv::destroyAllWindows();
//                    cv::destroyWindow("wh");


                    matchs_12[i1] = bestId;
                }
                else
                {
                    ///使用描述符匹配成功的线段
                }
            }
        }
    }

    int LSDmatcher::SearchByDescriptor(KeyFrame* pKF, Frame &currentF, vector<MapLine*> &vpMapLineMatches)
    {
        const vector<MapLine*> vpMapLinesKF = pKF->GetMapLineMatches();

        vpMapLineMatches = vector<MapLine*>(currentF.NL,static_cast<MapLine*>(NULL));

        int nmatches = 0;
        BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
        Mat ldesc1, ldesc2;
        vector<vector<DMatch>> lmatches;
        ldesc1 = pKF->mLineDescriptors;
        ldesc2 = currentF.mLdesc;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        const float minRatio=1.0f/1.5f;
        currentF.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.5;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for(int i=0; i<lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][0].distance/lmatches[i][1].distance;
            if(dist_12<minRatio)
            {
                MapLine* mapLine = vpMapLinesKF[qdx];

                if(mapLine)
                {
                    vpMapLineMatches[tdx]=mapLine;
                    nmatches++;
                }

            }
        }
        return nmatches;
    }

    int LSDmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th)
    {
        double th_angle = 10.0;
        double th_rad = th_angle / 180.0 * M_PI;
        double cos_th_angle = std::cos(th_rad);

        int nmatches = 0;
        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH]; // HISTO_LENGTH=30
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat twc = -Rcw.t() * tcw;

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat tlc = Rlw * twc + tlw;

        for (int i = 0; i < LastFrame.NL; i++)
        {
            MapLine *pML = LastFrame.mvpMapLines[i];

            if (pML)
            {
                if (!LastFrame.mvbLineOutlier[i])
                {
                    // Project
                    Vector6d Lw = pML->GetWorldPos();
                    if (!CurrentFrame.isInFrustum(pML, 0.5))
                        continue;

                    int nLastOctave = pML->mnTrackScaleLevel;

                    // Search in a window. Size depends on scale
                    float radius = th;

                    vector<size_t> vIndices2;

                    vIndices2 = CurrentFrame.GetFeaturesInAreaForLine(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2,
                                                                      radius, nLastOctave - 1, nLastOctave + 1, 0.96);
                    // vIndices2 = CurrentFrame.GetLinesInArea(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2, radius, nLastOctave-1, nLastOctave+1, 0.96);

                    if (vIndices2.empty())
                        continue;

                    const cv::Mat dML = pML->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if (CurrentFrame.mvpMapLines[i2])
                            if (CurrentFrame.mvpMapLines[i2]->Observations() > 0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mLdesc.row(i2);

                        // Check orientation
                        cv::Mat v_line_current = (cv::Mat_<double>(1, 2)
                                                      << CurrentFrame.mvKeylinesUn[i2].ePointInOctaveX - CurrentFrame.mvKeylinesUn[i2].sPointInOctaveX,
                                                  CurrentFrame.mvKeylinesUn[i2].ePointInOctaveY - CurrentFrame.mvKeylinesUn[i2].sPointInOctaveY);

                        cv::Mat v_line_last = (cv::Mat_<double>(1, 2)
                                                   << LastFrame.mvKeylinesUn[i].ePointInOctaveX - LastFrame.mvKeylinesUn[i].sPointInOctaveX,
                                               LastFrame.mvKeylinesUn[i].ePointInOctaveY - LastFrame.mvKeylinesUn[i].sPointInOctaveY);

                        double angle = computeAngle2D(v_line_current, v_line_last);

                        if (angle < cos_th_angle)
                            continue;

                        const int dist = DescriptorDistance(dML, d);

                        float max_ = std::max(LastFrame.mvKeylinesUn[i].lineLength, CurrentFrame.mvKeylinesUn[i2].lineLength);
                        float min_ = std::min(LastFrame.mvKeylinesUn[i].lineLength, CurrentFrame.mvKeylinesUn[i2].lineLength);

                        if (min_ / max_ < 0.75)
                            continue;

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    if (bestDist <= 95)
                    {
                        CurrentFrame.mvpMapLines[bestIdx2] = pML;
                        nmatches++;
                    }
                }
            }
        }

        return nmatches;
    }

    void LSDmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++)
        {
            const int s = histo[i].size();
            if (s > max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if (s > max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if (s > max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float)max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if (max3 < 0.1f * (float)max1)
        {
            ind3 = -1;
        }
    }

    int LSDmatcher::SearchByProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const bool eval_orient, const float th)
    {
        int nmatches = 0;

        double th_angle = 15;
        double th_rad = th_angle / 180.0 * M_PI;
        double th_normal = std::cos(th_rad);

        const bool bFactor = th != 1.0;
        for (size_t iML = 0; iML < vpMapLines.size(); iML++)
        {
            MapLine *pML = vpMapLines[iML];

            if (!pML->mbTrackInView)
                continue;

            if (pML->isBad())
                continue;

            const int &nPredictLevel = pML->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            float r = RadiusByViewingCos(pML->mTrackViewCos);

            if (bFactor)
                r *= th;

            // vector<size_t> vIndices = F.GetLinesInArea(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2, r*F.mvScaleFactorsLine[nPredictLevel], nPredictLevel-1, nPredictLevel);

            vector<size_t> vIndices = F.GetFeaturesInAreaForLine(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2, r, nPredictLevel - 1, nPredictLevel);
            if (vIndices.empty())
                continue;
            const cv::Mat MLdescriptor = pML->GetDescriptor();
            Vector6d m_world_pose = pML->GetWorldPos();
            Eigen::Vector3d mWorldVectorML = pML->GetWorldVector();

            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                if (F.mvpMapLines[idx])
                    if (F.mvpMapLines[idx]->Observations() > 0)
                        continue;

                const cv::Mat &d = F.mLdesc.row(idx);

                Eigen::Vector3d mWorldVector_frame = F.mvLines3D[idx].first - F.mvLines3D[idx].second;

                float dot = mWorldVector_frame.dot(mWorldVectorML);
                float mag_f = std::sqrt(mWorldVector_frame.x() * mWorldVector_frame.x() + mWorldVector_frame.y() * mWorldVector_frame.y() + mWorldVector_frame.z() * mWorldVector_frame.z());

                float mag_ml = std::sqrt(mWorldVectorML.x() * mWorldVectorML.x() + mWorldVectorML.y() * mWorldVectorML.y() + mWorldVectorML.z() * mWorldVectorML.z());

                float angle = abs(dot / (mag_f * mag_ml));

                if (angle < th_normal)
                    continue;

                const int dist = DescriptorDistance(MLdescriptor, d);

                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeylinesUn[idx].octave;
                    bestIdx = idx;
                }
                else if (dist < bestDist2)
                {
                    bestLevel2 = F.mvKeylinesUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestDist <= 95)
            {
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                    continue;

                F.mvpMapLines[bestIdx] = pML;
                nmatches++;
            }
        }
        return nmatches;
    }

    int LSDmatcher::matchNNR(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12)
    {

        int matches = 0;
        ///初始化为-1
        matches_12.resize(desc1.rows, -1);

        std::vector<std::vector<cv::DMatch>> matches_;
        cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, false); // cross-check
        bfm->knnMatch(desc1, desc2, matches_, 2);

        if (desc1.rows != matches_.size())
            throw std::runtime_error("[matchNNR] Different size for matches and descriptors!");

        for (int idx = 0, nsize = desc1.rows; idx < nsize; ++idx)
        {
            if (matches_[idx][0].distance < matches_[idx][1].distance * nnr)
            {
                matches_12[idx] = matches_[idx][0].trainIdx;
                matches++;
            }
        }
        return matches;
    }

    int LSDmatcher::match(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12)
    {
        if (false)
        {
            int matches;
            std::vector<int> matches_21;
            // if (true)
            // {
            //     auto match_12 = std::async(std::launch::async, &matchNNR,
            //                                std::cref(desc1), std::cref(desc2), nnr, std::ref(matches_12));
            //     auto match_21 = std::async(std::launch::async, &matchNNR,
            //                                std::cref(desc2), std::cref(desc1), nnr, std::ref(matches_21));
            //     matches = match_12.get();
            //     match_21.wait();
            // }
            // else
            // {
            matches = matchNNR(desc1, desc2, nnr, matches_12);
            matchNNR(desc2, desc1, nnr, matches_21);
            // }

            for (int i1 = 0, nsize = matches_12.size(); i1 < nsize; ++i1)
            {
                int &i2 = matches_12[i1];
                if (i2 >= 0 && matches_21[i2] != i1)
                {
                    i2 = -1;
                    matches--;
                }
            }

            return matches;
        }
        else
            return matchNNR(desc1, desc2, nnr, matches_12);
    }

    int LSDmatcher::SearchDouble(KeyFrame *KF, Frame &CurrentFrame)
    {
        vector<MapLine *> LineMatches = vector<MapLine *>(CurrentFrame.NL, static_cast<MapLine *>(NULL));
        vector<int> tempMatches1 = vector<int>(KF->NL, -1);
        vector<int> tempMatches2 = vector<int>(CurrentFrame.NL, -1);

        Mat ldesc1, ldesc2;
        ldesc1 = KF->mLineDescriptors;
        ldesc2 = CurrentFrame.mLdesc;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        thread thread12(&LSDmatcher::FrameBFMatch, this, ldesc1, ldesc2, std ::ref(tempMatches1), TH_LOW);
        thread thread21(&LSDmatcher::FrameBFMatch, this, ldesc2, ldesc1, std ::ref(tempMatches2), TH_LOW);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches2.size(); i++)
        {
            int j = tempMatches2[i];
            if (j >= 0)
            {
                if (tempMatches1[j] == i)
                {
                    MapLine *pML = KF->GetMapLine(j);
                    if (!pML)
                        continue;
                    CurrentFrame.mvpMapLines[i] = pML;
                    nmatches++;
                }
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchDouble(Frame &InitialFrame, Frame &CurrentFrame, vector<int> &LineMatches)
    {
        LineMatches = vector<int>(InitialFrame.NL, -1);
        vector<int> tempMatches1, tempMatches2;

        Mat ldesc1, ldesc2;
        ldesc1 = InitialFrame.mLdesc;
        ldesc2 = CurrentFrame.mLdesc;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        thread thread12(&LSDmatcher::FrameBFMatch, this, ldesc1, ldesc2, std ::ref(tempMatches1), TH_LOW);
        thread thread21(&LSDmatcher::FrameBFMatch, this, ldesc2, ldesc1, std ::ref(tempMatches2), TH_LOW);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches1.size(); i++)
        {
            int j = tempMatches1[i];
            if (j >= 0)
            {
                if (tempMatches2[j] != i)
                {
                    tempMatches1[i] = -1;
                }
                else
                {
                    nmatches++;
                }
            }
        }

        LineMatches = tempMatches1;

        return nmatches;
    }

    void LSDmatcher::FrameBFMatch(cv::Mat ldesc1, cv::Mat ldesc2, vector<int> &LineMatches, float TH)
    {
        LineMatches = vector<int>(ldesc1.rows, -1);

        vector<vector<DMatch>> lmatches;

        BFMatcher *bfm = new BFMatcher(NORM_HAMMING, false);
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);

        nn12_dist_th = nn12_dist_th * 0.5;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());

        for (int i = 0; i < lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;

            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if (dist_12 > nn12_dist_th && lmatches[i][0].distance < TH && lmatches[i][0].distance < mfNNratio * lmatches[i][1].distance)
                LineMatches[qdx] = tdx;
        }
    }

    void LSDmatcher::FrameBFMatchNew(cv::Mat ldesc1, cv::Mat ldesc2, vector<int> &LineMatches, vector<KeyLine> kls1, vector<KeyLine> kls2, vector<Eigen::Vector3d> kls2func, cv::Mat F, float TH)
    {
        LineMatches = vector<int>(ldesc1.rows, -1);

        vector<vector<DMatch>> lmatches;

        BFMatcher *bfm = new BFMatcher(NORM_HAMMING, false);
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());

        for (int i = 0; i < lmatches.size(); i++)
        {
            for (int j = 0; j < lmatches[i].size() - 1; j++)
            {
                int qdx = lmatches[i][j].queryIdx;
                int tdx = lmatches[i][j].trainIdx;

                cv::Mat p1 = (cv::Mat_<float>(3, 1) << kls1[qdx].startPointX, kls1[qdx].startPointY, 1.0);
                cv::Mat p2 = (cv::Mat_<float>(3, 1) << kls1[qdx].endPointX, kls1[qdx].endPointY, 1.0);

                cv::Mat epi_p1 = F * p1;
                cv::Mat epi_p2 = F * p2;

                cv::Mat q1 = (cv::Mat_<float>(3, 1) << kls2[tdx].startPointX, kls2[tdx].startPointY, 1.0);
                cv::Mat q2 = (cv::Mat_<float>(3, 1) << kls2[tdx].endPointX, kls2[tdx].endPointY, 1.0);

                cv::Mat l2 = (cv::Mat_<float>(3, 1) << kls2func[tdx](0), kls2func[tdx](1), kls2func[tdx](2));
                cv::Mat p1_proj = l2.cross(epi_p1);
                cv::Mat p2_proj = l2.cross(epi_p2);

                if (fabs(p1_proj.at<float>(2)) > 1e-12 && fabs(p2_proj.at<float>(2)) > 1e-12)
                {
                    // normalize
                    p1_proj /= p1_proj.at<float>(2);
                    p2_proj /= p2_proj.at<float>(2);

                    std::vector<cv::Mat> collinear_points(4);
                    collinear_points[0] = p1_proj;
                    collinear_points[1] = p2_proj;
                    collinear_points[2] = q1;
                    collinear_points[3] = q2;
                    float score = mutualOverlap(collinear_points);

                    if (lmatches[i][j].distance < TH)
                    {
                        if (score > 0.8 && lmatches[i][j].distance < mfNNratio * lmatches[i][j + 1].distance)
                        {
                            LineMatches[qdx] = tdx;
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    continue;
                }
            }
        }
    }

    float LSDmatcher::mutualOverlap(const std::vector<cv::Mat> &collinear_points)
    {
        float overlap = 0.0f;

        if (collinear_points.size() != 4)
            return 0.0f;

        cv::Mat p1 = collinear_points[0];
        cv::Mat p2 = collinear_points[1];
        cv::Mat q1 = collinear_points[2];
        cv::Mat q2 = collinear_points[3];

        // find outer distance and inner points
        float max_dist = 0.0f;
        size_t outer1 = 0;
        size_t inner1 = 1;
        size_t inner2 = 2;
        size_t outer2 = 3;

        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = i + 1; j < 4; ++j)
            {
                float dist = norm(collinear_points[i] - collinear_points[j]);
                if (dist > max_dist)
                {
                    max_dist = dist;
                    outer1 = i;
                    outer2 = j;
                }
            }
        }

        if (max_dist < 1.0f)
            return 0.0f;

        if (outer1 == 0)
        {
            if (outer2 == 1)
            {
                inner1 = 2;
                inner2 = 3;
            }
            else if (outer2 == 2)
            {
                inner1 = 1;
                inner2 = 3;
            }
            else
            {
                inner1 = 1;
                inner2 = 2;
            }
        }
        else if (outer1 == 1)
        {
            inner1 = 0;
            if (outer2 == 2)
            {
                inner2 = 3;
            }
            else
            {
                inner2 = 2;
            }
        }
        else
        {
            inner1 = 0;
            inner2 = 1;
        }

        overlap = norm(collinear_points[inner1] - collinear_points[inner2]) / max_dist;

        return overlap;
    }

    void LSDmatcher::lineDescriptorMAD(vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const
    {
        vector<vector<DMatch>> matches_nn, matches_12;
        matches_nn = line_matches;
        matches_12 = line_matches;
        // cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

        // estimate the NN's distance standard deviation
        double nn_dist_median;
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_dist_median = matches_nn[int(matches_nn.size() / 2)][0].distance;

        for (unsigned int i = 0; i < matches_nn.size(); i++)
            matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_mad = 1.4826 * matches_nn[int(matches_nn.size() / 2)][0].distance;

        // estimate the NN's 12 distance standard deviation
        double nn12_dist_median;
        sort(matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
        nn12_dist_median = matches_12[int(matches_12.size() / 2)][1].distance - matches_12[int(matches_12.size() / 2)][0].distance;
        for (unsigned int j = 0; j < matches_12.size(); j++)
            matches_12[j][0].distance = fabsf(matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
        sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
        nn12_mad = 1.4826 * matches_12[int(matches_12.size() / 2)][0].distance;
    }

    int LSDmatcher::DescriptorDistance(const Mat &a, const Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    int LSDmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t>> &vMatchedPairs)
    {

        vMatchedPairs.clear();
        vector<int> tempMatches1, tempMatches2;

        Mat ldesc1, ldesc2;
        ldesc1 = pKF1->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        thread thread12(&LSDmatcher::FrameBFMatch, this, ldesc1, ldesc2, std ::ref(tempMatches1), TH_LOW);
        thread thread21(&LSDmatcher::FrameBFMatch, this, ldesc2, ldesc1, std ::ref(tempMatches2), TH_LOW);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches1.size(); i++)
        {
            int j = tempMatches1[i];
            if (j >= 0)
            {
                if (tempMatches2[j] == i)
                {

                    if (pKF1->GetMapLine(i) || pKF2->GetMapLine(j))
                        continue;

                    vMatchedPairs.push_back(make_pair(i, j));
                    nmatches++;
                }
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, vector<int> &vMatchedPairs, bool isDouble)
    {
        vMatchedPairs.clear();
        vMatchedPairs.resize(pKF1->NL, -1);
        vector<int> tempMatches1, tempMatches2;

        Mat ldesc1, ldesc2;
        ldesc1 = pKF1->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        thread thread12(&LSDmatcher::FrameBFMatch, this, ldesc1, ldesc2, std ::ref(tempMatches1), TH_HIGH);
        thread thread21(&LSDmatcher::FrameBFMatch, this, ldesc2, ldesc1, std ::ref(tempMatches2), TH_HIGH);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches1.size(); i++)
        {
            int j = tempMatches1[i];
            if (j >= 0)
            {
                if (isDouble && tempMatches2[j] != i)
                    continue;

                if (pKF1->GetMapLine(i) || pKF2->GetMapLine(j))
                    continue;

                vMatchedPairs[i] = j;
                nmatches++;
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchForTriangulationNew(KeyFrame *pKF1, KeyFrame *pKF2, vector<int> &vMatchedPairs, bool isDouble)
    {
        vMatchedPairs.clear();
        vMatchedPairs.resize(pKF1->NL, -1);
        vector<int> tempMatches1, tempMatches2;

        Mat ldesc1, ldesc2;
        ldesc1 = pKF1->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        vector<KeyLine> kls1 = pKF1->mvKeyLines;
        vector<KeyLine> kls2 = pKF2->mvKeyLines;
        vector<Eigen::Vector3d> kls1func = pKF1->mvKeyLineFunctions;
        vector<Eigen::Vector3d> kls2func = pKF2->mvKeyLineFunctions;

        cv::Mat F21 = ComputeF12(pKF2, pKF1);
        cv::Mat F12 = ComputeF12(pKF1, pKF2);

        thread thread12(&LSDmatcher::FrameBFMatchNew, this, ldesc1, ldesc2, std ::ref(tempMatches1), kls1, kls2, kls2func, F21, TH_LOW);
        thread thread21(&LSDmatcher::FrameBFMatchNew, this, ldesc2, ldesc1, std ::ref(tempMatches2), kls2, kls1, kls1func, F12, TH_LOW);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches1.size(); i++)
        {
            int j = tempMatches1[i];
            if (j >= 0)
            {
                if (isDouble && tempMatches2[j] != i)
                    continue;

                if (pKF1->GetMapLine(i) || pKF2->GetMapLine(j))
                    continue;

                vMatchedPairs[i] = j;
                nmatches++;
            }
        }

        return nmatches;
    }

    cv::Mat LSDmatcher::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
    {
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R12 = R1w * R2w.t();
        cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

        cv::Mat t12x = SkewSymmetricMatrix(t12);

        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        return K1.t().inv() * t12x * R12 * K2.inv();
    }

    int LSDmatcher::Fuse(KeyFrame *pKF, const vector<MapLine *> &vpMapLines, float th)
    {
        cv::Mat Rcw = pKF->GetRotation();
        cv::Mat tcw = pKF->GetTranslation();

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        cv::Mat Ow = pKF->GetCameraCenter();

        int nFused = 0;

        Mat lineDesc = pKF->mLineDescriptors;
        const int nMLs = vpMapLines.size();

        for (int i = 0; i < nMLs; i++)
        {
            MapLine *pML = vpMapLines[i];

            if (!pML)
                continue;

            if (pML->isBad() || pML->IsInKeyFrame(pKF))
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const float &SPcX = SPc.at<float>(0);
            const float &SPcY = SPc.at<float>(1);
            const float &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const float &EPcX = EPc.at<float>(0);
            const float &EPcY = EPc.at<float>(1);
            const float &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                return false;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if (!pKF->IsInImage(u1, v1))
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            // Depth must be positive
            if (!pKF->IsInImage(u2, v2))
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const cv::Mat OM = 0.5 * (SP + EP) - Ow;
            const float dist = cv::norm(OM);

            if (dist < minDistance || dist > maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Vector3d Pn = pML->GetNormal();
            cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

            if (OM.dot(pn) < 0.5 * dist)
                continue;

            int nPredictedLevel = pML->PredictScale(dist, pKF->mfLogScaleFactorLine);

            // Search in a radius
            const float radius = th * pKF->mvScaleFactorsLine[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetLinesInArea(u1, v1, u2, v2, radius);

            if (vIndices.empty())
                continue;

            Mat CurrentLineDesc = pML->mLDescriptor; 
            int bestDist = 256;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                const KeyLine &kl = pKF->mvKeyLines[idx];

                const int &kpLevel = kl.octave;

                if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                if (CurrentLineDesc.empty() || dKF.empty())
                    continue;
                const int dist = DescriptorDistance(CurrentLineDesc, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist <= TH_LOW)
            {
                MapLine *pMLinKF = pKF->GetMapLine(bestIdx);

                if (pMLinKF)
                {
                    if (!pMLinKF->isBad())
                    {
                        if (pMLinKF->Observations() > pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                else
                {
                    pML->AddObservation(pKF, bestIdx);
                    pKF->AddMapLine(pML, bestIdx);
                }
                nFused++;
            }
        }
        return nFused;
    }

    float LSDmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if (viewCos > 0.998)
            return 5.0;
        else
            return 8.0;
    }
}
