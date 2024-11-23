/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"
#include "MapPlane.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    int img_width = fSettings["Camera.width"];
    int img_height = fSettings["Camera.height"];

    if((mask = imread("./masks/mask.png", cv::IMREAD_GRAYSCALE)).empty())
        mask = cv::Mat();

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //cout<<"111"<<endl;
    int nFeaturesLine = fSettings["LINEextractor.nFeatures"];
    float fScaleFactorLine = fSettings["LINEextractor.scaleFactor"];
    int nLevelsLine = fSettings["LINEextractor.nLevels"];
    int min_length = fSettings["LINEextractor.min_line_length"];

    mpLSDextractorLeft = new LINEextractor(nLevelsLine, fScaleFactorLine, nFeaturesLine, min_length);

    //cout<<"222"<<endl;
    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    // Initialization of global Manhattan variables     
    mpManh = new Manhattan(K);
    mManhInit = false;
    mCoarseManhInit = false;

    // Initialization of global time variables     
    mSumMTimeFeatExtract = 0.0;
    mSumMTimeEptsLineOpt = 0.0;
    mSumTimePoseEstim = 0.0;
    mTimeCoarseManh = 0.0;

    ///Initialization of plane
    mfDThRef = fSettings["Plane.AssociationDisRef"];
    mfAThRef = fSettings["Plane.AssociationAngRef"];
    mfVerTh = fSettings["Plane.VerticalThreshold"];
    mfParTh = fSettings["Plane.ParallelThreshold"];
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}
///modify by wh
    cv::Mat Tracking::GrabImageRGBD_wh(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
    {
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if(mImGray.channels()==3)
        {
            if(mbRGB)
                cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            else
                cvtColor(mImGray,mImGray,CV_BGR2GRAY);
        }
        else if(mImGray.channels()==4)
        {
            if(mbRGB)
                cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            else
                cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
        }

//        if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
//            imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

//cout<<"______"<<endl;
        // Extract Frame
        //mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft, mpLSDextractorLeft, mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth,mask, mCoarseManhInit);
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft, mpLSDextractorLeft, mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth,mask, mCoarseManhInit,mDepthMapFactor);


        cout<<"第 "<<mCurrentFrame.mnId<<" 帧！"<<endl;
        //cout<<"frame nl = "<<mCurrentFrame.NL<<endl;
        mSumMTimeFeatExtract +=  mCurrentFrame.MTimeFeatExtract;


        for (size_t k = 0; k < mCurrentFrame.mvLineEq.size(); k++)
        {
            std::vector<int> v_idx_perp;
            std::vector<int> v_idx_par;

            mCurrentFrame.mvPerpLinesIdx[k] = new std::vector<int>(v_idx_perp);
            mCurrentFrame.mvParLinesIdx[k] = new std::vector<int>(v_idx_par);

            // Discard non-valid lines
            ///3d线的归一化线函数  为什么c=0不行 暂时把c=0计算  将未赋值的淘汰
            if(mCurrentFrame.mvLineEq[k][0] == -1.0 && mCurrentFrame.mvLineEq[k][1] == -1.0 && mCurrentFrame.mvLineEq[k][2] == -1.0)
                continue;

//            if (mCurrentFrame.mvLineEq[k][2] == 0.0)
//                continue;


            ///计算当前帧中某个线段的平行与垂直约束
            mpManh->computeStructConstrains(mCurrentFrame, k, v_idx_par, v_idx_perp);


            mCurrentFrame.mvPerpLinesIdx[k] = new std::vector<int>(v_idx_perp);
            mCurrentFrame.mvParLinesIdx[k] = new std::vector<int>(v_idx_par);
        }

        ///取出相机坐标系下互相平行的两条线的方向向量  然后通过方向向量计算消影点  查看消影点是否相同
        ///能否利用消影点对平行线做进一步的约束

//        cout<<"NL = "<<mCurrentFrame.NL<<endl;
//        for(int i=0;i<mCurrentFrame.NL;i++)
//        {
//            auto line1 = mCurrentFrame.mvLines3D[i];
//            Vector3d a = line1.second - line1.first;
//            //cout<<"size = "<<mCurrentFrame.mvParLinesIdx[i]->size()<<endl;
//            for(int j=0;j<mCurrentFrame.mvParLinesIdx[i]->size();j++)
//            {
//                auto line2 = mCurrentFrame.mvLines3D[mCurrentFrame.mvParLinesIdx[i]->at(j)];
//                //Vector3d b = line2.second - line2.first;
//                Vector3d b = Converter::toVector3d(mCurrentFrame.mvLineEq[j]);
//                if(a.dot(b)<0)
//                {
//                    cout<<"两个方向向量不同向！"<<endl;
//                }
//                cout<<"i = "<<i<<" j = "<<mCurrentFrame.mvParLinesIdx[i]->at(j)<<endl;
//                Vector3d vp1 = Converter::toVector3d(mCurrentFrame.mK * Converter::toCvMat(a));
//                Vector3d vp2 = Converter::toVector3d(mCurrentFrame.mK * Converter::toCvMat(b));
//                vp1[0]/=vp1[2],vp1[1]/=vp1[2];
//                vp2[0]/=vp2[2],vp2[1]/=vp2[2];
//                vp1 = vp1 / sqrt(vp1[0]*vp1[0] + vp1[1]*vp1[1] );
//                vp2 = vp2 / sqrt(vp2[0]*vp2[0] + vp2[1]*vp2[1] );
//                cout<<"vp1 = "<<vp1<<endl;
//                cout<<"vp2 = "<<vp2<<endl;
//            }
//        }
//        getchar();
        ///


        std::chrono::steady_clock::time_point t_st_line_opt = std::chrono::steady_clock::now();

        // Optimize line-endpoints using structural constraints
        Optimizer::LineOptStruct(&mCurrentFrame);

        std::chrono::steady_clock::time_point t_end_line_opt = std::chrono::steady_clock::now();
        chrono::duration<double> time_line_opt = chrono::duration_cast<chrono::duration<double>>(t_end_line_opt - t_st_line_opt);
        mSumMTimeEptsLineOpt += time_line_opt.count();

        Track();



        return mCurrentFrame.mTcw.clone();
    }
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);
    
    // Extract Frame
    //mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft, mpLSDextractorLeft, mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth,mask, mCoarseManhInit);
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft, mpLSDextractorLeft, mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth,mask, mCoarseManhInit,mDepthMapFactor);

    mSumMTimeFeatExtract +=  mCurrentFrame.MTimeFeatExtract;
   ///modify by wh
   ///原理还是利用 v = Kd 求解方向向量为d的3d直线  只要直线不与像素平面平面，那v肯定会有
   ///应该使用坐标系下的3d线
//   cout<<"mCurrentFrame.mvLineEq.size() = "<<mCurrentFrame.mvLineEq.size()<<endl;
//   for(size_t k =0; k < mCurrentFrame.mvLineEq.size(); k++)
//   {
//       Vector3d vp;
//       cv::Vec3f line3d = mCurrentFrame.mvLineEq[k];
//       cv::Mat temp (cv::Size(1,3),CV_32FC1,line3d.val);
//       cout<<"line3d = "<<line3d<<endl;
//       cout<<"temp = "<<temp<<endl;
//       cv::Mat ans  = mCurrentFrame.mK * temp;
//       vp = Converter::toVector3d(ans);
//       cout<<"vp = "<<vp<<endl;
//   }

   ///   下面计算线的平行与垂直约束
    // For each extracted line get its parallel and perpendicular line correspondences
//    int sum_par=0,sum_per=0,sum1=0,sum2=0;
    int cnt =0;
    for (size_t k = 0; k < mCurrentFrame.mvLineEq.size(); k++)
    {
        std::vector<int> v_idx_perp;
        std::vector<int> v_idx_par;

        mCurrentFrame.mvPerpLinesIdx[k] = new std::vector<int>(v_idx_perp);
        mCurrentFrame.mvParLinesIdx[k] = new std::vector<int>(v_idx_par);

        // Discard non-valid lines
        ///3d线的归一化线函数
        if (mCurrentFrame.mvLineEq[k][2] == 0.0)
            continue;
        cnt++;
        ///modify by wh
        //if(mCurrentFrame.mvLineEq[k][2] == -1.0)continue;
        ///先从vp_line中找到k号线  line_vp表示k号线所对应的vp
//        int line_vp=-1;
//        for(unsigned long i=0;i<3;i++)
//        {
//            for(unsigned long j=0;j<mCurrentFrame.vp_line[i].size();j++)
//            {
//                if( (int)k == mCurrentFrame.vp_line[i][j])
//                {
//                    line_vp = (int)i;
//                    break;
//                }
//            }
//            if(line_vp != -1)break;
//        }
//        if(line_vp!=-1)
//        {
//            sum1++;
//            ///找到了k号线对应的vp  然后计算k号线的平行与垂直约束
//            ///先添加平行约束
//            for(unsigned long i=0;i<mCurrentFrame.vp_line[line_vp].size();i++)
//            {
//                if(mCurrentFrame.vp_line[line_vp][i] == (int)k)continue;
//                v_idx_par.push_back(mCurrentFrame.vp_line[line_vp][i]);
//            }
//            ///再添加垂直约束
//            for(unsigned long i=0;i<3;i++)
//            {
//                if((int)i == line_vp)continue;
//                for(unsigned long j=0;j<mCurrentFrame.vp_line[i].size();j++)
//                {
//                    v_idx_perp.push_back(mCurrentFrame.vp_line[i][j]);
//                }
//            }
//            ///上面是经过消失点计算的到的结构约束
////            cout<<"vanish point "<<v_idx_par.size()<<"   "<<v_idx_perp.size()<<endl;
////            v_idx_par.clear();
////            v_idx_perp.clear();
////            mpManh->computeStructConstrains(mCurrentFrame, k, v_idx_par, v_idx_perp);
////            cout<<"computeStructConstrains "<<v_idx_par.size()<<"   "<<v_idx_perp.size()<<endl;
//            ///
//        }
//        else
//        {
//            sum2++;
//            mpManh->computeStructConstrains(mCurrentFrame, k, v_idx_par, v_idx_perp);
//        }



        ///计算当前帧中某个线段的平行与垂直约束
        mpManh->computeStructConstrains(mCurrentFrame, k, v_idx_par, v_idx_perp);
//        //cout<<"line3d = "<<mCurrentFrame.mvLines3D.<<endl;
//        auto a = mCurrentFrame.mvLines3D[k];
//        cout<<"line3d = "<<a.first<<"   "<<a.second<<endl;
//        cout<<"与其平行的线 = "<<endl;
//        for(int j=0;j<v_idx_par.size();j++)
//        {
//            auto b = mCurrentFrame.mvLines3D[j];
//            cout<<"par line = "<<b.first<<"   "<<b.second<<endl;
//        }

        mCurrentFrame.mvPerpLinesIdx[k] = new std::vector<int>(v_idx_perp);
        mCurrentFrame.mvParLinesIdx[k] = new std::vector<int>(v_idx_par);
//        sum_par+=mCurrentFrame.mvParLinesIdx[k]->size();
//        sum_per+=mCurrentFrame.mvPerpLinesIdx[k]->size();

//        cout<<"line3d = "<<mCurrentFrame.mvLineEq[k]<<endl;
//        cout<<"与其平行的线 = "<<endl;
//        for(int j = 0;j<v_idx_par.size();j++)
//        {
//            cout<<"第 "<<j+1<<" 条平行线 = "<<mCurrentFrame.mvLineEq[v_idx_par[j]]<<endl;
//        }
        cv::Mat line_img = mImGray.clone();
        cvtColor(line_img,line_img,CV_GRAY2BGR);
        ///红色
        cv::line(line_img,mCurrentFrame.mvKeylinesUn[k].getStartPoint(),mCurrentFrame.mvKeylinesUn[k].getEndPoint(),cv::Scalar(0,0,255),2,CV_AA);
        for(int i=0;i<v_idx_par.size();i++)
        {
            int no = v_idx_par[i];
            ///平行的用绿色
            cv::line(line_img,mCurrentFrame.mvKeylinesUn[no].getStartPoint(),mCurrentFrame.mvKeylinesUn[no].getEndPoint(),cv::Scalar(0,255,0),2,CV_AA);
        }
        for(int i=0;i<v_idx_perp.size();i++)
        {
            int no = v_idx_perp[i];
            ///垂直的用蓝色
            cv::line(line_img,mCurrentFrame.mvKeylinesUn[no].getStartPoint(),mCurrentFrame.mvKeylinesUn[no].getEndPoint(),cv::Scalar(255,0,0),2,CV_AA);
        }
        //cv::imshow("wh",line_img);
        //getchar();
        //cv::destroyWindow("wh");
    }
    //cout<<"有"<<sum1<<"条线是经过vp算的约束  有"<<sum2<<"条线通过computeStructConstrains计算约束"<<endl;
    //cout<<"--------------------"<<endl;
    //cout<<"第"<<mCurrentFrame.mnId<<"帧";
    //cout<<"总共的平行和垂直约束一共有 "<<sum_par<<"   "<<sum_per<<endl;


    cout<<"计算了 "<<cnt<<" 条线的结构约束"<<endl;
    ///上面对frame的初始化中已经完成了根据消失点对线段的聚类 计划下一步完成  同一个方向聚类的线段一定是平行的
    std::chrono::steady_clock::time_point t_st_line_opt = std::chrono::steady_clock::now();

    // Optimize line-endpoints using structural constraints
    Optimizer::LineOptStruct(&mCurrentFrame);

    std::chrono::steady_clock::time_point t_end_line_opt = std::chrono::steady_clock::now();
    chrono::duration<double> time_line_opt = chrono::duration_cast<chrono::duration<double>>(t_end_line_opt - t_st_line_opt);
    mSumMTimeEptsLineOpt += time_line_opt.count();

    ///modify by wh
//    for(size_t k = 0;k < mCurrentFrame.mvLineEq.size(); k++)
//    {
//        if(mCurrentFrame.mvLineEq[k][2] == 0.0 || mCurrentFrame.mvLineEq[k][2] ==-1.0)continue;
//        cv::Vec3f line3d = mCurrentFrame.mvLineEq[k];
//        vector<int> idx = mCurrentFrame.mvParLinesIdx[k];
//        for(int j =0 ;j<mCurrentFrame.mvParLinesIdx[k]->size();j++)
//        {
//            cout<<"line3d = "<<line3d<<endl;
//            cout<<"与其平行的线 = "<<mCurrentFrame.mvLineEq[mCurrentFrame.mvParLinesIdx[k][j]]<<endl;
//        }
//    }

    Track();

    return mCurrentFrame.mTcw.clone();
}

bool Tracking::ExtractCoarseManhAx()
{
    // Compute the Coarse Manhattan Axis
    float succ_rate = -1.0;

    // TODO 0: Avoid this conversion to improve efficiency
    std::vector<cv::Mat> lines_vector;
    ///使用更鲁棒的线函数  存入lines_vector
    for (size_t i = 0; i < mCurrentFrame.mvLineEq.size(); i++)
    {
        if (mCurrentFrame.mvLineEq[i][2] == -1.0 ||
            mCurrentFrame.mvLineEq[i][2] == 1.0)
            continue;

        cv::Mat line_vector = (Mat_<double>(3, 1)
                                   << mCurrentFrame.mvLineEq[i][0],
                               mCurrentFrame.mvLineEq[i][1],
                               mCurrentFrame.mvLineEq[i][2]);

        lines_vector.push_back(line_vector);
    }

    // FindCoordAxis function produces a set of initial candidate directions by evaluating line vectors and point normals.
    // This directions are used to estimate the course Manh. axes.

    // TODO 0: mCurrentFrame.mRepLines and mCurrentFrame.mRepNormals contains a vector of vector. Mod.
    ///mRepLines存放的是合格的鲁棒的线函数
    std::vector<cv::Mat> coord_cand_lines;
    mpManh->findCoordAxis(mCurrentFrame.mRepLines, coord_cand_lines);

    std::vector<cv::Mat> coord_cand;
    mpManh->findCoordAxis(mCurrentFrame.mRepNormals, coord_cand);

    std::vector<cv::Mat> v_lines_n_normals(mCurrentFrame.mvPtNormals);
    v_lines_n_normals.insert(v_lines_n_normals.end(), lines_vector.begin(), lines_vector.end());

    std::vector<cv::Mat> v_lines_n_normals_cand(coord_cand);
    v_lines_n_normals_cand.insert(v_lines_n_normals_cand.end(), coord_cand_lines.begin(), coord_cand_lines.end());
    
    cv::Mat manh_axis;
    if (mpManh->extractCoarseManhAxes(v_lines_n_normals, v_lines_n_normals_cand, manh_axis, succ_rate) && succ_rate > 0.95)
    {
        // Assign the extracted Manh. axes to frame lines
        std::vector<int> line_axis_corresp;
        mpManh->LineManhAxisCorresp(manh_axis, mCurrentFrame.mvLineEq, line_axis_corresp);
        mCurrentFrame.vManhAxisIdx = line_axis_corresp;
        std::cerr << "EXTRACTED MANH AXES with succes rate: " << succ_rate  << std::endl;
        mpMap->SetWorldManhAxis(manh_axis.clone());
        mpLocalMapper->setManhAxis(manh_axis.clone());
        return true;
    } 
    return false;
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    static int count=0;
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
    {
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpLSDextractorLeft,mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth,mask);
    }
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpLSDextractorLeft,mpORBVocabulary,mpManh, mK,mDistCoef,mbf,mThDepth,mask);
    
    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
        {

            ///初始化第一帧的Rcm   可以理解为曼哈顿到世界坐标的一个旋转   在整个slam过程中不变
            Rotation_cm = cv::Mat::zeros(cv::Size(3, 3), CV_32F);

            std::chrono::steady_clock::time_point t_st_coarse_manh = std::chrono::steady_clock::now();

            ///粗提取mh
            if (ExtractCoarseManhAx())
                mCoarseManhInit = true;
            else
            {
                std::cerr << "WARNING -- Not able to seek manh init" << std::endl;
                // Assign lines to Manh. Axes 0, which means that do not correspond to a Manh. Axis
                std::vector<int> v_zeros(mCurrentFrame.mvLineEq.size(), 0);
                mCurrentFrame.vManhAxisIdx = v_zeros;
            }

            std::chrono::steady_clock::time_point t_end_coarse_manh = std::chrono::steady_clock::now();
            chrono::duration<double> time_coarse_manh = chrono::duration_cast<chrono::duration<double>>(t_end_coarse_manh - t_st_coarse_manh);
            mTimeCoarseManh += time_coarse_manh.count();
            ///初始化   这里的plane已经修改完成
            StereoInitialization();

            Rotation_cm = mpMap->FindManhattan(mCurrentFrame, mfVerTh, true);
            Rotation_cm = TrackManhattanFrame(Rotation_cm,mCurrentFrame.vSurfaceNormal,mCurrentFrame.mVF3DLines).clone();
            mLastRcm = Rotation_cm.clone();

            //cout<<"初始帧的曼哈顿 = "<<Rotation_cm<<endl;
        }
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // If coarse Manh. Axes has not been extracted, try to extract it
        if(!mCoarseManhInit)
            if (ExtractCoarseManhAx())
                mCoarseManhInit = true;

        // Evaluate if the Local Mapper has computed the fine Manhattan Axes
        if (!mManhInit && mCoarseManhInit)
        {
            cv::Mat opt_manh_wm;

            ///细模块 估计mh   这里其实只是一个判断函数 真正的优化函数在local mapping线程中
            if (mpLocalMapper->optManhInitAvailable(opt_manh_wm))
            {
                mManhInit = true; 
                // Update the coarse Manh. axes from the map     
                mpMap->SetWorldManhAxis(opt_manh_wm.clone());
                
            }
        }

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        std::chrono::steady_clock::time_point t_st_cam_pose_estim = std::chrono::steady_clock::now();
        if(!mbOnlyTracking)
        {

            ///MF_can是当前帧中的ma到当前帧的旋转
            cv::Mat MF_can = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
            ///转置
            cv::Mat MF_can_T = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
            MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();
            ///将当前帧的ma更新为上一帧的ma
            MF_can.copyTo(mLastRcm);

            //cout<<"mnid = "<<mCurrentFrame.mnId<<" ma = "<<MF_can<<endl;
            MF_can_T = MF_can.t();
            ///这里的Rotation_cm是在初始化时得到的一个ma  是从第一个ma到第一帧的旋转  但这个旋转固定了
            ///MF_can存储的是曼哈顿到当前帧的一个旋转
            mRotation_wc = Rotation_cm * MF_can_T;
            ///没错 后面给pose赋值的时候使用的还是转置

            ///planar slam中使用 mRotation_wc给当前帧的Tcw赋值
            mRotation_wc = mRotation_wc.t();

            ///上一帧的旋转
            cv::Mat Rlw = mLastFrame.mTcw.rowRange(0,3).colRange(0,3);
            ///上一帧到当前帧的旋转
            coarseRcl = mRotation_wc * Rlw.t();


            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            ///track
                // Local Mapping might have changed some MapPoints tracked in last frame
                /// plane添加完成
                CheckReplacedInLastFrame();
           
                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    ///已经看过了   plane 已修改
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

                    ///调用的优化函数都是一样的  plane 已修改
                    ///后面返回时参数有点模糊  暂时放开
                    bOK = TrackWithMotionModel();

                    if(!bOK)
                        ///plane 已修改
                        bOK = TrackReferenceKeyFrame();
                }
                ///track
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                bOK = TrackLocalMapWithLines();

            }
            ///track
            else bOK = Relocalization();

        }


        // update rotation from manhattan
        ///new_Rotation_wc == Rwc
        cv::Mat new_Rotation_wc = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
        ///Rotation_mc == Rmw
        cv::Mat Rotation_mc = Rotation_cm.t();
        cv::Mat MF_can_T;
        ///MF_can_T == Rmw * Rwc   c-w-m   c-m  == Rmc
        MF_can_T = Rotation_mc * new_Rotation_wc;
        mLastRcm = MF_can_T.t();

        std::chrono::steady_clock::time_point t_end_cam_pose_estim  = std::chrono::steady_clock::now();
        chrono::duration<double> t_time_cam_pose_est = chrono::duration_cast<chrono::duration<double>>(t_end_cam_pose_estim -t_st_cam_pose_estim);
        mSumTimePoseEstim += t_time_cam_pose_est.count();

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        ///modify by wh
        ///计算地图点与平面的距离
        mpMap->FlagMatchedPlanePoints(mCurrentFrame, mfDThRef);


        ///modify by wh
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
            MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
            if (pMP) {
                ///暂时不清楚这一步在干什么
                pMP->UpdateCoefficientsAndPoints(mCurrentFrame, i);
            } else if (!mCurrentFrame.mvbPlaneOutlier[i]) {
                mCurrentFrame.mbNewPlane = true;
            }
        }


        // If tracking was good, check if it should be inserted as a keyframe
        if(bOK)
        {
            // Assign the Manh. axes to each line. If not assign 0, which means non-associated axis
            if (!(mpMap->GetWorldManhAxis().empty()))
            {
                cv::Mat frame_mRcw;
                mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).copyTo(frame_mRcw);
                frame_mRcw.convertTo(frame_mRcw, CV_64F);
                std::vector<int> line_axis_corresp(mCurrentFrame.mvLineEq.size(), 0);
                mpManh->LineManhAxisCorresp(frame_mRcw, mCurrentFrame.mvLineEq, line_axis_corresp);
                mCurrentFrame.vManhAxisIdx = line_axis_corresp;
            }
            else
            {
                std::vector<int> v_zeros(mCurrentFrame.mvLineEq.size(), 0);
                mCurrentFrame.vManhAxisIdx = v_zeros;
            }

            // Update motion model
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;

            }
            else
                mVelocity = cv::Mat();



            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
            }

            for (int i = 0; i < mCurrentFrame.NL; i++)
            {
                MapLine *pML = mCurrentFrame.mvpMapLines[i];
                if (pML)
                    if (pML->Observations() < 1)
                    {
                        mCurrentFrame.mvbLineOutlier[i] = false;
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    }
            }
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
                if (pMP)
                    if (pMP->Observations() < 1) {
                        mCurrentFrame.mvbPlaneOutlier[i] = false;
                        mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(NULL);
                    }

                MapPlane *pVMP = mCurrentFrame.mvpVerticalPlanes[i];
                if (pVMP)
                    if (pVMP->Observations() < 1) {
                        mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                        mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(NULL);
                    }

                MapPlane *pPMP = mCurrentFrame.mvpParallelPlanes[i];
                if (pVMP)
                    if (pVMP->Observations() < 1) {
                        mCurrentFrame.mvbParPlaneOutlier[i] = false;
                        mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();
            // Check if we need to insert a new keyframe
            if (NeedNewKeyFrame())
            {
                //cout<<"current id = "<<mCurrentFrame.mnId<<endl;
                CreateNewKeyFrame();
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }

            for(int i=0; i<mCurrentFrame.NL; i++)
            {
                if(mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                    mCurrentFrame.mvpMapLines[i]= static_cast<MapLine*>(NULL);
            }
        }
        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }
        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        ///copy
        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}

    axiSNV Tracking::ProjectSN2Conic(int a, const cv::Mat &R_mc, const vector<SurfaceNormal> &vTempSurfaceNormal,
                                     vector<FrameLine> &vVanishingDirection) {
        int numInConic = 0;
        vector<cv::Point2d> m_j_selected;
        cv::Mat R_cm_Rec = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
        cv::Mat R_cm_NULL = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        //cv::Mat R_mc=cv::Mat::zeros(cv::Size(3,3),CV_32F);
        vector<SurfaceNormal> vSNCadidate;
        axiSNV tempaxiSNV;
        tempaxiSNV.axis = a;


        size_t sizeOfSurfaceNormal = vTempSurfaceNormal.size() + vVanishingDirection.size();
        tempaxiSNV.SNVector.reserve(sizeOfSurfaceNormal);
//        cout << "size of SN" << sizeOfSurfaceNormal << endl;
        for (size_t i = 0; i < sizeOfSurfaceNormal; i++) {

            cv::Point3f n_ini;
            if (i < vTempSurfaceNormal.size()) {
                n_ini.x = R_mc.at<float>(0, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(0, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(0, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.y = R_mc.at<float>(1, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(1, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(1, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.z = R_mc.at<float>(2, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(2, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(2, 2) * vTempSurfaceNormal[i].normal.z;

                double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
                //cout<<lambda<<endl;
                if (lambda < sin(0.2018)) //0.25
                {

                    //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                    //numInConic++;
                    tempaxiSNV.SNVector.push_back(vTempSurfaceNormal[i]);


                }
            } else {   //cout<<"vanishing"<<endl;
                int tepSize = i - vTempSurfaceNormal.size();
                //cout<<vVanishingDirection[tepSize].direction.x<<"vanishing"<<endl;

                n_ini.x = R_mc.at<float>(0, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(0, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(0, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.y = R_mc.at<float>(1, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(1, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(1, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.z = R_mc.at<float>(2, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(2, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(2, 2) * vVanishingDirection[tepSize].direction.z;

                double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
                //cout<<lambda<<endl;
                if (lambda < sin(0.1018)) //0.25
                {

                    //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                    //numInConic++;
                    tempaxiSNV.Linesvector.push_back(vVanishingDirection[tepSize]);


                }

            }


        }

        return tempaxiSNV;//numInConic;

    }

    ResultOfMS Tracking::ProjectSN2MF(int a, const cv::Mat &R_mc, const vector<SurfaceNormal> &vTempSurfaceNormal,
                                      vector<FrameLine> &vVanishingDirection, const int numOfSN) {
        vector<cv::Point2d> m_j_selected;
        cv::Mat R_cm_Rec = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        cv::Mat R_cm_NULL = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        ResultOfMS RandDen;
        RandDen.axis = a;

        size_t sizeOfSurfaceNormal = vTempSurfaceNormal.size() + vVanishingDirection.size();
        m_j_selected.reserve(sizeOfSurfaceNormal);

        for (size_t i = 0; i < sizeOfSurfaceNormal; i++) {
            //cv::Mat temp=cv::Mat::zeros(cv::Size(1,3),CV_32F);

            cv::Point3f n_ini;
            int tepSize = i - vTempSurfaceNormal.size();
            if (i >= vTempSurfaceNormal.size()) {

                n_ini.x = R_mc.at<float>(0, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(0, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(0, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.y = R_mc.at<float>(1, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(1, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(1, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.z = R_mc.at<float>(2, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(2, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(2, 2) * vVanishingDirection[tepSize].direction.z;
            } else {

                n_ini.x = R_mc.at<float>(0, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(0, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(0, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.y = R_mc.at<float>(1, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(1, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(1, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.z = R_mc.at<float>(2, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(2, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(2, 2) * vTempSurfaceNormal[i].normal.z;
            }


            double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
            //cout<<lambda<<endl;
            //inside the cone
            if (lambda < sin(0.2518)) //0.25
            {
                double tan_alfa = lambda / std::abs(n_ini.z);
                double alfa = asin(lambda);
                double m_j_x = alfa / tan_alfa * n_ini.x / n_ini.z;
                double m_j_y = alfa / tan_alfa * n_ini.y / n_ini.z;
                if (!std::isnan(m_j_x) && !std::isnan(m_j_y))
                    m_j_selected.push_back(cv::Point2d(m_j_x, m_j_y));
                if (i < vTempSurfaceNormal.size()) {
                    if (a == 1) {
                        mCurrentFrame.vSurfaceNormalx.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointx.push_back(vTempSurfaceNormal[i].cameraPosition);
                    } else if (a == 2) {
                        mCurrentFrame.vSurfaceNormaly.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointy.push_back(vTempSurfaceNormal[i].cameraPosition);
                    } else if (a == 3) {
                        mCurrentFrame.vSurfaceNormalz.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointz.push_back(vTempSurfaceNormal[i].cameraPosition);
                    }
                } else {
                    if (a == 1) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinex.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCx.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    } else if (a == 2) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLiney.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCy.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    } else if (a == 3) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinez.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCz.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    }
                }


            }
        }
        //cout<<"a=1:"<<mCurrentFrame.vSurfaceNormalx.size()<<",a =2:"<<mCurrentFrame.vSurfaceNormaly.size()<<", a=3:"<<mCurrentFrame.vSurfaceNormalz.size()<<endl;
        //cout<<"m_j_selected.push_back(temp)"<<m_j_selected.size()<<endl;

        if (m_j_selected.size() > numOfSN) {
            sMS tempMeanShift = MeanShift(m_j_selected);
            cv::Point2d s_j = tempMeanShift.centerOfShift;// MeanShift(m_j_selected);
            float s_j_density = tempMeanShift.density;
            //cout<<"tracking:s_j"<<s_j.x<<","<<s_j.y<<endl;
            float alfa = norm(s_j);
            float ma_x = tan(alfa) / alfa * s_j.x;
            float ma_y = tan(alfa) / alfa * s_j.y;
            cv::Mat temp1 = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
            temp1.at<float>(0, 0) = ma_x;
            temp1.at<float>(1, 0) = ma_y;
            temp1.at<float>(2, 0) = 1;
            cv::Mat rtemp = R_mc.t();
            R_cm_Rec = rtemp * temp1;
            R_cm_Rec = R_cm_Rec / norm(R_cm_Rec); //列向量
            RandDen.R_cm_Rec = R_cm_Rec;
            RandDen.s_j_density = s_j_density;

            return RandDen;
        }
        RandDen.R_cm_Rec = R_cm_NULL;
        return RandDen;

    }
    sMS Tracking::MeanShift(vector<cv::Point2d> &v2D) {
        sMS tempMS;
        int numPoint = v2D.size();
        float density;
        cv::Point2d nominator;
        double denominator = 0;
        double nominator_x = 0;
        double nominator_y = 0;
        for (int i = 0; i < numPoint; i++) {
            double k = exp(-20 * norm(v2D.at(i)) * norm(v2D.at(i)));
            nominator.x += k * v2D.at(i).x;
            nominator.y += k * v2D.at(i).y;
            denominator += k;
        }
        tempMS.centerOfShift = nominator / denominator;
        tempMS.density = denominator / numPoint;

        return tempMS;
    }

    cv::Mat Tracking::TrackManhattanFrame(cv::Mat &mLastRcm, vector<SurfaceNormal> &vSurfaceNormal,
                                          vector<FrameLine> &vVanishingDirection){
        //cout << "begin Tracking" << endl;
        ///将粗略的曼哈顿估计作为初始值
        //cout<<"mLastRcm = "<<mLastRcm<<endl;
        cv::Mat R_cm_update = mLastRcm.clone();
        int isTracked = 0;
        vector<double> denTemp(3, 0.00001);
        for (int i = 0; i <1; i++) {

            cv::Mat R_cm = R_cm_update;//cv::Mat::eye(cv::Size(3,3),CV_32FC1);  // 对角线为1的对角矩阵(3, 3, CV_32FC1);
            int directionFound1 = 0;
            int directionFound2 = 0;
            int directionFound3 = 0; //三个方向
            int numDirectionFound = 0;
            vector<axiSNVector> vaxiSNV(4);
            vector<int> numInCone = vector<int>(3, 0);
            vector<cv::Point2f> vDensity;
            //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            for (int a = 1; a < 4; a++) {
                //在每个conic有多少 点
                cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                int c1 = (a + 3) % 3;
                int c2 = (a + 4) % 3;
                int c3 = (a + 5) % 3;
                R_mc.at<float>(0, 0) = R_cm.at<float>(0, c1);
                R_mc.at<float>(0, 1) = R_cm.at<float>(0, c2);
                R_mc.at<float>(0, 2) = R_cm.at<float>(0, c3);
                R_mc.at<float>(1, 0) = R_cm.at<float>(1, c1);
                R_mc.at<float>(1, 1) = R_cm.at<float>(1, c2);
                R_mc.at<float>(1, 2) = R_cm.at<float>(1, c3);
                R_mc.at<float>(2, 0) = R_cm.at<float>(2, c1);
                R_mc.at<float>(2, 1) = R_cm.at<float>(2, c2);
                R_mc.at<float>(2, 2) = R_cm.at<float>(2, c3);
                cv::Mat R_mc_new = R_mc.t();
//                cout << "R_mc_new" << R_mc_new << endl;
                vaxiSNV[a - 1] = ProjectSN2Conic(a, R_mc_new, vSurfaceNormal, vVanishingDirection);
                numInCone[a - 1] = vaxiSNV[a - 1].SNVector.size();
                //cout<<"2 a:"<<vaxiSNV[a-1].axis<<",vector:"<<numInCone[a - 1]<<endl;
            }
            //chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
            //cout << "first sN time: " << time_used.count() << endl;
            int minNumOfSN = vSurfaceNormal.size() / 20;
            //cout<<"minNumOfSN"<<minNumOfSN<<endl;
            //排序  a<b<c
            int a = numInCone[0];
            int b = numInCone[1];
            int c = numInCone[2];
            //cout<<"a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
            int temp = 0;
            if (a > b) temp = a, a = b, b = temp;
            if (b > c) temp = b, b = c, c = temp;
            if (a > b) temp = a, a = b, b = temp;
            //cout<<"sequence  a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
            if (b < minNumOfSN) {
                minNumOfSN = (b + a) / 2;
                cout << "thr" << minNumOfSN << endl;
            }

            //cout<<"new  minNumOfSN"<<minNumOfSN<<endl;
            //chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
            for (int a = 1; a < 4; a++) {
                cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                int c1 = (a + 3) % 3;
                int c2 = (a + 4) % 3;
                int c3 = (a + 5) % 3;
                R_mc.at<float>(0, 0) = R_cm.at<float>(0, c1);
                R_mc.at<float>(0, 1) = R_cm.at<float>(0, c2);
                R_mc.at<float>(0, 2) = R_cm.at<float>(0, c3);
                R_mc.at<float>(1, 0) = R_cm.at<float>(1, c1);
                R_mc.at<float>(1, 1) = R_cm.at<float>(1, c2);
                R_mc.at<float>(1, 2) = R_cm.at<float>(1, c3);
                R_mc.at<float>(2, 0) = R_cm.at<float>(2, c1);
                R_mc.at<float>(2, 1) = R_cm.at<float>(2, c2);
                R_mc.at<float>(2, 2) = R_cm.at<float>(2, c3);
                cv::Mat R_mc_new = R_mc.t();
                vector<SurfaceNormal> *tempVVSN;
                vector<FrameLine> *tempLineDirection;
                for (int i = 0; i < 3; i++) {
                    if (vaxiSNV[i].axis == a) {

                        tempVVSN = &vaxiSNV[i].SNVector;
                        tempLineDirection = &vaxiSNV[i].Linesvector;
                        break;
                    }

                }

                ResultOfMS RD_temp = ProjectSN2MF(a, R_mc_new, *tempVVSN, *tempLineDirection, minNumOfSN);

                //chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
                //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t4-t3);
                //cout << "second SN time: " << time_used.count() << endl;

                //cout << "test projectSN2MF" << ra << endl;
                if (sum(RD_temp.R_cm_Rec)[0] != 0) {
                    numDirectionFound += 1;
                    if (a == 1) directionFound1 = 1;//第一个轴
                    else if (a == 2) directionFound2 = 1;
                    else if (a == 3) directionFound3 = 1;
                    R_cm_update.at<float>(0, a - 1) = RD_temp.R_cm_Rec.at<float>(0, 0);
                    R_cm_update.at<float>(1, a - 1) = RD_temp.R_cm_Rec.at<float>(1, 0);
                    R_cm_update.at<float>(2, a - 1) = RD_temp.R_cm_Rec.at<float>(2, 0);
                    //RD_temp.s_j_density;

                    vDensity.push_back(cv::Point2f(RD_temp.axis, RD_temp.s_j_density));

                }
            }

            if (numDirectionFound < 2) {
                cout << "oh, it has happened" << endl;
                R_cm_update = R_cm;
                numDirectionFound = 0;
                isTracked = 0;
                directionFound1 = 0;
                directionFound2 = 0;
                directionFound3 = 0;
                break;
            } else if (numDirectionFound == 2) {
                if (directionFound1 && directionFound2) {
                    cv::Mat v1 = R_cm_update.colRange(0, 1).clone();
                    cv::Mat v2 = R_cm_update.colRange(1, 2).clone();
                    cv::Mat v3 = v1.cross(v2);
                    R_cm_update.at<float>(0, 2) = v3.at<float>(0, 0);
                    R_cm_update.at<float>(1, 2) = v3.at<float>(1, 0);
                    R_cm_update.at<float>(2, 2) = v3.at<float>(2, 0);
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 2) = -v3.at<float>(0, 0);
                        R_cm_update.at<float>(1, 2) = -v3.at<float>(1, 0);
                        R_cm_update.at<float>(2, 2) = -v3.at<float>(2, 0);
                    }

                } else if (directionFound2 && directionFound3) {
                    cv::Mat v2 = R_cm_update.colRange(1, 2).clone();
                    cv::Mat v3 = R_cm_update.colRange(2, 3).clone();
                    cv::Mat v1 = v3.cross(v2);
                    R_cm_update.at<float>(0, 0) = v1.at<float>(0, 0);
                    R_cm_update.at<float>(1, 0) = v1.at<float>(1, 0);
                    R_cm_update.at<float>(2, 0) = v1.at<float>(2, 0);
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 0) = -v1.at<float>(0, 0);
                        R_cm_update.at<float>(1, 0) = -v1.at<float>(1, 0);
                        R_cm_update.at<float>(2, 0) = -v1.at<float>(2, 0);
                    }
                } else if (directionFound1 && directionFound3) {
                    cv::Mat v1 = R_cm_update.colRange(0, 1).clone();
                    cv::Mat v3 = R_cm_update.colRange(2, 3).clone();
                    cv::Mat v2 = v1.cross(v3);
                    R_cm_update.at<float>(0, 1) = v2.at<float>(0, 0);
                    R_cm_update.at<float>(1, 1) = v2.at<float>(1, 0);
                    R_cm_update.at<float>(2, 1) = v2.at<float>(2, 0);
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 1) = -v2.at<float>(0, 0);
                        R_cm_update.at<float>(1, 1) = -v2.at<float>(1, 0);
                        R_cm_update.at<float>(2, 1) = -v2.at<float>(2, 0);
                    }

                }
            }
            //cout<<"svd before"<<R_cm_update<<endl;
            SVD svd;
            cv::Mat U, W, VT;

            svd.compute(R_cm_update, W, U, VT);

            R_cm_update = U* VT;
            vDensity.clear();
            if (acos((trace(R_cm.t() * R_cm_update)[0] - 1.0)) / 2 < 0.001) {
                cout << "go outside" << endl;
                break;
            }
        }
        isTracked = 1;
        return R_cm_update.clone();
    }

void Tracking::StereoInitialization()
{
    //if(mCurrentFrame.N + mCurrentFrame.NL > 100)
    if (mCurrentFrame.N > 50 || mCurrentFrame.NL > 15)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }
        for(int i=0; i<mCurrentFrame.NL;i++)
        {
            ///拟只是用结构线
            //if(mCurrentFrame.isStructLine[i] == false)continue;
            if(mCurrentFrame.mvKeylinesUn[i].startPointX > 0.0 && mCurrentFrame.mvLines3D[i].first.x() !=0)
            {
                Eigen::Vector3d st_3D_w = mCurrentFrame.PtToWorldCoord(mCurrentFrame.mvLines3D[i].first);
                Eigen::Vector3d e_3D_w = mCurrentFrame.PtToWorldCoord(mCurrentFrame.mvLines3D[i].second);
                Vector6d w_l_endpts;
                w_l_endpts << st_3D_w.x(), st_3D_w.y(),st_3D_w.z(),
                e_3D_w.x(), e_3D_w.y(),e_3D_w.z();

                MapLine* pML = new MapLine(w_l_endpts,mCurrentFrame.vManhAxisIdx[i], pKFini, mpMap);
                pML->AddObservation(pKFini, i);
                pKFini->AddMapLine(pML, i);
                pML->ComputeDistinctiveDescriptors();
                pML->UpdateManhAxis();
                mpMap->AddMapLine(pML);
                mCurrentFrame.mvpMapLines[i]=pML;
            }
        }
        /// add plane
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
            cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
            MapPlane *pNewMP = new MapPlane(p3D, pKFini, mpMap);
            pNewMP->AddObservation(pKFini,i);
            pKFini->AddMapPlane(pNewMP, i);
            pNewMP->UpdateCoefficientsAndPoints();
            mpMap->AddMapPlane(pNewMP);
            mCurrentFrame.mvpMapPlanes[i] = pNewMP;
        }

//        cout << "KF0 - New map created with " << mpMap->MapPointsInMap() << " points" << endl;
//        cout << "KF0 - New map created with " << mpMap->GetAllMapLines().size() << " lines" << endl;
//        cout<<" KF0 - New map created with "<<mpMap->GetAllMapPlanes().size()<<" planes "<<endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mvpLocalMapLines=mpMap->GetAllMapLines();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{
    int num = 100;
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>num)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvIniLineMatches.begin(),mvIniLineMatches.end(),-1);
            mvIniLastLineMatches = vector<int>(mCurrentFrame.mvKeys.size(), -1);

            mbIniFirst = false;

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=num)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvIniLineMatches.begin(),mvIniLineMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        LSDmatcher lmatcher;   
        int lineMatches = lmatcher.SearchDouble(mLastFrame, mCurrentFrame, mvIniLineMatches);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        if(!mbIniFirst)
        {
            mvIniLastLineMatches = mvIniLineMatches;
            mbIniFirst = true;
        }else{
            for(int i = 0; i < mInitialFrame.mvKeys.size(); i++)
            {
                int j = mvIniLastLineMatches[i];
                if(j >= 0 ){
                    mvIniLastLineMatches[i] = mvIniLineMatches[j];
                }
            }

            lmatcher.SearchDouble(mInitialFrame,mCurrentFrame, mvIniLineMatches);
            for(int i = 0; i < mInitialFrame.mvKeys.size(); i++)
            {
                int j = mvIniLastLineMatches[i];
                int k = mvIniLineMatches[i];
                if(j != k){
                    mvIniLastLineMatches[i] = -1;
                }
            }
        }

        mvIniLineMatches = mvIniLastLineMatches;

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        vector<bool> mvbLineTriangulated; 

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, mvIniLineMatches, mvLineS3D, mvLineE3D, mvbLineTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            for(size_t i=0, iend=mvIniLineMatches.size(); i<iend;i++)
            {
                if(mvIniLineMatches[i]>=0 && !mvbLineTriangulated[i])
                {
                    mvIniLineMatches[i]=-1;
                    lineMatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);
           
            CreateInitialMapMonoWithLine();
        }

        mLastFrame = Frame(mCurrentFrame);
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        mpMap->AddMapPoint(pMP);
    }

    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
    mState=OK;  
}

void Tracking::CreateInitialMapMonoWithLine()
{
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size(); i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        // Create MapPoint
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        // Add to Map
        mpMap->AddMapPoint(pMP);
    }

    for(size_t i=0; i<mvIniLineMatches.size(); i++)
    {
        if(mvIniLineMatches[i] < 0)
            continue;

        // Create MapLine
        Vector6d worldPos;
        worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;

        int manh_idx = 0;
        MapLine* pML = new MapLine(worldPos, manh_idx, pKFcur, mpMap);

        pKFini->AddMapLine(pML,i);
        pKFcur->AddMapLine(pML,mvIniLineMatches[i]);

        pML->AddObservation(pKFini, i);
        pML->AddObservation(pKFcur, mvIniLineMatches[i]);

        pML->ComputeDistinctiveDescriptors();

        pML->UpdateAverageDir();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapLines[mvIniLineMatches[i]] = pML;
        mCurrentFrame.mvbLineOutlier[mvIniLineMatches[i]] = false;

        // step5.4: Add to Map
        mpMap->AddMapLine(pML);
    }

    cout << "this Map created with " << mpMap->MapPointsInMap() << " points, and "<< mpMap->MapLinesInMap() << " lines." << endl;
    Optimizer::GlobalBundleAdjustemnt(mpMap, 20, true); 
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    cout << "medianDepth = " << medianDepth << endl;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<80)
    {
        cout << "Wrong initialization, reseting ... " << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale Points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    // Scale Line Segments
    vector<MapLine*> vpAllMapLines = pKFini->GetMapLineMatches();
    for(size_t iML=0; iML < vpAllMapLines.size(); iML++)
    {
        if(vpAllMapLines[iML])
        {
            MapLine* pML = vpAllMapLines[iML];
            pML->SetWorldPos(pML->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mvpLocalMapLines = mpMap->GetAllMapLines();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->SetReferenceMapLines(mvpLocalMapLines);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
    for(int i=0; i<mLastFrame.NL; i++)
    {
        MapLine* pML = mLastFrame.mvpMapLines[i];

        if(pML)
        {
            MapLine* pReL = pML->GetReplaced();
            if(pReL)
            {
                mLastFrame.mvpMapLines[i] = pReL;
            }
        }
    }

    ///modify by wh
    for (int i = 0; i < mLastFrame.mnPlaneNum; i++) {
        MapPlane *pMP = mLastFrame.mvpMapPlanes[i];

        if (pMP) {
            MapPlane *pRep = pMP->GetReplaced();
            if (pRep) {
                mLastFrame.mvpMapPlanes[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{   
    //cout<<"[Debug] Calling TrackReferenceKeyFrameWithLine(), mCurrentFrame.mnId:"<<mCurrentFrame.mnId<<endl;

    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
    
    fill(mCurrentFrame.mvpMapLines.begin(),mCurrentFrame.mvpMapLines.end(),static_cast<MapLine*>(NULL));

    LSDmatcher lmatcher(0.85, true);
    ///modify by wh
    PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

    //cout<<"debug debug debug11111111"<<endl;
    ///特征点匹配
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    int lmatches = 0;

    std::vector<int> matches_12;
    ///线特征匹配
    ///虽然是参考关键帧跟踪 但对线特征的匹配却是上一帧与当前帧   有待商榷。。。
    ///应该是因为只有在第一帧或者重定位之后才会使用 TrackReferenceKeyFrame  所以直接用当前帧和上一帧没什么问题
    ///通过描述子匹配
    int nl_matches = lmatcher.match(mLastFrame.mLdesc, mCurrentFrame.mLdesc, 0.9, matches_12);

//    ///画出上一帧与当前帧之间的线特征匹配关系
//    cv::Mat img1 = mLastFrame.ImageGray;
//    cv::Mat img2 = mCurrentFrame.ImageGray;
//    cvtColor(img1,img1,CV_GRAY2BGR);
//    cvtColor(img2,img2,CV_GRAY2BGR);
//    for(int i=0;i<mLastFrame.mvKeylinesUn.size();i++)
//    {
//        KeyLine l = mLastFrame.mvKeylinesUn[i];
//        cv::line(img1,l.getStartPoint(),l.getEndPoint(),cv::Scalar(0,255,0),1);
//    }
//    for(int i=0;i<mCurrentFrame.mvKeylinesUn.size();i++)
//    {
//        KeyLine l = mCurrentFrame.mvKeylinesUn[i];
//        cv::line(img2,l.getStartPoint(),l.getEndPoint(),cv::Scalar(0,165,255),1);
//    }
//    cv::Mat combine_img;
//    cvtColor(combine_img,combine_img,CV_GRAY2BGR);
//    cv::hconcat(img1,img2,combine_img);
//    for(int i1=0;i1<matches_12.size();i1++)
//    {
//        cv::Mat temp = combine_img.clone();
//        KeyLine a = mLastFrame.mvKeylinesUn[i1];
//        int i2 = matches_12[i1];
//        KeyLine b = mCurrentFrame.mvKeylinesUn[i2];
//        //cv::line(combine_img,a.getStartPoint(),a.getEndPoint(),cv::Scalar(0,255,0),1);
//        cv::Point2f a_mid = (a.getEndPoint() + a.getStartPoint())*0.5;
//        cv::Point2f b_mid = (b.getEndPoint() + b.getStartPoint())*0.5;
//        b_mid.x += img1.cols;
//        //cv::line(combine_img,cv::Point2f(b.startPointX+(float)img1.cols,b.startPointY+(float)img1.rows),cv::Point2f(b.endPointX+(float)img1.cols,b.endPointY+(float)img1.rows),cv::Scalar(255,0,0),1);
//        //cv::line(combine_img,cv::Point2f(b.startPointX+(float)img1.cols,b.startPointY),cv::Point2f(b.endPointX+(float)img1.cols,b.endPointY),cv::Scalar(255,0,0),1);
//        //cv::line(combine_img,cv::Point2f(b.startPointX,b.startPointY),cv::Point2f(b.endPointX,b.endPointY),cv::Scalar(255,0,0),1);
//
//        cv::line(combine_img,a_mid,b_mid,cv::Scalar(0,0,255),1);
//        //cv::circle(combine_img,a_mid,1,cv::Scalar(0,0,255),1);
//        //cv::circle(combine_img,b_mid,1,cv::Scalar(0,0,128),1);
//        //cv::imshow("ly",temp);
//        //getchar();
//    }
//    cv::imshow("wh",combine_img);
//    getchar();
//    cv::destroyWindow("wh");

    int flow_num = lmatcher.optical_flow_line(mCurrentFrame,mLastFrame,matches_12,coarseRcl);

    const double deltaAngle = M_PI/8.0;
    const double deltaWidth = (mCurrentFrame.mnMaxX-mCurrentFrame.mnMinX)*0.1;
    const double deltaHeight = (mCurrentFrame.mnMaxY-mCurrentFrame.mnMinY)*0.1;
    int delta_angle = 0;
    int delta_pose = 0;
    int not_found = 0;
    int i2_var = 0;
    const int nmatches_12 = matches_12.size();
    for (int i1 = 0; i1 < nmatches_12; ++i1) {
        ///匹配的线在上一帧中没有映射到地图中
        if(!mLastFrame.mvpMapLines[i1]){
          not_found ++;
          continue;
        }
        const int i2 = matches_12[i1];
        ///匹配不成功就是-1
        if (i2 < 0)
        {
            i2_var ++;
            continue;
        }
        ///这个过滤....
        if(mCurrentFrame.mvKeylinesUn[i2].startPointX == 0) continue;

        // Evaluate orientation and position in image
        if(true) {
            // Orientation
            double theta = mCurrentFrame.mvKeylinesUn[i2].angle-mLastFrame.mvKeylinesUn[i1].angle;
            if(theta<-M_PI) theta+=2*M_PI;
            else if(theta>M_PI) theta-=2*M_PI;
            ///角度差的太多 标记为误匹配
            if(fabs(theta)>deltaAngle) {
                matches_12[i1] = -1;
                delta_angle++;
                continue;
            }

            // Position
            const float& sX_curr = mCurrentFrame.mvKeylinesUn[i2].startPointX;
            const float& sX_last = mLastFrame.mvKeylinesUn[i1].startPointX;
            const float& sY_curr = mCurrentFrame.mvKeylinesUn[i2].startPointY;
            const float& sY_last = mLastFrame.mvKeylinesUn[i1].startPointY;
            const float& eX_curr = mCurrentFrame.mvKeylinesUn[i2].endPointX;
            const float& eX_last = mLastFrame.mvKeylinesUn[i1].endPointX;
            const float& eY_curr = mCurrentFrame.mvKeylinesUn[i2].endPointY;
            const float& eY_last = mLastFrame.mvKeylinesUn[i1].endPointY;
            ///位置也不能差太多
            if(fabs(sX_curr-sX_last)>deltaWidth || fabs(eX_curr-eX_last)>deltaWidth || fabs(sY_curr-sY_last)>deltaHeight || fabs(eY_curr-eY_last)>deltaHeight )
            {
                matches_12[i1] = -1;
                delta_pose ++;
                continue;
            }
        }

        ///到这里才正确匹配
        mCurrentFrame.mvpMapLines[i2] = mLastFrame.mvpMapLines[i1];
        ++lmatches;
    }

    ///画出上一帧与当前帧之间的线特征匹配关系
    cv::Mat img11 = mLastFrame.ImageGray;
    cv::Mat img21 = mCurrentFrame.ImageGray;
    cvtColor(img11,img11,CV_GRAY2BGR);
    cvtColor(img21,img21,CV_GRAY2BGR);
    for(int i=0;i<mLastFrame.mvKeylinesUn.size();i++)
    {
        KeyLine l = mLastFrame.mvKeylinesUn[i];
        cv::line(img11,l.getStartPoint(),l.getEndPoint(),cv::Scalar(0,255,0),1);
    }
    for(int i=0;i<mCurrentFrame.mvKeylinesUn.size();i++)
    {
        KeyLine l = mCurrentFrame.mvKeylinesUn[i];
        cv::line(img21,l.getStartPoint(),l.getEndPoint(),cv::Scalar(0,165,255),1);
    }
    cv::Mat combine_img1;
    cvtColor(combine_img1,combine_img1,CV_GRAY2BGR);
    cv::hconcat(img11,img21,combine_img1);
    for(int i1=0;i1<matches_12.size();i1++)
    {
        cv::Mat temp = combine_img1.clone();
        KeyLine a = mLastFrame.mvKeylinesUn[i1];
        int i2 = matches_12[i1];
        if(i2==-1)continue;
        KeyLine b = mCurrentFrame.mvKeylinesUn[i2];
        //cv::line(combine_img,a.getStartPoint(),a.getEndPoint(),cv::Scalar(0,255,0),1);
        cv::Point2f a_mid = (a.getEndPoint() + a.getStartPoint())*0.5;
        cv::Point2f b_mid = (b.getEndPoint() + b.getStartPoint())*0.5;
        b_mid.x += img11.cols;
        //cv::line(combine_img,cv::Point2f(b.startPointX+(float)img1.cols,b.startPointY+(float)img1.rows),cv::Point2f(b.endPointX+(float)img1.cols,b.endPointY+(float)img1.rows),cv::Scalar(255,0,0),1);
        //cv::line(combine_img,cv::Point2f(b.startPointX+(float)img1.cols,b.startPointY),cv::Point2f(b.endPointX+(float)img1.cols,b.endPointY),cv::Scalar(255,0,0),1);
        //cv::line(combine_img,cv::Point2f(b.startPointX,b.startPointY),cv::Point2f(b.endPointX,b.endPointY),cv::Scalar(255,0,0),1);

        cv::line(combine_img1,a_mid,b_mid,cv::Scalar(0,0,255),1);
        //cv::circle(combine_img,a_mid,1,cv::Scalar(0,0,255),1);
        //cv::circle(combine_img,b_mid,1,cv::Scalar(0,0,128),1);
        //cv::imshow("ly",temp);
        //getchar();
    }
    cv::imshow("wh",combine_img1);
    getchar();
    cv::destroyWindow("wh");



    ///modify by wh
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());
    int initialMatches = nmatches + lmatches + planeMatches;
    if( initialMatches < 5 )
        return false;

    ///如何对消影点进行匹配  两帧之间  由相同的平行线计算出的消影点是匹配的
    ///已知 mLastFrame  mCurrentFrame
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    //pOnLMatch(mCurrentFrame);

    // Pose optimization using the reprojection error of 3D-2D

    ///传统的点线重投影误差
    Optimizer::PoseOptimization(&mCurrentFrame);


//    ///这里已经得到了当前帧的位姿
//    //cv::Mat Tcl =
//    cout<<">>>"<<endl;
//    cv::Mat img_last = mLastFrame.ImageGray.clone();
//    cv::Mat img_cur = mCurrentFrame.ImageGray.clone();
//    cv::Mat LastRwc = cv::Mat::eye(3,3,CV_32F);
//    mLastFrame.GetRotationInverse().copyTo(LastRwc.rowRange(0,3).colRange(0,3));
//    cv::Mat CurRcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
//    cv::Mat Rcl = CurRcw * LastRwc;
//    cv::Mat Kinv;
//    bool tag = cv::invert(mCurrentFrame.mK,Kinv);
//    ///对线段进行切分
//        for(int i=0;i<mLastFrame.NL;i++)
//        {
//            double len = cv::norm(
//                    mLastFrame.mvKeylinesUn[i].getStartPoint() - mLastFrame.mvKeylinesUn[i].getEndPoint());
//            double numSmp = (double) min((int) len, 20);
//            vector<cv::Point2f> temp, ans,ans2;
//            vector<uchar> status,status2;
//            vector<float> error,error2;
//            for (int j = 0; j <= numSmp; j++)
//            {
//                cv::Point2d pt = mLastFrame.mvKeylinesUn[i].getStartPoint() * (1 - j / numSmp) +
//                                 mLastFrame.mvKeylinesUn[i].getEndPoint() * (j / numSmp);
//                if (pt.x < 0 || pt.y < 0 || pt.x >= mLastFrame.ImageDepth.cols ||
//                    pt.y >= mLastFrame.ImageGray.rows)
//                    continue;
//                temp.push_back(pt);
//            }
//            vector<cv::Point2f> wh;
//            for(int j=0;j<temp.size();j++)
//            {
//                cv::Mat p1 = (cv::Mat_<float>(3,1) << temp[j].x , temp[j].y , 1.0);
//                cv::Mat mat;
//                mat = mCurrentFrame.mK * Rcl * Kinv * p1;
//                cv::Point2f p2(mat.at<float>(0,0),mat.at<float>(1,0));
//                wh.emplace_back(p2);
//            }
//            ///直接光流
//            cv::calcOpticalFlowPyrLK(img_last, img_cur, temp, ans, status, error);
//            ///位姿辅助的光流
//            cv::calcOpticalFlowPyrLK(img_cur, img_cur, wh, ans2, status2, error2);
//            vector<cv::Point2f> pts,pts2;
//            for(int j = 0; j < ans.size(); j++)if(status[j])pts.emplace_back(ans[j]);
//            for(int j = 0;j < ans2.size(); j++)if(status2[j])pts2.emplace_back(ans2[j]);
//            std::pair<double, double> fit_line = LSDmatcher::fitLineRANSAC(pts, 10, 10);
//            std::pair<double, double> fit_line2 = LSDmatcher::fitLineRANSAC(pts2, 10, 10);
//            cout << "normal lk flow best fit line : y = " << fit_line.first << " x + " << fit_line.second << endl;
//            cout << "Tcl lk flow best fit line : y = " << fit_line2.first << " x + " << fit_line2.second << endl;
//
//            ///draw line
//            cv::Point pt1(0,fit_line.second);
//            cv::Point pt1_2(0,fit_line2.second);
//            cv::Point pt2(mLastFrame.ImageGray.cols,fit_line.first*mLastFrame.ImageGray.cols+fit_line.second);
//            cv::Point pt2_2(mLastFrame.ImageGray.cols,fit_line2.first*mLastFrame.ImageGray.cols+fit_line2.second);
//            cv::Mat whiteImage(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
//            for(int j=0;j<pts.size();j++)
//            {
//                ///red  is  normal LK
//                if(status[j])cv::circle(whiteImage,ans[j],2,cv::Scalar(0,0,255));
//            }
//            ///green is Tcl LK
//            for(int j=0;j<pts2.size();j++)if(status2[j])cv::circle(whiteImage,ans2[j],2,cv::Scalar(0,255,0));
//            cv::line(whiteImage,pt1,pt2,cv::Scalar(255,0,0),1);
//            cv::line(whiteImage,pt1_2,pt2_2,cv::Scalar(0,255,125),1);
//            cv::imshow("ppp",whiteImage);
//            getchar();
//            cv::destroyWindow("ppp");
//        }


    // Discard Point outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                ///当前地图点经过优化后是外点的话就剔除
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            ///不是外点且有观测
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    // Discard Line outliers
    ///与点类似
    int nLinematchesMap = 0;
    for(int i =0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(mCurrentFrame.mvbLineOutlier[i])
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i]=false;
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                lmatches--;
            }
            else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                nLinematchesMap++;
        }

        mCurrentFrame.mvbLineOutlier[i] = false;
    }
    //cout<<"nLinematchesMap = "<<nLinematchesMap<<endl;
    //int del_par_track = 0;
    //discard plane outliers
    //cout<<"debug debug debug33333333333"<<endl;
    ////地图面没有观测一说 是outlier直接剔除
    int nmatchesPlaneMap = 0;
    for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
        if (mCurrentFrame.mvpMapPlanes[i]) {
            if (mCurrentFrame.mvbPlaneOutlier[i]) {
            } else
                nmatchesPlaneMap++;
        }

        if (mCurrentFrame.mvpParallelPlanes[i]) {
            if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                mCurrentFrame.mvbParPlaneOutlier[i] = false;
            }
        }

        if (mCurrentFrame.mvpVerticalPlanes[i]) {
            if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                mCurrentFrame.mvbVerPlaneOutlier[i] = false;
            }
        }
    }

    if((nmatchesMap + nLinematchesMap + nmatchesPlaneMap) < 3)
    {
        mCurrentFrame.SetPose(mLastFrame.mTcw);
        return false;
    }
    return true;
}


void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }

    // TODO 1: When the SLAM procedure will be ready in the future, add lines for the relocalisation mode .
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);
    LSDmatcher lmatcher;
    ///modify by wh
    PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    ///这个函数没有更新线 但planar slam中更新了
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
   
    // Match Lines: Two options
    int lmatches = 0;
    
    float radius_th = 3.0;
    ///这个变量在matchNNR中已经全部初始化为-1
    vector<int> matches_12;

//    ///画出上一帧与当前帧之间的线特征匹配关系
//    cv::Mat img1 = mLastFrame.ImageGray;
//    cv::Mat img2 = mCurrentFrame.ImageGray;
//    cvtColor(img1,img1,CV_GRAY2BGR);
//    cvtColor(img2,img2,CV_GRAY2BGR);
//    for(int i=0;i<mLastFrame.mvKeylinesUn.size();i++)
//    {
//        KeyLine l = mLastFrame.mvKeylinesUn[i];
//        cv::line(img1,l.getStartPoint(),l.getEndPoint(),cv::Scalar(255,0,0),1);
//    }
//    for(int i=0;i<mCurrentFrame.mvKeylinesUn.size();i++)
//    {
//        KeyLine l = mCurrentFrame.mvKeylinesUn[i];
//        cv::line(img2,l.getStartPoint(),l.getEndPoint(),cv::Scalar(255,0,0),1);
//    }
//    cv::Mat combine_img;
//    cvtColor(combine_img,combine_img,CV_GRAY2BGR);
//    cv::hconcat(img1,img2,combine_img);
//    cv::imshow("ttt",combine_img);


    // 1/ Search by projection combining geometrical and appearance constraints
    // lmatches  = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, radius_th);

    // 2/ Option from Ruben Gomez Ojeda -- line segments f2f tracking
    ///先通过lbd描述符初步匹配当前帧和上一帧的线特征匹配关系 最后把上一帧的地图线按照匹配关系赋值给当前帧
    float des_th = 0.95;
    lmatches  = lmatcher.SearchByGeomNApearance(mCurrentFrame, mLastFrame, des_th,matches_12);

    int flow_num = lmatcher.optical_flow_line(mCurrentFrame,mLastFrame,matches_12,coarseRcl);


    //cv::destroyWindow("ttt");


    ///画出上一帧与当前帧之间的线特征匹配关系
//    cv::Mat img1 = mLastFrame.ImageGray;
//    cv::Mat img2 = mCurrentFrame.ImageGray;
//    cvtColor(img1,img1,CV_GRAY2BGR);
//    cvtColor(img2,img2,CV_GRAY2BGR);
//    for(int i=0;i<mLastFrame.mvKeylinesUn.size();i++)
//    {
//        KeyLine l = mLastFrame.mvKeylinesUn[i];
//        cv::line(img1,l.getStartPoint(),l.getEndPoint(),cv::Scalar(255,0,0),1);
//    }
//    for(int i=0;i<mCurrentFrame.mvKeylinesUn.size();i++)
//    {
//        KeyLine l = mCurrentFrame.mvKeylinesUn[i];
//        cv::line(img2,l.getStartPoint(),l.getEndPoint(),cv::Scalar(255,0,0),1);
//    }
//    cv::Mat combine_img;
//    cvtColor(combine_img,combine_img,CV_GRAY2BGR);
//    cv::hconcat(img1,img2,combine_img);
//    for(int i1=0;i1<matches_12.size();i1++)
//    {
//        KeyLine a = mLastFrame.mvKeylinesUn[i1];
//        int i2 = matches_12[i1];
//        KeyLine b = mCurrentFrame.mvKeylinesUn[i2];
//        if(i2==-1)
//        {
//            ///说明该线段在当前帧中无匹配
//            cv::Point2f a_mid = (a.getEndPoint() + a.getStartPoint())*0.5;
//            cv::Point2f b_mid = cv::Point2f(img1.cols,0.0);
//            ///红色的线
//            cv::line(combine_img,a_mid,b_mid,cv::Scalar(0,0,255),1);
//        }
//        else
//        {
//            cv::Point2f a_mid = (a.getEndPoint() + a.getStartPoint())*0.5;
//            cv::Point2f b_mid = (b.getEndPoint() + b.getStartPoint())*0.5;
//            b_mid.x += img1.cols;
//            ///1.两条匹配的线段应该是大致平行的
//            Vector2d l1;
//            cv::Point2f line1 = mLastFrame.mvKeylinesUn[i1].getEndPoint() - mLastFrame.mvKeylinesUn[i1].getStartPoint();
//            l1 << (double)line1.x , (double)line1.y;
//            l1[0] /= sqrt(l1[0]*l1[0] + l1[1]*l1[1]);
//            l1[1] /= sqrt(l1[0]*l1[0] + l1[1]*l1[1]);
//            Vector2d nor1(l1[1],-1.0*l1[0]);
//            Vector2d l2;
//            cv::Point2f line2 = mCurrentFrame.mvKeylinesUn[i2].getEndPoint() - mCurrentFrame.mvKeylinesUn[i2].getStartPoint();
//            l2 << (double)line2.x , (double)line2.y;
//            l2[0] /= sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
//            l2[1] /= sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
//
//
//            ///如何没有匹配 点乘是nan
//            ///点乘按道理来说应该是接近0的一个数字 误匹配会是一个比较大的值  但也有一些特殊的误匹配  值接近0 应该使用其它条件筛选
//            double error = nor1.dot(l2);
//            double dis = 0.0;
//            ///mid_p 是上一帧中一条线段的中点   dis为mid_p到当前帧中匹配的线段的距离
//            Vector2d mid_p;
//            mid_p << (double)(mLastFrame.mvKeylinesUn[i1].startPointX+mLastFrame.mvKeylinesUn[i1].endPointX)*0.5,
//                    (double)(mLastFrame.mvKeylinesUn[i1].startPointY+mLastFrame.mvKeylinesUn[i1].endPointY)*0.5;
//            dis = mid_p[0]*mCurrentFrame.mvKeyLineFunctions[i2][0] + mid_p[1]*mCurrentFrame.mvKeyLineFunctions[i2][1] + mCurrentFrame.mvKeyLineFunctions[i2][2];
//
//            if(error>0.1 || dis>10.0)
//            {
//                cv::line(combine_img,a_mid,b_mid,cv::Scalar(0,0,255),1);
//            }
//            else
//            {
//                cv::line(combine_img,a_mid,b_mid,cv::Scalar(0,255,0),1);
//            }
//        }
//    }
//    cv::imshow("wh",combine_img);
//    getchar();
//    cv::destroyWindow("wh");

    ///
//    vector<MapLine *>vpMapLineMatches;
//    lmatches = lmatcher.SearchByDescriptor(mpReferenceKF,mCurrentFrame,vpMapLineMatches);
//    mCurrentFrame.mvpMapLines = vpMapLineMatches;

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    // fill(mCurrentFrame.mvpMapLines.begin(),mCurrentFrame.mvpMapLines.end(),static_cast<MapLine*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if((nmatches + lmatches) <20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
        lmatches  = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, 2*radius_th);
    }

    ///modify by wh
    int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());


    int initialMatches = nmatches + lmatches + planeMatches;

    if(initialMatches < 10)
        return false;

    ///add point on line structure
    //pOnLMatch(mCurrentFrame);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard Pt outliers
    int n_matches_map = 0;
    int nLinematchesMap = 0;
    int nmatchesPlaneMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                n_matches_map++;
        }
    }    

    // Discard Line outliers

    for(int i =0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(mCurrentFrame.mvbLineOutlier[i])
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i]=false;
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                lmatches--;
            }
            else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                nLinematchesMap++;
        }
        mCurrentFrame.mvbLineOutlier[i] = false;
    }

    ///modify by wh

    for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
        if (mCurrentFrame.mvpMapPlanes[i]) {
            if (mCurrentFrame.mvbPlaneOutlier[i]) {
            } else
                nmatchesPlaneMap++;
        }

        if (mCurrentFrame.mvpParallelPlanes[i]) {
            if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                mCurrentFrame.mvbParPlaneOutlier[i] = false;
            }
        }

        if (mCurrentFrame.mvpVerticalPlanes[i]) {
            if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                mCurrentFrame.mvbVerPlaneOutlier[i] = false;
            }
        }
    }

    int n_matches_map_pts_lines =  n_matches_map + nLinematchesMap + nmatchesPlaneMap;

    int nmatches_pts_lines = nmatches + lmatches;

    if(mbOnlyTracking)
    {
        mbVO = n_matches_map_pts_lines<20;
        return nmatches_pts_lines>20;
    }

    return n_matches_map_pts_lines>=5;
}

///modify by wh
void Tracking::get_vp(int k)
{
    for(int i=0;i<mCurrentFrame.mvParLinesIdx[k]->size();i++)
    {
        //int idx = mCurrentFrame.mvParLinesIdx[k][i];
    }
}
///todo   做完了 感觉没问题
///1.将地图线投影到当前帧 经过投影得到的线特征记为临时线特征
///2.利用临时线特征计算消影点  用两条平行的临时线特征还是一条当前帧的线特征一条临时线特征  可以都试试
///3.与当前帧中的消影点进行关联
void Tracking::match_vp(ORB_SLAM2::Frame &CurrentFrame,vector<int> match_12)
{
    ///存储二维线函数
    vector<Vector3d> mvKeyLineFunctions_temp;
    ///初始化
    mvKeyLineFunctions_temp.resize(CurrentFrame.mvKeylinesUn.size(),Vector3d (-1.0, -1.0, -1.0));
    int size = CurrentFrame.mvpMapLines.size();
    ///遍历当前帧的每条有效的地图线  计算对应的像素平面的归一化线函数
    for(int i=0;i<size;i++)
    {
        MapLine *ml = CurrentFrame.mvpMapLines[i];
        if(!ml)continue;
        ///取出起点和终点的3d坐标 世界坐标系
        Vector3d ml_sw = ml->mWorldPos.head(3);
        Vector3d ml_ew = ml->mWorldPos.tail(3);
        Mat Tcw = CurrentFrame.mTcw;
        Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
        Mat tcw = Tcw.rowRange(0,3).col(3);
        Mat ml_sw_mat = Converter::toCvMat(ml_sw);
        Mat ml_ew_mat = Converter::toCvMat(ml_ew);
        ///应该把T分解为R和t
        ///将地图线的两个端点变换到相机坐标系
        Mat ml_sc_mat = Rcw * ml_sw_mat + tcw;
        Mat ml_ec_mat = Rcw * ml_ew_mat + tcw;
        Vector3d ml_sc = Converter::toVector3d(ml_sc_mat);
        Vector3d ml_ec = Converter::toVector3d(ml_ec_mat);
        ///已经计算出临时线特征在相机坐标系下两个端点的3d坐标
        ///下面进行 cam_project   相机坐标系->归一化坐标系
        Vector2d sc(ml_sc[0]/ml_sc[2],ml_sc[1]/ml_sc[2]);
        Vector2d ec(ml_ec[0]/ml_ec[2],ml_ec[1]/ml_ec[2]);
        float fx = CurrentFrame.fx;
        float fy = CurrentFrame.fy;
        float cx = CurrentFrame.cx;
        float cy = CurrentFrame.cy;
        ///归一化坐标系->像素坐标系
        Vector2d sc_img(sc[0]*fx+cx,sc[1]*fy+cy);
        Vector2d ec_img(ec[0]*fx+cx,ec[1]*fy+cy);
        ///利用像素坐标的端点计算归一化线函数
        Vector3d sc_img_3d;
        sc_img_3d << sc_img[0],sc_img[1],1.0;
        Vector3d ec_img_3d;
        ec_img_3d << ec_img[0],ec_img[1],1.0;
        Vector3d lineV;
        lineV << sc_img_3d.cross(ec_img_3d);
        lineV = lineV / sqrt(lineV(0) * lineV(0) + lineV(1) * lineV(1));
        mvKeyLineFunctions_temp[i] = lineV;
    }
    int size2 = CurrentFrame.vanish_point.size();
    ///遍历当前帧的每个消影点
    int cnt1=0,cnt2=0;
    int xx=0,yy=0;
    for(int i=0;i<size2;i++)
    {
        ///取出构成当前消影点的 当前帧中一对平行线的id
        pair<int,int> t = CurrentFrame.vp_map[i+1];
        int id1 = t.first , id2 = t.second;
        ///取出id对应的线函数
        auto line1 = mvKeyLineFunctions_temp[id1];
        auto line2 = mvKeyLineFunctions_temp[id2];
        //cout<<"line1 = "<<line1<<"   line2 = "<<line2<<endl;
        ///求交点
        cv::Point2d crossPoint;
        double d = line1[0]*line2[1] - line1[1]*line2[0];
        if(d!=0.0)
        {
            //xx++;
            crossPoint.x = (line1[1]*line2[2] - line1[2]*line2[1])/d;
            crossPoint.y = (line2[0]*line1[2] - line1[0]*line2[2])/d;
            cout<<"两点之间的欧式距离 = "<<sqrt(pow(CurrentFrame.vanish_point[i].x-crossPoint.x,2) + pow(CurrentFrame.vanish_point[i].y-crossPoint.y,2))<<endl;
            Mat p1(3,1,CV_32FC1,cv::Scalar(crossPoint.x,crossPoint.y,1.0));
            //Mat p1(3,1,CV_32FC1,cv::Scalar(mCurrentFrame.mvKeysUn[i].pt.x,mCurrentFrame.mvKeysUn[i].pt.y,1.0));
            Mat p2(3,1,CV_32FC1,cv::Scalar(CurrentFrame.vanish_point[i].x,CurrentFrame.vanish_point[i].y,1.0));
            Mat r1 = CurrentFrame.mK.inv() * p1;
            Mat r2 = CurrentFrame.mK.inv() * p2;
            Vector3d s1 = Converter::toVector3d(r1);
            Vector3d s2 = Converter::toVector3d(r2);
//            cout<<"s1 = "<<s1<<endl;
//            cout<<"s2 = "<<s2<<endl;
            s1 = s1 / sqrt(s1[0]*s1[0] + s1[1]*s1[1] + s1[2]*s1[2]);
            s2 = s2 / sqrt(s2[0]*s2[0] + s2[1]*s2[1] + s2[2]*s2[2]);
//            cout<<"after s1 = "<<s1<<endl;
//            cout<<"after s2 = "<<s2<<endl;
            //cout<<"distance = "<<sqrt(pow(s2[0]-s1[0],2) + pow(s2[1]-s1[1],2) + pow(s2[2]-s1[2],2))<<endl;
            if((s1[0]*s1[0] + s1[1]*s1[1] + s1[2]*s1[2] == 1) && (s2[0]*s2[0] + s2[1]*s2[1] + s2[2]*s2[2] == 1))
            {
                cnt1++;
                cout<<"distance = "<<sqrt(pow(s2[0]-s1[0],2) + pow(s2[1]-s1[1],2) + pow(s2[2]-s1[2],2))<<endl;
            }
            else cnt2++;
            if(crossPoint.x>=CurrentFrame.mnMinX && crossPoint.x<=CurrentFrame.mnMaxX && crossPoint.y>=CurrentFrame.mnMinY && crossPoint.y<=CurrentFrame.mnMaxY)
            {
                if(CurrentFrame.vanish_point[i].x>=CurrentFrame.mnMinX && CurrentFrame.vanish_point[i].x<=CurrentFrame.mnMaxX && CurrentFrame.vanish_point[i].y>=CurrentFrame.mnMinY && CurrentFrame.vanish_point[i].y<=CurrentFrame.mnMaxY)yy++;
                else xx++;
            }
            else xx++;
        }
        cout<<endl;
    }
    cout<<"在单位球上有 "<<cnt1<<endl;
    cout<<"其它 "<<cnt2<<endl;
    cout<<"在图像内的有 "<<yy<<endl;
    cout<<"其它 "<<xx<<endl;
    //cout<<"xx = "<<xx<<"   yy = "<<yy<<endl;
}
void Tracking::draw_vp(ORB_SLAM2::Frame &CurrentFrame)
{
    cv::Mat ori_img = CurrentFrame.ImageGray.clone();
    cvtColor(ori_img,ori_img,CV_GRAY2BGR);
    cv::Mat canvas(5000,5000,ori_img.type(),cv::Scalar(255,255,255));
    int startX = canvas.cols/2 - ori_img.cols/2;
    int startY = canvas.rows/2 - ori_img.rows/2;
    cv::Rect roi = cv::Rect(startX,startY,ori_img.cols,ori_img.rows);
    ori_img.copyTo(canvas(roi));
    int size = CurrentFrame.vanish_point.size();
    for(int i=0;i<size;i++)
    {
        cv:Point2f vp = CurrentFrame.vanish_point[i];
        vp.x += startX;
        vp.y += startY;
        cv::circle(canvas , vp,5,cv::Scalar(0,0,255),-1);
    }
    cv::Mat ans;
    resize(canvas,ans,Size(),0.3,0.3);
    cv::imshow("ww",ans);
    getchar();
    cv::destroyWindow("ww");
}
void Tracking::match_vpBydis(ORB_SLAM2::Frame &CurrentFrame, const ORB_SLAM2::Frame &LastFrame,
                             std::vector<int> &match_12, double distance){
    int cur_size = CurrentFrame.vanish_point.size();
    int la_size = LastFrame.vanish_point.size();
    bool matched[LastFrame.vanish_point.size()];
    memset(matched,0, sizeof(matched));
    cout<<"LastFrame.index_vp.size = "<<LastFrame.index_vp.size()<<endl;
    for(int i=0;i<cur_size;i++)
    {
        cv::Point2f cur_vp = CurrentFrame.vanish_point[i];
        double best_dis=999999.0;
        int best_id = 0;
        for(auto j = LastFrame.index_vp.begin();j!=LastFrame.index_vp.end();j++)
        {
            cv::Point2f la_vp = j->second;
            double dis = sqrt(pow(cur_vp.x-la_vp.x,2) + pow(cur_vp.y-la_vp.y,2));
            if( dis < best_dis )
            {
                best_dis = dis;
                best_id = j->first;
                //matched[j->first] = true;
            }
        }
        cout<<"第 "<<i+1<<" 个消影点匹配的最优距离是 "<<best_dis<<endl;
        if(best_dis < distance)
        {
            match_12[i] = best_id;
            matched[best_id] = true;
        }
    }

}
void Tracking::draw_2vp(ORB_SLAM2::Frame &CurrentFrame, const ORB_SLAM2::Frame &LastFrame,vector<int> matches)
{
    cv::Mat ori_img = CurrentFrame.ImageGray.clone();
    cvtColor(ori_img,ori_img,CV_GRAY2BGR);
    cv::Mat canvas(5000,5000,ori_img.type(),cv::Scalar(255,255,255));
    int startX = canvas.cols/2 - ori_img.cols/2;
    int startY = canvas.rows/2 - ori_img.rows/2;
    cv::Rect roi = cv::Rect(startX,startY,ori_img.cols,ori_img.rows);
    ori_img.copyTo(canvas(roi));
    int size = CurrentFrame.vanish_point.size();
    int size2 = LastFrame.vanish_point.size();
    for(int i=0;i<matches.size();i++)
    {
        if(matches[i]!=0)
        {
            cv::Point2f vp1 = CurrentFrame.vanish_point[i];
            cv::Point2f vp2 = LastFrame.vanish_point[matches[i]];
            double dis = sqrt(pow(vp1.x-vp2.x,2) + pow(vp1.y-vp2.y,2));
            cout<<"two match distance = "<<dis<<endl;
            vp1.x+=startX,vp1.y+=startY;
            vp2.x+=startX,vp2.y+=startY;
            cv::circle(canvas , vp1,5,cv::Scalar(0,0,255),-1);
            cv::circle(canvas , vp2,5,cv::Scalar(0,255,0),-1);
        }
    }
//    for(int i=0;i<size;i++)
//    {
//        cv::Point2f vp = CurrentFrame.vanish_point[i];
//        vp.x += startX;
//        vp.y += startY;
//        cv::circle(canvas , vp,5,cv::Scalar(0,0,255),-1);
//    }
//    for(int i=0;i<size;i++)
//    {
//        cv::Point2f vp = LastFrame.vanish_point[i];
//        vp.x += startX;
//        vp.y += startY;
//        cv::circle(canvas , vp,5,cv::Scalar(0,255,0),-1);
//    }
    cv::Mat ans;
    resize(canvas,ans,Size(),0.3,0.3);
    cv::imshow("qq",ans);
    getchar();
    cv::destroyWindow("qq");
}
void Tracking::draw(vector<pair<cv::Point2f, cv::Point2f>> vp)
{
    cv::Mat ori_img = mCurrentFrame.ImageGray.clone();
    cvtColor(ori_img,ori_img,CV_GRAY2BGR);
    cv::Mat canvas(5000,5000,ori_img.type(),cv::Scalar(255,255,255));
    int startX = canvas.cols/2 - ori_img.cols/2;
    int startY = canvas.rows/2 - ori_img.rows/2;
    cv::Rect roi = cv::Rect(startX,startY,ori_img.cols,ori_img.rows);
    ori_img.copyTo(canvas(roi));
    for(int i=0;i<vp.size();i++)
    {
        cv::Point2f vp1 = vp[i].first;
        cv::Point2f vp2 = vp[i].second;
        vp1.x+=startX,vp1.y+=startY;
        vp2.x+=startX,vp2.y+=startY;
        cv::circle(canvas , vp1,5,cv::Scalar(0,0,255),-1);
        cv::circle(canvas , vp2,5,cv::Scalar(0,255,0),-1);
    }
    cv::Mat ans;
    resize(canvas,ans,Size(),0.3,0.3);
    cv::imshow("ww",ans);
    getchar();
    cv::destroyWindow("ww");
}
///vp match  这是从两帧之间匹配消影点  2d to 2d
///1.先通过线特征匹配求出两帧之间的线特征匹配关系
///2.根据上一帧的消影点得到上一帧的一对平行线id
///3.利用上一帧的平行线id与matches_12得到当前帧中对应的平行线id
///4.利用3求得的当前帧的平行线id求出当前帧中对应的消影点
void Tracking::match2frame_vp(ORB_SLAM2::Frame &CurrentFrame, ORB_SLAM2::Frame &LastFrame)
{
//    LSDmatcher lmatcher;
//    float des_th = 0.95;
//    vector<int> match_12;
//    int ans = lmatcher.SearchByGeomNApearance(CurrentFrame,LastFrame,des_th,match_12);
//    vector< pair< cv::Point2f ,cv::Point2f > >matched_vp;
//    int cnt = 0;
//    int size1 = LastFrame.vanish_point.size();
//    bool flag[CurrentFrame.vanish_point.size()];
//    memset(flag,0, sizeof(flag));
//    for(int i=0;i<size1;i++)
//    {
//        int vp_id = i+1;
//        pair<int,int> temp = LastFrame.vp_map[vp_id];
//        if(match_12[temp.first]==-1 || match_12[temp.second]==-1)continue;
//        pair<int,int> temp2 = make_pair(match_12[temp.first],match_12[temp.second]);
//        int size2 = mCurrentFrame.vanish_point.size();
//        for(int j=0;j<size2;j++)
//        {
//            if(flag[j])continue;
//            int vp_id2 = j+1;
//            pair<int,int> temp3 = CurrentFrame.vp_map[vp_id2];
//            pair<int,int> temp4 = make_pair(temp3.second,temp3.first);
//            if(temp2==temp3 || temp2==temp4)
//            {
//                flag[j]=true;
//                matched_vp.push_back(make_pair(LastFrame.vanish_point[i],CurrentFrame.vanish_point[j]));
//                cnt++;
//                break;
//            }
//        }
//    }
//    cout<<"匹配到的消影点有 "<<cnt<<endl;
}

int Tracking::pOnLMatch(ORB_SLAM2::Frame &CurrentFrame)
{
//    for(int i=0;i<CurrentFrame.NL;i++)
//    {
//        Vector3d  l = CurrentFrame.mvKeyLineFunctions[i];
//        KeyLine line = CurrentFrame.mvKeylinesUn[i];
////        cv::Mat img = CurrentFrame.ImageGray.clone();
////        cvtColor(img,img,CV_GRAY2BGR);
////        cv::line(img,line.getStartPoint(),line.getEndPoint(),cv::Scalar(255,0,0),1);
//        vector<int> temp;
//        for(int j=0;j<CurrentFrame.N;j++)
//        {
//            cv::Point2f p = CurrentFrame.mvKeysUn[j].pt;
//            double dis = (p.x*l[0] + p.y*l[1] + l[2]) / sqrt(l[0]*l[0] + l[1]*l[1]);
//            if(std::abs(dis)<1.0)
//            {
//                if(p.x<fmin(line.startPointX,line.endPointX))continue;
//                if(p.x>fmax(line.startPointX,line.endPointX))continue;
//                if(p.y<fmin(line.startPointY,line.endPointY))continue;
//                if(p.y>fmax(line.startPointY,line.endPointY))continue;
//                //cout<<"第"<<i<<"条线与第"<<j<<"个点的距离 = "<<abs(dis)<<endl;
//                temp.push_back(j);
//                CurrentFrame.pOnL++;
////                cv::circle(img,p,1,cv::Scalar(0,0,255),2);
//            }
//        }
//        //cout<<"这条线上有 "<<temp.size()<<" 个点！"<<endl;
////        cv::imshow("wh",img);
//        //getchar();
////        cv::destroyWindow("wh");
//        CurrentFrame.fa.push_back(make_pair(i,temp));
//    }
}
bool Tracking::TrackLocalMapWithLines()
{
    PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);
    UpdateLocalMap();

    thread threadPoints(&Tracking::SearchLocalPoints, this);
    thread threadLines(&Tracking::SearchLocalLines, this);
    thread threadPlanes(&Tracking::SearchLocalPlanes, this);
    threadPoints.join();
    threadLines.join();
    threadPlanes.join();
    pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());
       
    std::chrono::steady_clock::time_point t1_line_opt = std::chrono::steady_clock::now();

    mpManh->computeStructConstInMap(mCurrentFrame, mvpLocalMapLines_InFrustum);
      std::chrono::steady_clock::time_point t2_line_opt = std::chrono::steady_clock::now();
    chrono::duration<double> time_used_line_opt = chrono::duration_cast<chrono::duration<double>>(t2_line_opt - t1_line_opt);

    //pOnLMatch(mCurrentFrame);
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;
    mnLineMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Update MapLines Statistics
    for(int i=0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(!mCurrentFrame.mvbLineOutlier[i])
            {
                mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                        mnLineMatchesInliers++;
                }
                else
                    mnLineMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
        }
    }

    for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
        if (mCurrentFrame.mvpMapPlanes[i]) {
            if (mCurrentFrame.mvbPlaneOutlier[i]) {
            } else {
                mCurrentFrame.mvpMapPlanes[i]->IncreaseFound();
                mnMatchesInliers++;
            }
        }

        if (mCurrentFrame.mvpParallelPlanes[i]) {
            if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                mCurrentFrame.mvbParPlaneOutlier[i] = false;
            }
        }

        if (mCurrentFrame.mvpVerticalPlanes[i]) {
            if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                mCurrentFrame.mvbVerPlaneOutlier[i] = false;
            }
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    match_true = mnMatchesInliers - 3;
    //cout<<"mnLineMatchesInliers = "<<mnLineMatchesInliers<<endl;
    ///50
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers + mnLineMatchesInliers <1)
        return false;


    ///15
    if(mnMatchesInliers + mnLineMatchesInliers <1)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints and MapLines in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    int nRefMatchesLines = mpReferenceKF->TrackedMapLines(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points and lines are being tracked and how many could be potentially created.
    // This stage differs from ORB-SLAM2, we use the ratio, because it better adapts to low-textured environments. 
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    float ratio_pts = nNonTrackedClose / float(nTrackedClose + nNonTrackedClose);

    int nNonTrackedCloseLine = 0;
    int nTrackedCloseLine= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.NL; i++)
        {
            if(mCurrentFrame.mvLines3D[i].first[2]>0 && mCurrentFrame.mvLines3D[i].first[2]<mThDepth)
            {
                if(mCurrentFrame.mvpMapLines[i] && !mCurrentFrame.mvbLineOutlier[i])
                    nTrackedCloseLine++;
                else
                    nNonTrackedCloseLine++;
            }
        }
    }

    float ratio_lines = nNonTrackedCloseLine / float(nTrackedCloseLine + nNonTrackedCloseLine);
    bool bNeedToInsertClosePtsLine = ratio_pts >0.6 || ratio_lines > 0.6;
    
    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=(mnLastKeyFrameId + 30)) && bLocalMappingIdle);
   
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.4 || bNeedToInsertClosePtsLine || mnLineMatchesInliers < nRefMatchesLines*0.4) ;

    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClosePtsLine) && mnMatchesInliers>15) || ((mnLineMatchesInliers<nRefMatchesLines*thRefRatio|| bNeedToInsertClosePtsLine) && mnLineMatchesInliers>15));

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            
                nPoints++;

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }
     if(mSensor!=System::MONOCULAR)
    {
        vector<pair<float, int>> vDepthIdxLines;
        vDepthIdxLines.reserve(mCurrentFrame.NL);
        for (int i = 0; i < mCurrentFrame.NL; i++)
        {
            ///过滤掉非结构线
            //if(mCurrentFrame.isStructLine[i]==false)continue;
            if(mCurrentFrame.mvLines3D[i].first.z() == 0 || mCurrentFrame.mvLines3D[i].second.z()  == 0)
                continue;

            double sz = mCurrentFrame.mvLines3D[i].first.z();
            double ez = mCurrentFrame.mvLines3D[i].second.z();
            float z = sz > ez ? sz : ez;
            vDepthIdxLines.push_back(make_pair(z, i));
        }
       
        if (!vDepthIdxLines.empty())
        {
            sort(vDepthIdxLines.begin(), vDepthIdxLines.end());

            int nLines = 0;
            for (size_t j = 0; j < vDepthIdxLines.size(); j++)
            {
                int i = vDepthIdxLines[j].second;

                bool bCreateNew = false;

                MapLine *pML = mCurrentFrame.mvpMapLines[i];
                if (!pML)
                    bCreateNew = true;
                else if (pML->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }

                if (bCreateNew)
                {
                    // Select a valid Line
                    if (mCurrentFrame.mvLines3D[i].first.z() != 0.0 )
                    {
                        Eigen::Vector3d st_3D_w = mCurrentFrame.PtToWorldCoord(mCurrentFrame.mvLines3D[i].first);
                        Eigen::Vector3d e_3D_w = mCurrentFrame.PtToWorldCoord(mCurrentFrame.mvLines3D[i].second);

                        Vector6d w_l_endpts;
                        w_l_endpts << st_3D_w.x(), st_3D_w.y(), st_3D_w.z(),
                            e_3D_w.x(), e_3D_w.y(), e_3D_w.z();

                       MapLine *pNewML = new MapLine(w_l_endpts, mCurrentFrame.vManhAxisIdx[i], pKF, mpMap);
                       pNewML->UpdateManhAxis();

                        pNewML->AddObservation(pKF, i);

                        std::vector<int> v_idx_perp;
                        std::vector<int> v_idx_par;

                        mpManh->computeStructConstrains(mCurrentFrame, pNewML, v_idx_par, v_idx_perp);

                        if (mCurrentFrame.mvParallelLines[i]->size() > 0)
                        {
                            pNewML->AddParObservation(pKF, v_idx_par);
                        }

                        if (mCurrentFrame.mvPerpLines[i]->size() > 0)
                        {
                            pNewML->AddPerpObservation(pKF, v_idx_perp);
                        }
                        pKF->AddMapLine(pNewML, i);
                        pNewML->ComputeDistinctiveDescriptors();
                        // TODO 0: check if the two following lines are required 
                        // pNewML-> UpdateAverageDir();
                        //  pNewML->UpdateNormalAndDepth();
                        mpMap->AddMapLine(pNewML);

                        mCurrentFrame.mvpMapLines[i] = pNewML;
                        nLines++;
                    }
                    else
                    {
                        nLines++;
                    }

                    if (vDepthIdxLines[j].first > mThDepth && nLines > 100)
                        break;
                }
             }
        }

        ///plane  modify by wh
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
            if (mCurrentFrame.mvpParallelPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
                mCurrentFrame.mvpParallelPlanes[i]->AddParObservation(pKF, i);
            }
            if (mCurrentFrame.mvpVerticalPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
                mCurrentFrame.mvpVerticalPlanes[i]->AddVerObservation(pKF, i);
            }

            if (mCurrentFrame.mvpMapPlanes[i]) {
                mCurrentFrame.mvpMapPlanes[i]->AddObservation(pKF, i);
                continue;
            }

            if (mCurrentFrame.mvbPlaneOutlier[i]) {
                mCurrentFrame.mvbPlaneOutlier[i] = false;
                continue;
            }

            cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
            MapPlane *pNewMP = new MapPlane(p3D, pKF, mpMap);
            pNewMP->AddObservation(pKF,i);
            pKF->AddMapPlane(pNewMP, i);
            pNewMP->UpdateCoefficientsAndPoints();
            mpMap->AddMapPlane(pNewMP);
        }


    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::SearchLocalLines()
{
    bool eval_orient = true;
    if(mSensor==System::MONOCULAR)
    {
        eval_orient = false;
    }

    // vector<MapLine*> mvpLocalMapLines_InFrustum;
    //cout<<"SearchLocalLines "<<mCurrentFrame.mvpMapLines.size()<<endl;
    int normal = 0;
    for(vector<MapLine*>::iterator vit=mCurrentFrame.mvpMapLines.begin(), vend=mCurrentFrame.mvpMapLines.end(); vit!=vend; vit++)
    {
        MapLine* pML = *vit;
        if(pML)
        {
            //cout<<"pml id = "<<pML->mnId<<endl;
            if(pML->isBad())
            {
                *vit = static_cast<MapLine*>(NULL);
            } 
            else{
                //cout<<"normal line = "<<pML->mnId<<endl;
                normal++;
                pML->IncreaseVisible();
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                pML->mbTrackInView = false;
            }
        }
    }
    //cout<<"normal = "<<normal<<endl;

    int nToMatch = 0;
    mvpLocalMapLines_InFrustum.clear();
    int wh = 0,ly=0;

    for (vector<MapLine *>::iterator vit = mvpLocalMapLines.begin(), vend = mvpLocalMapLines.end(); vit != vend; vit++)
    {
        MapLine *pML = *vit;
        if (pML->mnLastFrameSeen == mCurrentFrame.mnId)
        {
            //cout<<"current frame pml id = "<<pML->mnId<<endl;
            wh++;
            continue;
        }
        if (pML->isBad())
        {
            ly++;
            continue;
        }

         // Project (this fills MapLine variables for matching)
//         wh++;

         ///判断其是否在成像范围内
        if (mCurrentFrame.isInFrustum(pML, 0.5))
        {
            pML->IncreaseVisible();
            nToMatch++;
            mvpLocalMapLines_InFrustum.push_back(pML);
        }
    }
    //cout<<"wh = "<<wh<<endl;
    //cout<<"ly = "<<ly<<endl;

    //cout<<"nToMatch = "<<nToMatch<<endl;
    if(nToMatch>0)
    {

        LSDmatcher matcher;
        int th = 1;

        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        //cout<<"here!"<<endl;
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, eval_orient, th);

        if(nmatches)
        {
            //cout<<"nmatches = "<<nmatches<<endl;
            for(int i = 0; i<mCurrentFrame.mvpMapLines.size(); i++)
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];
                if(pML)
                {
                    Eigen::Vector3d tWorldVector = pML->GetWorldVector();
                    cv::Mat tWorldVector_ = (cv::Mat_<float>(3, 1) << tWorldVector(0), tWorldVector(1), tWorldVector(2));
                    KeyLine tkl = mCurrentFrame.mvKeylinesUn[i];
                    cv::Mat tklS = (cv::Mat_<float>(3, 1) << tkl.startPointX, tkl.startPointY, 1.0);
                    cv::Mat tklE = (cv::Mat_<float>(3, 1) << tkl.endPointX, tkl.endPointY, 1.0);
                    cv::Mat K = mCurrentFrame.mK;
                    cv::Mat tklS_ = K.inv() * tklS; cv::Mat tklE_ = K.inv() * tklE;

                    cv::Mat NormalVector_ = tklS_.cross(tklE_);
                    double norm_ = cv::norm(NormalVector_);
                    NormalVector_ /= norm_;

                    cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
                    cv::Mat tCameraVector_ = Rcw * tWorldVector_;
                    double CosSita = abs(NormalVector_.dot(tCameraVector_));

                    if(CosSita>0.09)
                    {
                        mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                    }
                }
            }
        }

    }
    //cout<<"????"<<endl;

}

void Tracking::UpdateLocalMap()
{
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    
    // TODO: 0 Join Lines, perform it in the back-end
    // std::vector<std::vector<int>> sim_lines_idx;
    // FindSimilarLines(sim_lines_idx);
    // JoinLines(sim_lines_idx);

    mpMap->SetReferenceMapLines(mvpLocalMapLines);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
    UpdateLocalLines();
}

void Tracking::FindSimilarLines(std::vector<std::vector<int>> &sim_lines_idx)
{
    double angle = 10;
    double th_rad = angle / 180.0 * M_PI;
    double th_angle = std::cos(th_rad);

    std::vector<bool> found_idx;
    found_idx.resize(mvpLocalMapLines.size(), false);

    for (size_t i = 0; i < mvpLocalMapLines.size(); i++)
    {
        if(found_idx[i])
            continue;

        Vector6d v1 = mvpLocalMapLines[i]->GetWorldPos();
        cv::Mat sp1 = (Mat_<double>(3, 1) << v1(0), v1(1), v1(2));
        cv::Mat ep1 = (Mat_<double>(3, 1) << v1(3), v1(4), v1(5));
        cv::Mat l1 = ep1 - sp1;

        std::vector<int> sim_l_idx;

        sim_l_idx.push_back(i);

        for (size_t j = i + 1; j < mvpLocalMapLines.size(); j++)
        {
            if(found_idx[j])
            continue;

            // Descriptor distance
            int desc_dist = norm(mvpLocalMapLines[i]->GetDescriptor(), mvpLocalMapLines[j]->GetDescriptor(), NORM_HAMMING);

            if (desc_dist > 100)
                continue;

            Vector6d v2 = mvpLocalMapLines[j]->GetWorldPos();

            cv::Mat sp2 = (Mat_<double>(3, 1) << v2(0), v2(1), v2(2));
            cv::Mat ep2 = (Mat_<double>(3, 1) << v2(3), v2(4), v2(5));
            cv::Mat l2 = ep2 - sp2;
          
            // Evaluate orientation
            double angle = mpLSDextractorLeft->computeAngle(l1, l2);

            if (angle < th_angle)
                continue;

            float pt_l_dist = PointToLineDist(sp1, ep1, sp2, ep2);

            if (pt_l_dist > 0.02)
                continue;

            // Evaluate euclidean distance between pts
            cv::Mat m_distances = (Mat_<float>(1, 4) << cv::norm(sp1 - sp2), cv::norm(ep1 - ep2), cv::norm(sp1 - ep2), cv::norm(ep1 - sp2));

            // Get minimum value;
            double min, max;
            cv::minMaxLoc(m_distances, &min, &max);

            if (min > 0.2)
                continue;

            sim_l_idx.push_back(j);
            found_idx[j] = true;
        }

        if(sim_l_idx.size()>1)
            sim_lines_idx.push_back(sim_l_idx);
    }
}

void Tracking::JoinLines(const std::vector<std::vector<int>> &sim_lines_idx)
{
    for (size_t i = 0; i < sim_lines_idx.size(); i++)
    {
        std::vector<int> single_sim_lines = sim_lines_idx[i];
      
        int rep_idx = ComputeRepresentMapLine(single_sim_lines);
        int idx_desc = FindRepresentDesc(single_sim_lines);

        // Evaluates if a representative IDX is found
        if(rep_idx < 0)
            continue;

        for (size_t j = 0; j < single_sim_lines.size(); j++)
        {
            if(single_sim_lines[j] == rep_idx)
            {

                if (rep_idx != idx_desc)
                mvpLocalMapLines[rep_idx]->mLDescriptor = mvpLocalMapLines[idx_desc]->GetDescriptor();
                continue;
            }
            mvpLocalMapLines[single_sim_lines[j]]->SetBadFlag();
        }
    }

}

int Tracking::FindRepresentDesc(std::vector<int> v_idxs)
{
    int nl = v_idxs.size();
        std::vector<std::vector<float>> Distances;
        Distances.resize(nl, vector<float>(nl, 0));
        for(size_t i=0; i<nl; i++)
        {
            Distances[i][i]=0;
            for(size_t j=0; j<nl; j++)
            {
                int distij = norm(mvpLocalMapLines[v_idxs[i]]->GetDescriptor(), mvpLocalMapLines[v_idxs[j]]->GetDescriptor(), NORM_HAMMING);

                Distances[i][j]=distij;
                Distances[j][i]=distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for(size_t i=0; i<nl; i++)
        {
            vector<int> vDists(Distances[i].begin(), Distances[i].end());
            sort(vDists.begin(), vDists.end());

            int median = vDists[0.5*(nl-1)];

            if(median<BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }
        return v_idxs[BestIdx];
        // {
        //     unique_lock<mutex> lock(mMutexFeatures);
        //     mLDescriptor = vDescriptors[BestIdx].clone();
        // }
}

int Tracking::ComputeRepresentMapLine(const std::vector<int> &v_idxs)
{
    std::vector<double> lengths;
    lengths.resize(v_idxs.size(), 0);

    double max_length = 0.0;
    int max_length_idx = -1;

    for (size_t i = 0; i < v_idxs.size(); i++)
    {
        Vector6d pts = mvpLocalMapLines[v_idxs[i]]->GetWorldPos();

        cv::Mat spt = (Mat_<float>(3, 1) << pts(0), pts(1), pts(2));
        cv::Mat ept = (Mat_<float>(3, 1) << pts(3), pts(4), pts(5));
        // compute length
        double line_length = abs(cv::norm(ept - spt));

        if(line_length> max_length)
        {
            max_length = line_length;
            max_length_idx = i;
        }

        lengths.push_back(line_length);
    }

   return(v_idxs[max_length_idx]);
}

float Tracking::PointToLineDist(const cv::Mat &sp1, const cv::Mat &ep1, const cv::Mat &sp2, const cv::Mat &ep2)
{

    cv::Mat v = sp1;
    cv::Mat ab = ep2 - sp2;
    cv::Mat av = v - sp2;
    if( av.dot(ab)<= 0.0f )
    {
        return(cv::norm(av));
    }
    cv::Mat bv = v -ep2;

    if( bv.dot(ab)<= 0.0f )
    {
        return(cv::norm(bv));
    }

    return (cv::norm(ab.cross( av ))) / (cv::norm(ab)) ;   
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalLines()
{
    mvpLocalMapLines.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapLine*> vpMLs = pKF->GetMapLineMatches();
        //cout<<"vpMLs MapLine = "<<vpMLs.size()<<endl;

        int qq = 0;
        for(vector<MapLine*>::const_iterator itML=vpMLs.begin(), itEndML=vpMLs.end(); itML!=itEndML; itML++)
        {
            MapLine* pML = *itML;
            if(!pML)
                continue;
            if(pML->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pML->isBad())
            {
                qq++;
                mvpLocalMapLines.push_back(pML);
                pML->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
        //cout<<"qq = "<<qq<<endl;
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    std::cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    std::cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    std::cout << " done" << endl;

    // Reset Loop Closing
    std::cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    std::cout << " done" << endl;

    // Clear BoW Database
    std::cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    std::cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}
///modify by wh
    void Tracking::SearchLocalPlanes() {
        for (vector<MapPlane *>::iterator vit = mCurrentFrame.mvpMapPlanes.begin(), vend = mCurrentFrame.mvpMapPlanes.end();
             vit != vend; vit++) {
            MapPlane *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPlane *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                }
            }
        }
    }

} //namespace ORB_SLAM
