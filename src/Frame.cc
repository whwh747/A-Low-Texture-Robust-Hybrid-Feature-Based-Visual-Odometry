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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>
#include "LocalMapping.h"
#include "lineIterator.h"
#include "SurfaceNormal.h"
#include "Config.h"
#include <unordered_set>


#define USE_CV_RECT
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;
using namespace pcl;

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight), mpManh(frame.mpManh),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     mnScaleLevelsLine(frame.mnScaleLevelsLine),
     mfScaleFactorLine(frame.mfScaleFactorLine), mfLogScaleFactorLine(frame.mfLogScaleFactorLine),
     mvScaleFactorsLine(frame.mvScaleFactorsLine), mvInvScaleFactorsLine(frame.mvInvScaleFactorsLine),
     mvLevelSigma2Line(frame.mvLevelSigma2Line), mvInvLevelSigma2Line(frame.mvInvLevelSigma2Line),
     mLdesc(frame.mLdesc), NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn),mvSupposedVectors(frame.mvSupposedVectors), vManhAxisIdx(frame.vManhAxisIdx), mvPerpLines(frame.mvPerpLines), mvParallelLines(frame.mvParallelLines), mvpMapLines(frame.mvpMapLines), mvLines3D(frame.mvLines3D), 
     mvLineEq(frame.mvLineEq),
     mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions), ImageGray(frame.ImageGray.clone()),vanish_point(frame.vanish_point),vpId(frame.vpId),vp_map(frame.vp_map),index_vp(frame.index_vp),vp_Kd(frame.vp_Kd),mnPlaneNum(frame.mnPlaneNum),mvPlanePoints(frame.mvPlanePoints),mvPlaneCoefficients(frame.mvPlaneCoefficients),mvpMapPlanes(frame.mvpMapPlanes),
     mvpParallelPlanes(frame.mvpParallelPlanes),mvpVerticalPlanes(frame.mvpVerticalPlanes),mvbPlaneOutlier(frame.mvbPlaneOutlier),
     mvbParPlaneOutlier(frame.mvbPlaneOutlier),mvbVerPlaneOutlier(frame.mvbVerPlaneOutlier),fa(frame.fa),pOnL(frame.pOnL),
     ImageDepth(frame.ImageDepth),mVF3DLines(frame.mVF3DLines),
     vSurfaceNormalx(frame.vSurfaceNormalx),vSurfaceNormaly(frame.vSurfaceNormaly),vSurfaceNormalz(frame.vSurfaceNormalz),
     vSurfacePointx(frame.vSurfacePointx),vSurfacePointy(frame.vSurfacePointy),vSurfacePointz(frame.vSurfacePointz),
     vVanishingLinex(frame.vVanishingLinex),vVanishingLiney(frame.vVanishingLiney),vVanishingLinez(frame.vVanishingLinez)
{
    // Points
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    // Lines
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGridForLine[i][j]=frame.mGridForLine[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

/// Stereo Frame
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, Manhattan* manh, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc), mpManh(manh), mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
  
   // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();
    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

/// RGB-D Frame
Frame::Frame(cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor, LINEextractor* lsdextractor, ORBVocabulary* voc, Manhattan* manh, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const cv::Mat &mask, const bool &bManhInit,const double &depthMapFactor)
    :mpORBvocabulary(voc), mpManh(manh), mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),mpLSDextractorLeft(lsdextractor), 
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    imGray.copyTo(ImageGray);

    imDepth.copyTo(ImageDepth);

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // Scale Level Info for line
    mnScaleLevelsLine = mpLSDextractorLeft->GetLevels();
    mfScaleFactorLine = mpLSDextractorLeft->GetScaleFactor();
    mfLogScaleFactorLine = log(mfScaleFactor);
    mvScaleFactorsLine = mpLSDextractorLeft->GetScaleFactors();
    mvInvScaleFactorsLine = mpLSDextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2Line = mpLSDextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2Line = mpLSDextractorLeft->GetInverseScaleSigmaSquares();
  
    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

     mb = mbf / fx;

    ///modify by wh
    cv::Mat depth;
    if (depthMapFactor != 1 || imDepth.type() != CV_32F) {
        imDepth.convertTo(depth, CV_32F, depthMapFactor);
    }

     std::chrono::steady_clock::time_point t1_fet_extr = std::chrono::steady_clock::now();


     if (bManhInit)
     {
         thread threadPoint(&Frame::ExtractORBNDepth, this, imGray, depth);
         thread threadLine(&Frame::ExtractLSD, this, imGray, depth);
         thread threadPlane(&Frame::ComputePlanes,this,depth,imDepth,imGray,K,depthMapFactor);
         threadPlane.join();
         threadPoint.join();
         threadLine.join();
     }
    // Extract pt normals until the Manh. Axes is computed
     else
     {
         thread threadPoint(&Frame::ExtractORBNDepth, this, imGray, depth);
         thread threadLine(&Frame::ExtractLSD, this, imGray, depth);
         thread threadNormals(&Frame::ExtractMainImgPtNormals, this, imDepth, K);
         thread threadPlane(&Frame::ComputePlanes,this,depth,imDepth,imGray,K,depthMapFactor);
         threadPlane.join();
         threadPoint.join();
         threadLine.join();
         threadNormals.join();
     }

    //imDepth.convertTo(imDepth,CV_32F,depthMapFactor);
    std::chrono::steady_clock::time_point t2_fet_extr = std::chrono::steady_clock::now();
    chrono::duration<double> time_fet_extr = chrono::duration_cast<chrono::duration<double>>(t2_fet_extr - t1_fet_extr);
    MTimeFeatExtract = time_fet_extr.count();

    NL = mvKeylinesUn.size();

    if (mvKeys.empty())
        return;

    ///mh还没被初始化时 将有深度的线特征加入mRepLines  后面用来优化mh
    if (!bManhInit)
    {
        // TODO 0: Avoid this conversion
        std::vector<cv::Mat> v_line_eq;

        for (size_t i = 0; i < mvLineEq.size(); i++)
        {
            if (mvLineEq[i][2] == -1 || mvLineEq[i][2] == 0)
                continue;

            cv::Mat line_vector = (Mat_<double>(3, 1) << mvLineEq[i][0],
                                   mvLineEq[i][1],
                                   mvLineEq[i][2]);
            std::vector<cv::Mat> v_line;                                   
            v_line.push_back(line_vector);
            mRepLines.push_back(v_line);
        }
    }

    ////将点和线连接起来  看看效果  在像素坐标系下  计算点与线段的距离
    ///把这个写成一个函数  每次在调用位姿优化之前调用    fa = vector<int>(N,-1);
    ///初始化 值为-1 说明这个点不在任何线上
//    for(int i=0;i<N;i++)fa[i]=-1;
//    for(int i=0;i<NL;i++)
//    {
//        Vector3d l = mvKeyLineFunctions[i];
//        KeyLine line = mvKeylinesUn[i];
//        cv::Mat img = imGray.clone();
//        cvtColor(img,img,CV_GRAY2BGR);
//        cv::line(img,mvKeylinesUn[i].getStartPoint(),mvKeylinesUn[i].getEndPoint(),cv::Scalar(255,0,0),1);
//        for(int j=0;j<mvKeysUn.size();j++)
//        {
//            cv::Point2f p = mvKeysUn[j].pt;
//            double dis = (p.x*l[0] + p.y*l[1] + l[2]) / sqrt(l[0]*l[0] + l[1]*l[1]);
//            if(abs(dis)<1)
//            {
//                ///当距离小于一定的值还不够 还需要点在两个轴上的投影在线段得到投影之内
//                ///这个条件还不行   点到线段中点的距离必须小于等于线段长度的1/2
//                //if(p.x<min(line.startPointX,line.endPointX) || p.x>max(line.startPointY,line.endPointX) || p.y<min(line.startPointY,line.endPointY) || p.y>max(line.startPointY,line.endPointY))continue;
//                //if(!(fmin(line.startPointX,line.endPointX)<p.x<fmax(line.startPointX,line.endPointX) && fmin(line.startPointY,line.endPointY)<p.y<fmax(line.startPointY,line.endPointY)))continue;
//
//                if(p.x<fmin(line.startPointX,line.endPointX))continue;
//                if(p.x>fmax(line.startPointX,line.endPointX))continue;
//                if(p.y<fmin(line.startPointY,line.endPointY))continue;
//                if(p.y>fmax(line.startPointY,line.endPointY))continue;
//                fa[j] = i;
//                pOnL++;
//                cv::circle(img,p,1,cv::Scalar(255,0,255),1);
//            }
//        }
////        cv::imshow("wh",img);
////        getchar();
////        cv::destroyWindow("wh");
//    }



     mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
     mvbOutlier = vector<bool>(N, false);

     mvpMapLines = vector<MapLine *>(NL, static_cast<MapLine *>(NULL));
     mvbLineOutlier = vector<bool>(NL, false);

     thread threadAssignPoint(&Frame::AssignFeaturesToGrid, this);
     thread threadAssignLine(&Frame::AssignFeaturesToGridForLine, this);
     threadAssignPoint.join();
     threadAssignLine.join();

     mvParallelLines = vector<std::vector<MapLine *>*>(NL, static_cast<std::vector<MapLine *>*>(nullptr));
     mvPerpLines = vector<std::vector<MapLine *>*>(NL, static_cast<std::vector<MapLine *>*>(nullptr));

    mvParLinesIdx = std::vector<std::vector<int>*>(NL, static_cast<std::vector<int>*>(nullptr));
    mvPerpLinesIdx = std::vector<std::vector<int>*>(NL, static_cast<std::vector<int>*>(nullptr));


        mnPlaneNum = mvPlanePoints.size();
        mvpMapPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpParallelPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpVerticalPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        //mvPlanePointMatches = vector<vector<MapPoint *>>(mnPlaneNum);
        //mvPlaneLineMatches = vector<vector<MapLine *>>(mnPlaneNum);
        mvbPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbVerPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbParPlaneOutlier = vector<bool>(mnPlaneNum, false);


    ///modify bu wh  这里只是通过消失点聚类线特征
    ///计算前默认所有的线条都不是结构线
    isStructLine = vector<bool>(NL,false);
    if(NL > 1)
    {
        getVPHypVia2Lines(mvKeylinesUn , para_vector , length_vector , ori_vector , vpHypo);
        getSphereGrids(mvKeylinesUn , para_vector , length_vector , ori_vector , sphereGrid);
        getBestVpsHyp(sphereGrid , vpHypo , tmp_vps);
        ///走到这里 已经获得了37800个假设中最优的假设  下面使用最优消失点对当前帧的线特征进行将聚类
        line2Vps(mvKeylinesUn , thAngle , tmp_vps , clusters , local_vp_ids);
        ///clusters 存放的是对应的消失点的下标和对应的结构线的下标
//        ///到这里的时候  与三个消失方向平行的线都已经被计算出来
//        vp_line.resize(clusters.size());
//        std::copy(clusters.begin() , clusters.end() , vp_line.begin());
        //drawClusters(imGray , mvKeylinesUn , clusters);
//        cout<<"第 "<<mnId<<" 帧"<<endl;
//        //cout<<"经过消失点聚类后得到的平行的约束有 "<<clusters[0].size()*(clusters[0].size()-1)/2 + clusters[1].size()*(clusters[1].size()-1)/2 + clusters[2].size()*(clusters[2].size()-1)/2<<endl;
//        //cout<<"经过消失点聚类后得到的垂直的约束有 "<<clusters[0].size()*clusters[1].size() + clusters[0].size()*clusters[2].size() + clusters[1].size()*clusters[2].size()<<endl;
//        //cout<<"good line 3d = "<<good_line3d<<endl;
//        //cout<<"tmp_vps = "<<tmp_vps[0]<<" "<<tmp_vps[1]<<" "<<tmp_vps[2]<<endl;
//        para_vector.clear();
//        length_vector.clear();
//        ori_vector.clear();
//        vpHypo.clear();
//        sphereGrid.clear();
//        tmp_vps.clear();
//        clusters.clear();
//        local_vp_ids.clear();
    }
}

/// RGB Frame
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *orbextractor, LINEextractor *lsdextractor, ORBVocabulary *voc, Manhattan* manh, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const cv::Mat &mask)
    : mpORBvocabulary(voc), mpManh(manh), mpORBextractorLeft(orbextractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)), mpLSDextractorLeft(lsdextractor),
      mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    imGray.copyTo(ImageGray);

    // Scale Level Info for point
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // Scale Level Info for line
    mnScaleLevelsLine = mpLSDextractorLeft->GetLevels();
    mfScaleFactorLine = mpLSDextractorLeft->GetScaleFactor();
    mfLogScaleFactorLine = log(mfScaleFactor);
    mvScaleFactorsLine = mpLSDextractorLeft->GetScaleFactors();
    mvInvScaleFactorsLine = mpLSDextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2Line = mpLSDextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2Line = mpLSDextractorLeft->GetInverseScaleSigmaSquares();

    cv::Mat mUndistX, mUndistY, mImGray_remap;
    initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3,3), mK, Size(imGray.cols, imGray.rows), CV_32F, mUndistX, mUndistY);
    cv::remap(imGray, mImGray_remap, mUndistX, mUndistY, cv::INTER_LINEAR);

    thread threadPoint(&Frame::ExtractORB, this, 0, imGray);
    thread threadLine(&Frame::ExtractLSD, this, mImGray_remap, mask);
    threadPoint.join();
    threadLine.join();

    NL = mvKeylinesUn.size(); 
    N = mvKeys.size();

    if(mvKeys.empty())
        return;
        
    //mvKeysUn = mvKeys;
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));  
    mvbOutlier = vector<bool>(N,false);   

    mvpMapLines = vector<MapLine*>(NL,static_cast<MapLine*>(NULL));
    mvbLineOutlier = vector<bool>(NL,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    thread threadAssignPoint(&Frame::AssignFeaturesToGrid, this);
    thread threadAssignLine(&Frame::AssignFeaturesToGridForLine, this);
    threadAssignPoint.join();
    threadAssignLine.join();

}


void Frame::getVPHypVia2Lines(vector<cv::line_descriptor::KeyLine> cur_keyline, vector<Eigen::Vector3d> &para_vector,
                              vector<double> &length_vector, vector<double> &ori_vector,
                              vector<vector<Eigen::Vector3d>> &vpHypo){
        int num = cur_keyline.size();
        ///给定线段的离群比0.5  即有一半的线段没有相应的灭点
        double noiseRatio = 0.5;
        double p = 1.0 / 3.0 * pow( 1.0 - noiseRatio, 2 );
        double confEfficience = 0.9999;
        ///it是至少获取一个两线MSS所需的迭代次数
        int it = log( 1 - confEfficience ) / log( 1.0 - p );
        int numVp2 = 360;
        double stepVp2 = 2.0 * CV_PI / numVp2;
        for ( int i = 0; i < num; ++i )
        {
            Vector3d p1(cur_keyline[i].getStartPoint().x, cur_keyline[i].getStartPoint().y, 1.0);
            Vector3d p2(cur_keyline[i].getEndPoint().x, cur_keyline[i].getEndPoint().y, 1.0);
            ///这里的p1 和 p2是像素平面线特征的端点加了第三维  第三维都是1  叉积是线函数  未归一化的线函数
            para_vector.push_back(p1.cross( p2 ));

            double dx = cur_keyline[i].getEndPoint().x - cur_keyline[i].getStartPoint().x;
            double dy = cur_keyline[i].getEndPoint().y - cur_keyline[i].getStartPoint().y;
            ///像素平面的线特征的长度
            length_vector.push_back(sqrt( dx * dx + dy * dy ));

            ///二维线特征的方向
            double orientation = atan2( dy, dx );
            if ( orientation < 0 )
            {
                orientation += CV_PI;
            }
            ori_vector.push_back(orientation);
        }
        vpHypo = std::vector<std::vector<Vector3d> > ( it * numVp2, std::vector<Vector3d>(4) );
        int count = 0;
        srand((unsigned)time(NULL));
        for ( int i = 0; i < it; ++ i )
        {
            ///随机取两个不同的线特征
            int idx1 = rand() % num;
            int idx2 = rand() % num;
            while ( idx2 == idx1 )
            {
                idx2 = rand() % num;
            }

            // get the vp1
            ///求两个线函数的交点
            Vector3d vp1_Img = para_vector[idx1].cross( para_vector[idx2] );
            if ( vp1_Img(2) == 0 )
            {
                i --;
                continue;
            }

            ///将两条线的交点转换为等效球面上的单位向量
            Vector3d vp1(vp1_Img(0) / vp1_Img(2) - cx,
                         vp1_Img(1) / vp1_Img(2) - cy,
                         fx );
            if ( vp1(2) == 0 ) { vp1(2) = 0.0011; }
            double N = sqrt( vp1(0) * vp1(0) + vp1(1) * vp1(1) + vp1(2) * vp1(2) );
            vp1 *= 1.0 / N;

            // get the vp2 and vp3
            Vector3d vp2( 0.0, 0.0, 0.0 );
            Vector3d vp3( 0.0, 0.0, 0.0 );
            Vector3d vp4( 0.0, 0.0, 0.0 );

            ///vp1对应的大圆的一圈 以1度作为精度  共采样360个
            for ( int j = 0; j < numVp2; ++ j )
            {
                // vp2
                double lambda = j * stepVp2;

                double k1 = vp1(0) * sin( lambda ) + vp1(1) * cos( lambda );
                double k2 = vp1(2);
                double phi = atan( - k2 / k1 );

                double Z = cos( phi );
                double X = sin( phi ) * sin( lambda );
                double Y = sin( phi ) * cos( lambda );

                vp2(0) = X;  vp2(1) = Y;  vp2(2) = Z;
                if ( vp2(2) == 0.0 ) { vp2(2) = 0.0011; }
                N = sqrt( vp2(0) * vp2(0) + vp2(1) * vp2(1) + vp2(2) * vp2(2) );
                vp2 *= 1.0 / N;
                if ( vp2(2) < 0 ) { vp2 *= -1.0; }

                // vp3
                vp3 = vp1.cross( vp2 );
                if ( vp3(2) == 0.0 ) { vp3(2) = 0.0011; }
                N = sqrt( vp3(0) * vp3(0) + vp3(1) * vp3(1) + vp3(2) * vp3(2) );
                vp3 *= 1.0 / N;
                if ( vp3(2) < 0 ) { vp3 *= -1.0; }
                //
                vpHypo[count][0] = Vector3d( vp1(0), vp1(1), vp1(2) );
                vpHypo[count][1] = Vector3d( vp2(0), vp2(1), vp2(2) );
                vpHypo[count][2] = Vector3d( vp3(0), vp3(1), vp3(2) );

                count ++;
            }
        }
        ///一共有 105 x 360 = 37800 个消失点假设
        //cout<<"初步得到的消失点假设共有  "<<vpHypo.size()<<endl;
}
void Frame::getSphereGrids(vector<cv::line_descriptor::KeyLine> cur_keyline, vector<Eigen::Vector3d> &para_vector,
                           vector<double> &length_vector, vector<double> &ori_vector,
                           vector<vector<double>> &sphereGrid) {
    // build sphere grid with 1 degree accuracy
    double angelAccuracy = 1.0 / 180.0 * CV_PI;
    /// 90 x 360
    double angleSpanLA = CV_PI / 2.0;
    double angleSpanLO = CV_PI * 2.0;
    int gridLA = angleSpanLA / angelAccuracy;
    int gridLO = angleSpanLO / angelAccuracy;

    sphereGrid = std::vector< std::vector<double> >( gridLA, std::vector<double>(gridLO) );
    for ( int i=0; i<gridLA; ++i )
    {
        for ( int j=0; j<gridLO; ++j )
        {
            sphereGrid[i][j] = 0.0;
        }
    }

    // put intersection points into the grid
    double angelTolerance = 60.0 / 180.0 * CV_PI;
    Vector3d ptIntersect;
    double x = 0.0, y = 0.0;
    double X = 0.0, Y = 0.0, Z = 0.0, N = 0.0;
    double latitude = 0.0, longitude = 0.0;
    int LA = 0, LO = 0;
    double angleDev = 0.0;
    for ( int i=0; i<cur_keyline.size()-1; ++i )
    {
        for ( int j=i+1; j<cur_keyline.size(); ++j )
        {
            ptIntersect = para_vector[i].cross( para_vector[j] );

            ///拿到符合条件的交点
            if ( ptIntersect(2) == 0 )
            {
                continue;
            }

            ///归一化的像素坐标
            x = ptIntersect(0) / ptIntersect(2);
            y = ptIntersect(1) / ptIntersect(2);

            ///相机坐标
            X = x - cx;
            Y = y - cy;
            Z = fx;
            N = sqrt( X * X + Y * Y + Z * Z );

            ///计算经纬度
            latitude = acos( Z / N );
            longitude = atan2( X, Y ) + CV_PI;

            LA = int( latitude / angelAccuracy );
            ///确定在极坐标上的位置
            if ( LA >= gridLA )
            {
                LA = gridLA - 1;
            }

            LO = int( longitude / angelAccuracy );
            if ( LO >= gridLO )
            {
                LO = gridLO - 1;
            }

            ///  不清楚这是在干嘛
            angleDev = abs( ori_vector[i] - ori_vector[j] );
            angleDev = min( CV_PI - angleDev, angleDev );
            if ( angleDev > angelTolerance )
            {
                continue;
            }

            ///累加评分
            sphereGrid[LA][LO] += sqrt( length_vector[i] * length_vector[j] ) * ( sin( 2.0 * angleDev ) + 0.2 ); // 0.2 is much robuster
        }
    }

    //
    int halfSize = 1;
    int winSize = halfSize * 2 + 1;
    int neighNum = winSize * winSize;

    // get the weighted line length of each grid
    std::vector< std::vector<double> > sphereGridNew = std::vector< std::vector<double> >( gridLA, std::vector<double>(gridLO) );
    for ( int i=halfSize; i<gridLA-halfSize; ++i )
    {
        for ( int j=halfSize; j<gridLO-halfSize; ++j )
        {
            double neighborTotal = 0.0;
            for ( int m=0; m<winSize; ++m )
            {
                for ( int n=0; n<winSize; ++n )
                {
                    neighborTotal += sphereGrid[i-halfSize+m][j-halfSize+n];
                }
            }

            sphereGridNew[i][j] = sphereGrid[i][j] + neighborTotal / neighNum;
        }
    }
    sphereGrid = sphereGridNew;
}
void Frame::getBestVpsHyp(vector<vector<double>> &sphereGrid, std::vector<vector<Vector3d>> &vpHypo,
                          vector<Eigen::Vector3d> &vps) {
    ///所有消失点假设
    int num = vpHypo.size();
    ///精度
    double oneDegree = 1.0 / 180.0 * CV_PI;

    // get the corresponding line length of every hypotheses
    std::vector<double> lineLength( num, 0.0 );
    for ( int i = 0; i < num; ++ i )
    {
        std::vector<cv::Point2d> vpLALO( 3 );
        for ( int j = 0; j < 3; ++ j )
        {
            ///去掉不合理的消失点假设
            if ( vpHypo[i][j](2) == 0.0 )
            {
                continue;
            }

            if ( vpHypo[i][j](2) > 1.0 || vpHypo[i][j](2) < -1.0 )
            {
                cout<<1.0000<<endl;
            }
            double latitude = acos( vpHypo[i][j](2) );
            double longitude = atan2( vpHypo[i][j](0), vpHypo[i][j](1) ) + CV_PI;

            int gridLA = int( latitude / oneDegree );
            if ( gridLA == 90 )
            {
                gridLA = 89;
            }

            int gridLO = int( longitude / oneDegree );
            if ( gridLO == 360 )
            {
                gridLO = 359;
            }

            lineLength[i] += sphereGrid[gridLA][gridLO];
        }
    }

    // get the best hypotheses
    int bestIdx = 0;
    double maxLength = 0.0;
    for ( int i = 0; i < num; ++ i )
    {
        if ( lineLength[i] > maxLength )
        {
            maxLength = lineLength[i];
            bestIdx = i;
        }
    }

    vps = vpHypo[bestIdx];
}
void Frame::line2Vps(vector<cv::line_descriptor::KeyLine> cur_keyline, double thAngle, vector<Eigen::Vector3d> &vps,
                     vector<vector<int>> &clusters, vector<int> &vp_idx) {
    clusters.clear();
    clusters.resize( 3 );

    int vps_size = 3;
    //get the corresponding vanish points on the image plane
    ///将三个消失方向的三个消失点转到像素平面
    std::vector<cv::Point2d> vp2D( vps_size );
    for ( int i = 0; i < vps_size; ++ i )
    {
        vp2D[i].x =  vps[i](0) * fx /
                     vps[i](2) + cx;
        vp2D[i].y =  vps[i](1) * fy /
                     vps[i](2) + cy;
    }

    for ( int i = 0; i < cur_keyline.size(); ++ i )
    {
        double x1 = cur_keyline[i].getStartPoint().x;
        double y1 = cur_keyline[i].getStartPoint().y;
        double x2 = cur_keyline[i].getEndPoint().x;
        double y2 = cur_keyline[i].getEndPoint().y;
        double xm = ( x1 + x2 ) / 2.0;
        double ym = ( y1 + y2 ) / 2.0;

        double v1x = x1 - x2;
        double v1y = y1 - y2;
        double N1 = sqrt( v1x * v1x + v1y * v1y );
        v1x /= N1;   v1y /= N1;

        double minAngle = 1000.0;
        int bestIdx = 0;
        for ( int j = 0; j < vps_size; ++ j )
        {
            double v2x = vp2D[j].x - xm;
            double v2y = vp2D[j].y - ym;
            double N2 = sqrt( v2x * v2x + v2y * v2y );
            v2x /= N2;  v2y /= N2;

            double crossValue = v1x * v2x + v1y * v2y;
            if ( crossValue > 1.0 )
            {
                crossValue = 1.0;
            }
            if ( crossValue < -1.0 )
            {
                crossValue = -1.0;
            }
            double angle = acos( crossValue );
            angle = min( CV_PI - angle, angle );

            if ( angle < minAngle )
            {
                minAngle = angle;
                bestIdx = j;
            }
        }

        ///这里的阈值是1度
        if ( minAngle < thAngle )
        {
            clusters[bestIdx].push_back( i );
            vp_idx.push_back(bestIdx);
            ///到这里说明是结构线
            isStructLine[i] = true;
        }
        else
            vp_idx.push_back(3);
    }
}
void Frame::drawClusters(cv::Mat &img, vector<cv::line_descriptor::KeyLine> &lines, vector<vector<int>> &clusters) {
    Mat vp_img = img.clone();
    Mat line_img = img.clone();
    int cols = img.cols;
    int rows = img.rows;
    vector< Vector3d > vp_c = tmp_vps;
    vector<cv::Point2d > vp2d(3);

    cvtColor(vp_img, vp_img, CV_GRAY2BGR);
    cvtColor(line_img, line_img, CV_GRAY2BGR);
    //draw lines
    std::vector<cv::Scalar> lineColors( 3 );
    ///blue
    lineColors[0] = cv::Scalar( 0, 0, 255 );
    ///green
    lineColors[1] = cv::Scalar( 0, 255, 0 );
    ///red
    lineColors[2] = cv::Scalar( 255, 0, 0 );
//    lineColors[3] = cv::Scalar( 0, 255, 255 );

    for ( int i=0; i<lines.size(); ++i )
    {
        int idx = i;
        cv::Point2f pt_s = lines[i].getStartPoint();
        cv::Point2f pt_e = lines[i].getEndPoint();
        cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

        ///black
        //cv::line( vp_img, pt_s, pt_e, cv::Scalar(0,255,255), 2, CV_AA );
        ///青色
        cv::line( line_img, pt_s, pt_e, cv::Scalar(0,255,255), 2, CV_AA );
    }

    for ( int i = 0; i < clusters.size(); ++i )
    {
        //cout<<clusters.size()<<"   "<<clusters[i].size()<<endl;
        for ( int j = 0; j < clusters[i].size(); ++j )
        {
            int idx = clusters[i][j];

            cv::Point2f pt_s = lines[idx].getStartPoint();
            cv::Point2f pt_e = lines[idx].getEndPoint();
            cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

            cv::line( vp_img, pt_s, pt_e, lineColors[i], 2, CV_AA );
        }
    }
    //imshow("img", img);
    imshow("line img", line_img);
    imshow("vp img", vp_img);
    waitKey(1);
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::AssignFeaturesToGridForLine()
{
    int nReserve = 0.5f*NL/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGridForLine[i][j].reserve(nReserve);

    //#pragma omp parallel for
    for(int i=0;i<NL;i++)
    {
        const KeyLine &kl = mvKeylinesUn[i];

        list<pair<int, int>> line_coords;

        LineIterator* it = new LineIterator(kl.startPointX * mfGridElementWidthInv, kl.startPointY * mfGridElementHeightInv, kl.endPointX * mfGridElementWidthInv, kl.endPointY * mfGridElementHeightInv);

        std::pair<int, int> p;
        while (it->getNext(p))
            if (p.first >= 0 && p.first < FRAME_GRID_COLS && p.second >= 0 && p.second < FRAME_GRID_ROWS)
                mGridForLine[p.first][p.second].push_back(i);

        delete [] it;
    }
}

void Frame::ExtractORBNDepth(const cv::Mat &im, const cv::Mat &im_depth)
{
    (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);

    N = mvKeys.size();
    if (!mvKeys.empty())
    {
     UndistortKeyPoints();
    ComputeStereoFromRGBD(im_depth);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im )
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}


void Frame::ExtractLSD(const cv::Mat &im, const cv::Mat &im_depth)
{
     std::chrono::steady_clock::time_point t1_line_detect = std::chrono::steady_clock::now();

    cv::Mat mask;
    ///mvKeylinesUn是LSD提取到的所有的2d线特征 mvKeyLineFunctions线特征的归一化线函数
    mvKeylinesUn.clear();
    (*mpLSDextractorLeft)(im,mask,mvKeylinesUn, mLdesc, mvKeyLineFunctions);

     std::chrono::steady_clock::time_point t2_line_detect = std::chrono::steady_clock::now();
     chrono::duration<double> t_line_detect = chrono::duration_cast<chrono::duration<double>>(t2_line_detect - t1_line_detect);

     std::chrono::steady_clock::time_point t1_line_good = std::chrono::steady_clock::now();
     ////octavexy和xy是一样的
//     cout<<"nl = "<<NL<<endl;
//     for(int i=0;i<NL;i++)
//     {
//         cout<<"response = "<<mvKeylinesUn[i].response<<" num of pixel = "<<mvKeylinesUn[i].numOfPixels<<endl;
//         cout<<"class id = "<<mvKeylinesUn[i].class_id<<endl;
//         //cout<<"line"<<i<<" octavexy = "<<mvKeylinesUn[i].getStartPointInOctave()<<" xy = "<<mvKeylinesUn[i].getStartPoint()<<endl;
//     }

    ///
//    cv::Mat img = im.clone();
//    cvtColor(img,img,CV_GRAY2BGR);
//    for(int i=0;i<NL;i++)
//    {
//        KeyLine a = mvKeylinesUn[i];
//        cv::line(img,a.getStartPoint(),a.getEndPoint(),cv::Scalar(0,255,0),1);
//    }
//    cout<<"current mnid = "<<mnId<<endl;
//    cv::imshow("ly",img);
//    getchar();
    ///


    robust_Line.resize(NL);


    cullingLine(im,5,2.5,15,30);
//    cullingLine(im,7.5,3,20,30);
//    cullingLine(im,15,1.5,10,30);

    // Option 1: git yanyan-li/PlanarSLAM
     isLineGood(im, im_depth, mK);

    // Option 2: Changes proposed by us
    //  ComputeStereoFromRGBDLines(im_depth);
    
    // Option 3 Single Backprojection procedure
    //  ComputeDepthLines(im_depth);

     std::chrono::steady_clock::time_point t2_line_good = std::chrono::steady_clock::now();
     chrono::duration<double> time_line_good = chrono::duration_cast<chrono::duration<double>>(t2_line_good - t1_line_good);
}


void Frame::cullingLine(const cv::Mat &imGray, const double dis,const double angle,const double endpoint_dis,const double min_len_pow)
{
    robust_Line.clear();
    int num = mvKeylinesUn.size();
    ///用来标记当前线段是否已经被标记为合并   true为已合并
    bool tag[num];
    fill(tag,tag+num,false);
    for(int i=0;i<num;i++)
    {
        robust_Line.resize(num);
        if(tag[i])continue;
        KeyLine l1 = mvKeylinesUn[i];
        Vector3d line1 = mvKeyLineFunctions[i];
        Vector4d v1;
        v1 << l1.startPointX , l1.startPointY , l1.endPointX , l1.endPointY;
        for(int j=i+1;j<num;j++)
        {
            if(tag[j])continue;
            KeyLine l2 = mvKeylinesUn[j];
            Vector3d line2 = mvKeyLineFunctions[j];
            Vector4d v2;
            v2 << l2.startPointX , l2.startPointY , l2.endPointX , l2.endPointY;
            cv::Point2f mid_point12 = (l1.getStartPoint() + l1.getEndPoint()) * 0.5;
            cv::Point2f mid_point21 = (l2.getEndPoint() + l2.getStartPoint()) * 0.5;
            mid_point21 += l2.getStartPoint();
            double dis12 = PointLineDistance(v2 , mid_point12);
            double dis21 = PointLineDistance(v1 , mid_point21);
            if(!(dis12 < dis || dis21 < dis))continue;
            double x11 = l1.startPointX;
            double x12 = l1.endPointX;
            double y11 = l1.startPointY;
            double y12 = l1.endPointY;
            double x21 = l2.startPointX;
            double x22 = l2.endPointX;
            double y21 = l2.startPointY;
            double y22 = l2.endPointY;
            double angle_2line = TwoLineAngle(line1,line2);
            if(abs(angle_2line) > cos(angle*0.0174533))
            {
                ///判断端点的差值
                ///先判断在x轴和y轴的投影是否有重叠
                vector<double> by_x,by_y;
                by_x.push_back(x11),by_x.push_back(x12),by_x.push_back(x21),by_x.push_back(x22);
                by_y.push_back(y11),by_y.push_back(y12),by_y.push_back(y21),by_y.push_back(y22);
                sort(by_x.begin(),by_x.end());
                sort(by_y.begin(),by_y.end());
                double dx = by_x[3] - by_x[0];
                double dy = by_y[3] - by_y[0];
                double dx1 = std::abs(x11 - x12);
                double dx2 = std::abs(x21 - x22);
                double dy1 = std::abs(y11 - y12);
                double dy2 = std::abs(y21 - y22);
                ///x轴不重叠
                if(dx > dx1+dx2)
                {
                    ///两个最近端点的距离大于阈值
                    if(by_x[2]-by_x[1] > endpoint_dis)continue;
                }
                if(dy > dy1+dy2)
                {
                    if(by_y[2]-by_y[1] > endpoint_dis)continue;
                }
                ////到这里的时候已经确定了i和j条线满足merge条件
                ///在这里可以merge对i号线段与j号线段进行merge
                robust_Line[i].push_back(j);
                tag[i] = true;
                tag[j] = true;
            }
        }
    }

        cv::Mat line_image = imGray.clone();
        cvtColor(line_image,line_image,CV_GRAY2BGR);
        fill(tag,tag+num,false);
        vector< Eigen::Vector4f > new_Line;
        for(int i=0;i<robust_Line.size();i++)
        {
            Eigen::Vector4f x1;
            x1 << mvKeylinesUn[i].startPointX , mvKeylinesUn[i].startPointY,mvKeylinesUn[i].endPointX,mvKeylinesUn[i].endPointY;
            Eigen::Vector4f new_line = x1;
            for(int j=0;j<robust_Line[i].size();j++)
            {
                Eigen::Vector4f y1;
                y1 << mvKeylinesUn[robust_Line[i][j]].startPointX,mvKeylinesUn[robust_Line[i][j]].startPointY,mvKeylinesUn[robust_Line[i][j]].endPointX,mvKeylinesUn[robust_Line[i][j]].endPointY;
                new_line = MergeTwoLines(new_line , y1);
                tag[robust_Line[i][j]] = true;
                tag[i] = true;
            }
            if(robust_Line[i].size()!=0)
            {
                //if(pow(new_line[2]-new_line[0],2)+pow(new_line[3]-new_line[1],2) < min_len_pow*min_len_pow)continue;
                new_Line.push_back(new_line);
            }
            if(robust_Line[i].size()==0 && tag[i]==false)
            {
                KeyLine t = mvKeylinesUn[i];
                //if(pow(t.endPointX-t.startPointX,2)+pow(t.endPointY-t.startPointY,2) < min_len_pow*min_len_pow)continue;
                new_Line.push_back(Vector4f(t.startPointX,t.startPointY,t.endPointX,t.endPointY));
            }
        }
        for(int i=0;i<new_Line.size();i++)
        {
            Vector4f line = new_Line[i];
            cv::line(line_image,cv::Point2f(line[0],line[1]),cv::Point2f(line[2],line[3]),cv::Scalar(255,0,0),1.5);
        }

        mvKeylinesUn.clear();
        mvKeylinesUn.resize(0);
        ///现在得根据new_Line 创建新的KeyLine
        for(int i=0;i<new_Line.size();i++)
        {
            //if(abs(new_Line[i][0] - ImageGray.cols)<5 && abs(new_Line[i][2] - ImageGray.cols)<5)continue;
            //if(abs(new_Line[i][1] - ImageGray.rows)<5 && abs(new_Line[i][3] - ImageGray.rows)<5)continue;

            ///make a keyline
            KeyLine kl;
            kl.startPointX = new_Line[i][0];
            kl.startPointY = new_Line[i][1];
            kl.endPointX = new_Line[i][2];
            kl.endPointY = new_Line[i][3];
            kl.sPointInOctaveX = new_Line[i][0];
            kl.sPointInOctaveY = new_Line[i][1];
            kl.ePointInOctaveX = new_Line[i][2];
            kl.ePointInOctaveY = new_Line[i][3];
            kl.lineLength = (float)sqrt(pow(new_Line[i][0]-new_Line[i][2],2)+pow(new_Line[i][1]-new_Line[i][3],2));
            kl.octave = 0;
            kl.angle = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
            kl.size = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
            kl.pt = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );
            cv::LineIterator li(imGray.clone(),Point2f(new_Line[i][0],new_Line[i][1]),Point2f(new_Line[i][2],new_Line[i][3]));
            kl.numOfPixels = li.count;
            kl.response = kl.lineLength / max(imGray.cols,imGray.rows);
            //if(kl.lineLength < min_len_pow)continue;
            mvKeylinesUn.push_back(kl);
        }
    sort(mvKeylinesUn.begin(),mvKeylinesUn.end(),sort_lines_by_response());
        for(int i=0;i<mvKeylinesUn.size();i++)
        {
            //cout<<"i = "<<i<<"  numofpixes = "<<mvKeylinesUn[i].numOfPixels<<" response = "<<mvKeylinesUn[i].response<<endl;
            mvKeylinesUn[i].class_id = i;
        }
        ///mvKeylinesUn 已经修改完成   接下来计算描述子和mvKeyLineFunctions
    cv::Mat descriptors;
    Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
        lbd->compute(imGray.clone(),mvKeylinesUn,descriptors);
        mvKeyLineFunctions.clear();
        mvKeyLineFunctions.resize(0);
        for(vector<KeyLine>::iterator it = mvKeylinesUn.begin();it!=mvKeylinesUn.end();it++)
        {
            Eigen::Vector3d sp_l;
            sp_l << it->startPointX, it->startPointY, 1.0;
            Eigen::Vector3d ep_l;
            ep_l << it->endPointX, it->endPointY, 1.0;
            Eigen::Vector3d lineV;
            lineV << sp_l.cross(ep_l);
            lineV = lineV / sqrt(lineV(0) * lineV(0) + lineV(1) * lineV(1));
            mvKeyLineFunctions.push_back(lineV);
        }
        mLdesc.release();
        descriptors.copyTo(mLdesc);
//        cout<<"new line size = "<<new_Line.size()<<endl;
//        cv::imshow("ly",line_image);
//        getchar();
//        cv::destroyWindow("ly");
}
    double Frame::PointLineDistance(Eigen::Vector4d line, cv::Point2f point){
        double x0 = (double)point.x;
        double y0 = (double)point.y;
        double x1 = line(0);
        double y1 = line(1);
        double x2 = line(2);
        double y2 = line(3);
        double d = (std::fabs((y2 - y1) * x0 +(x1 - x2) * y0 + ((x2 * y1) -(x1 * y2)))) / (std::sqrt(std::pow(y2 - y1, 2) + std::pow(x1 - x2, 2)));
        return d;
    }
    double Frame::TwoLineAngle(Eigen::Vector3d line1, Eigen::Vector3d line2)
    {
        Vector3d v1 = line1;
        Vector3d v2 = line2;
        v1[0]/=v1[2] , v1[1]/=v1[2];
        v2[0]/=v2[2] , v2[1]/=v2[2];
        cv::Mat l1 = (Mat_<double>(2,1) << v1[0]/v1[2] , v1[1]/v1[2]);
        cv::Mat l2 = (Mat_<double>(2,1) << v2[0]/v2[2] , v2[1]/v2[2]);
        double a = l1.dot(l2);
        double b = std::sqrt(l1.at<double>(0) * l1.at<double>(0) + l1.at<double>(1)*l1.at<double>(1));
        double c = std::sqrt(l2.at<double>(0) * l2.at<double>(0) + l2.at<double>(1)*l2.at<double>(1));
        return abs(a / (b * c));
    }
    Eigen::Vector4f Frame:: MergeTwoLines(const Eigen::Vector4f& line1, const Eigen::Vector4f& line2){
        double xg = 0.0, yg = 0.0;
        double delta1x = 0.0, delta1y = 0.0, delta2x = 0.0, delta2y = 0.0;
        float ax = 0, bx = 0, cx = 0, dx = 0;
        float ay = 0, by = 0, cy = 0, dy = 0;
        double li = 0.0, lj = 0.0;
        double thi = 0.0, thj = 0.0, thr = 0.0;
        double axg = 0.0, bxg = 0.0, cxg = 0.0, dxg = 0.0, delta1xg = 0.0, delta2xg = 0.0;

        ax = line1(0);
        ay = line1(1);
        bx = line1(2);
        by = line1(3);

        cx = line2(0);
        cy = line2(1);
        dx = line2(2);
        dy = line2(3);

        float dlix = (bx - ax);
        float dliy = (by - ay);
        float dljx = (dx - cx);
        float dljy = (dy - cy);

        li = sqrt((double) (dlix * dlix) + (double) (dliy * dliy));
        lj = sqrt((double) (dljx * dljx) + (double) (dljy * dljy));

        xg = (li * (double) (ax + bx) + lj * (double) (cx + dx))
             / (double) (2.0 * (li + lj));
        yg = (li * (double) (ay + by) + lj * (double) (cy + dy))
             / (double) (2.0 * (li + lj));

        if(dlix == 0.0f) thi = CV_PI / 2.0;
        else thi = atan(dliy / dlix);

        if(dljx == 0.0f) thj = CV_PI / 2.0;
        else thj = atan(dljy / dljx);

        if (fabs(thi - thj) <= CV_PI / 2.0){
            thr = (li * thi + lj * thj) / (li + lj);
        }
        else{
            double tmp = thj - CV_PI * (thj / fabs(thj));
            thr = li * thi + lj * tmp;
            thr /= (li + lj);
        }

        axg = ((double) ay - yg) * sin(thr) + ((double) ax - xg) * cos(thr);
        bxg = ((double) by - yg) * sin(thr) + ((double) bx - xg) * cos(thr);
        cxg = ((double) cy - yg) * sin(thr) + ((double) cx - xg) * cos(thr);
        dxg = ((double) dy - yg) * sin(thr) + ((double) dx - xg) * cos(thr);

        delta1xg = std::min(axg, std::min(bxg, std::min(cxg,dxg)));
        delta2xg = std::max(axg, std::max(bxg, std::max(cxg,dxg)));

        delta1x = delta1xg * std::cos(thr) + xg;
        delta1y = delta1xg * std::sin(thr) + yg;
        delta2x = delta2xg * std::cos(thr) + xg;
        delta2y = delta2xg * std::sin(thr) + yg;

        Eigen::Vector4f new_line;
        new_line << (float)delta1x, (float)delta1y, (float)delta2x, (float)delta2y;
        return new_line;
    }
// Optimize Lines --> Small mods of the code from YanYan Li ICRA 2021
void Frame::isLineGood(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &K)
{
    mvLineEq.clear();
    mvLineEq.resize(mvKeylinesUn.size(),Vec3f(-1.0, -1.0, -1.0));
    mvLineNor.resize(mvKeylinesUn.size(),Vector3d (-1.0, -1.0, -1.0));
    mvLines3D.resize(mvKeylinesUn.size(), std::make_pair(Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0)));

    for (int i = 0; i < mvKeylinesUn.size(); ++i)
    { // each line
        ///2d长度  默认计算L2范数
        double len = cv::norm(mvKeylinesUn[i].getStartPoint() - mvKeylinesUn[i].getEndPoint());
        vector<cv::Point3d> pts3d;
        // iterate through a line
        ///将线切分   还是在像素平面上进行切分
        double numSmp = (double)min((int)len, 20); //number of line points sampled

        pts3d.reserve(numSmp);
        for (int j = 0; j <= numSmp; ++j)
        {
            // use nearest neighbor to querry depth value
            // assuming position (0,0) is the top-left corner of image, then the
            // top-left pixel's center would be (0.5,0.5)
            ///pt是线上切分的点
            cv::Point2d pt = mvKeylinesUn[i].getStartPoint() * (1 - j / numSmp) +
                             mvKeylinesUn[i].getEndPoint() * (j / numSmp);

            ///位置不符合
            if (pt.x < 0 || pt.y < 0 || pt.x >= imDepth.cols || pt.y >= imDepth.rows)
            {
                continue;
            }
            ////计算离该点最近的像素值
            int row, col; // nearest pixel for pt
            if ((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y))
            { // boundary issue
                col = max(int(pt.x - 1), 0);
                row = max(int(pt.y - 1), 0);
            }
            else
            {
                col = int(pt.x);
                row = int(pt.y);
            }

            float d = -1;
            ///深度太小就丢弃
            ////add
            if(row<0 || col<0 || row>=imDepth.cols || col>=imDepth.rows)continue;
            if (imDepth.at<float>(row, col) <= 0.01)
            { 
                continue;
            }
            else
            {
                d = imDepth.at<float>(row, col);
            }
            cv::Point3d p;

            ///将像素坐标转换为相机坐标
            p.z = d;
            p.x = (col - cx) * p.z * invfx;
            p.y = (row - cy) * p.z * invfy;

            pts3d.push_back(p);
        }

        ///线特征上深度符合的点数小于5  丢弃这条线
        if (pts3d.size() < 5){
            continue;
        }

        RandomLine3d tmpLine;
        vector<RandomPoint3d> rndpts3d;
        rndpts3d.reserve(pts3d.size());

        // compute uncertainty of 3d points
        ////后面这一步就理解为 对当前的线段进行优化 满足条件时得到了一条更鲁棒的线  剩下的都是在相机坐标系下的
        for (int j = 0; j < pts3d.size(); ++j)
        {
            rndpts3d.push_back(mpLSDextractorLeft->compPt3dCov(pts3d[j], K, 1));
        }
        // using ransac to extract a 3d line from 3d pts
        tmpLine = mpLSDextractorLeft->extract3dline_mahdist(rndpts3d);

        if (
        cv::norm(tmpLine.A - tmpLine.B) > 0.02)
        {
            ///估计出了一条线 tmpLine

            Eigen::Vector3d st_pt3D(tmpLine.A.x, tmpLine.A.y, tmpLine.A.z);
            Eigen::Vector3d e_pt3D(tmpLine.B.x, tmpLine.B.y, tmpLine.B.z);

            ///线的方向向量
            cv::Vec3f line_eq(tmpLine.B.x - tmpLine.A.x, tmpLine.B.y- tmpLine.A.y, tmpLine.B.z - tmpLine.A.z);
            ///线与相机光心所构成的平面的法向量 用两个3d端点的叉乘求得
            Vector3d line_nor = st_pt3D.cross(e_pt3D);
            float magn  = sqrt(line_eq[0] * line_eq[0] + line_eq[1] * line_eq[1]+ line_eq[2] * line_eq[2]);
            std::pair<Eigen::Vector3d, Eigen::Vector3d> line_ep_3D(st_pt3D, e_pt3D);
            ///相机坐标系下3d线的端点
            mvLines3D[i] = line_ep_3D;
            good_line3d++;
            ///相机坐标系下 3d线的方向向量
            mvLineEq[i] = line_eq/magn;
            ///法向量
            mvLineNor[i] = line_nor;

            FrameLine tempLine;
            tempLine.haveDepth = true;
            tempLine.rndpts3d = tmpLine.pts;
            tempLine.direction = tmpLine.director;
            tempLine.direct1 = tmpLine.direct1;
            tempLine.direct2 = tmpLine.direct2;
            tempLine.p = Point2d(mvKeylinesUn[i].endPointX, mvKeylinesUn[i].endPointY);
            tempLine.q = Point2d(mvKeylinesUn[i].startPointX, mvKeylinesUn[i].startPointY);
            mVF3DLines.push_back(tempLine);
        }
    }
}

void Frame::ExtractMainImgPtNormals(const cv::Mat &img, const cv::Mat &K)
{
    // mRepNormals --> used to extract initial candidates of the Manh. Axes in the Coarse Manh. Estimation
    // mvPtNormals --> used to cooroborate and to refine the coarse Manh. Estimation.
     (*mpManh)(img,K,mvPtNormals, mRepNormals);
}

void Frame::lineDescriptorMAD( vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const
{
    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = line_matches;
    matches_12 = line_matches;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_dist_median = matches_nn[int(matches_nn.size()/2)][0].distance;

    for(unsigned int i=0; i<matches_nn.size(); i++)
        matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
    nn12_dist_median = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
    for (unsigned int j=0; j<matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
    sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
    nn12_mad = 1.4826 * matches_12[int(matches_12.size()/2)][0].distance;
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit)
{
    pML->mbTrackInView = false;

    Vector6d P = pML->GetWorldPos();

    cv::Mat SP = (Mat_<float>(3,1) << P(0), P(1), P(2));
    cv::Mat EP = (Mat_<float>(3,1) << P(3), P(4), P(5));

    const cv::Mat SPc = mRcw*SP + mtcw;
    const float &SPcX = SPc.at<float>(0);
    const float &SPcY = SPc.at<float>(1);
    const float &SPcZ = SPc.at<float>(2);

    const cv::Mat EPc = mRcw*EP + mtcw;
    const float &EPcX = EPc.at<float>(0);
    const float &EPcY = EPc.at<float>(1);
    const float &EPcZ = EPc.at<float>(2);

    if(SPcZ<0.0f || EPcZ<0.0f)
        return false;

    const float invz1 = 1.0f/SPcZ;
    const float u1 = fx * SPcX * invz1 + cx;
    const float v1 = fy * SPcY * invz1 + cy;

    if(u1<mnMinX || u1>mnMaxX)
        return false;
    if(v1<mnMinY || v1>mnMaxY)
        return false;

    const float invz2 = 1.0f/EPcZ;
    const float u2 = fx*EPcX*invz2 + cx;
    const float v2 = fy*EPcY*invz2 + cy;

    if(u2<mnMinX || u2>mnMaxX)
        return false;
    if(v2<mnMinY || v2>mnMaxY)
        return false;

    const float maxDistance = pML->GetMaxDistanceInvariance();
    const float minDistance = pML->GetMinDistanceInvariance();
 
    const cv::Mat OM = 0.5*(SP+EP) - mOw;
    const float dist = cv::norm(OM);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Vector3d Pn = pML->GetNormal();
    cv::Mat pn = (Mat_<float>(3,1) << Pn(0), Pn(1), Pn(2));
    const float viewCos = OM.dot(pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

    // Data used by the tracking
    pML->mbTrackInView = true;
    pML->mTrackProjX1 = u1;
    pML->mTrackProjY1 = v1;
    pML->mTrackProjX2 = u2;
    pML->mTrackProjY2 = v2;
    pML->mnTrackScaleLevel = nPredictedLevel;
    pML->mTrackViewCos = viewCos;

    return true;
}
#pragma GCC push_options
#pragma GCC optimze (0)
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}
#pragma GCC pop_options
vector<size_t> Frame::GetFeaturesInAreaForLine(const float &x1, const float &y1, const float &x2, const float &y2, const float  &r, const int minLevel, const int maxLevel,const float TH) const
{
    vector<size_t> vIndices;
    vIndices.reserve(NL);
    unordered_set<size_t> vIndices_set;

    float x[3] = {x1, (x1+x2)/2.0, x2};
    float y[3] = {y1, (y1+y2)/2.0, y2}; 

    float delta1x = x1-x2;
    float delta1y = y1-y2;
    float norm_delta1 = sqrt(delta1x*delta1x + delta1y*delta1y);
    delta1x /= norm_delta1;
    delta1y /= norm_delta1;

    for(int i = 0; i<3;i++){
        const int nMinCellX = max(0,(int)floor((x[i]-mnMinX-r)*mfGridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            continue;

        const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x[i]-mnMinX+r)*mfGridElementWidthInv));
        if(nMaxCellX<0)
            continue;

        const int nMinCellY = max(0,(int)floor((y[i]-mnMinY-r)*mfGridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            continue;

        const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y[i]-mnMinY+r)*mfGridElementHeightInv));
        if(nMaxCellY<0)
            continue;

        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                const vector<size_t> vCell = mGridForLine[ix][iy];
                if(vCell.empty())
                    continue;

                for(size_t j=0, jend=vCell.size(); j<jend; j++)
                {
                    if(vIndices_set.find(vCell[j]) != vIndices_set.end())
                        continue;

                    const KeyLine &klUn = mvKeylinesUn[vCell[j]];

                    float delta2x = klUn.startPointX - klUn.endPointX;
                    float delta2y = klUn.startPointY - klUn.endPointY;
                    float norm_delta2 = sqrt(delta2x*delta2x + delta2y*delta2y);
                    delta2x /= norm_delta2;
                    delta2y /= norm_delta2;
                    float CosSita = abs(delta1x * delta2x + delta1y * delta2y);

                    if(CosSita < TH)
                        continue;

                    Eigen::Vector3d Lfunc = mvKeyLineFunctions[vCell[j]]; 
                    const float dist = Lfunc(0)*x[i] + Lfunc(1)*y[i] + Lfunc(2);

                    if(fabs(dist)<r)
                    {
                        if(vIndices_set.find(vCell[j]) == vIndices_set.end())
                        {
                            vIndices.push_back(vCell[j]);
                            vIndices_set.insert(vCell[j]);
                        }
                    }
                }
            }
        }
    }
    
    return vIndices;
}

vector<size_t> Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                                     const int minLevel, const int maxLevel, const float TH) const
{
    vector<size_t> vIndices;

    vector<KeyLine> vkl = this->mvKeylinesUn;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>0);

    float delta1x = x1-x2;
    float delta1y = y1-y2;
    float norm_delta1 = sqrt(delta1x*delta1x + delta1y*delta1y);
    delta1x /= norm_delta1;
    delta1y /= norm_delta1;

    for(size_t i=0; i<vkl.size(); i++)
    {
        KeyLine keyline = vkl[i];

        float distance = (0.5*(x1+x2)-keyline.pt.x)*(0.5*(x1+x2)-keyline.pt.x)+(0.5*(y1+y2)-keyline.pt.y)*(0.5*(y1+y2)-keyline.pt.y);
        if(distance > r*r)
            continue;

        float delta2x = vkl[i].startPointX - vkl[i].endPointX;
        float delta2y = vkl[i].startPointY - vkl[i].endPointY;
        float norm_delta2 = sqrt(delta2x*delta2x + delta2y*delta2y);
        delta2x /= norm_delta2;
        delta2y /= norm_delta2;
        float CosSita = abs(delta1x * delta2x + delta1y * delta2y);

        if(CosSita < TH)
            continue;

        if(bCheckLevels)
        {
            if(keyline.octave<minLevel)
                continue;
            if(maxLevel>=0 && keyline.octave>maxLevel)
                continue;
        }

        vIndices.push_back(i);
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);   
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N); 
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0 && d < 7.0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

void Frame::ComputeDepthLines(const cv::Mat imDepth)
{
    mvLines3D.clear();
    mvLines3D.resize(mvKeylinesUn.size(), std::make_pair(Vector3d(0.0, 0.0, 0.0),Vector3d(0.0, 0.0, 0.0)));

    mvLineEq.clear();
    mvLineEq.resize(mvKeylinesUn.size(),Vec3f(-1.0, -1.0, -1.0));

     for (int i = 0; i < mvKeylinesUn.size(); i++)
    {
         std::pair<cv::Point3f, cv::Point3f> pair_pts_3D = std::make_pair(cv::Point3f(-1.0, -1.0, -1.0),cv::Point3f(-1.0, -1.0, -1.0));

        if (!ComputeDepthEnpoints(imDepth, mvKeylinesUn[i], mK, pair_pts_3D))
         {
            mvKeylinesUn[i].startPointX = 0;
            mvKeylinesUn[i].startPointY = 0;
            mvKeylinesUn[i].endPointX = 0;
            mvKeylinesUn[i].endPointY = 0;
        }

        else
        {
            mvLineEq[i] = pair_pts_3D.second - pair_pts_3D.first;
         
            Eigen::Vector3d st_pt3D(pair_pts_3D.first.x,
                                    pair_pts_3D.first.y,
                                    pair_pts_3D.first.z);

            Eigen::Vector3d e_pt3D(pair_pts_3D.second.x,
                                   pair_pts_3D.second.y,
                                   pair_pts_3D.second.z);

            std::pair<Eigen::Vector3d, Eigen::Vector3d> line_ep_3D(st_pt3D, e_pt3D);
            mvLines3D[i] = line_ep_3D;
        }
    }
    

}

void Frame::ComputeStereoFromRGBDLines(const cv::Mat imDepth)
{
    mvLines3D.clear();
    mvLines3D.resize(mvKeylinesUn.size(), std::make_pair(Vector3d(0.0, 0.0, 0.0),Vector3d(0.0, 0.0, 0.0)));

    mvLineEq.clear();
    mvLineEq.resize(mvKeylinesUn.size(),Vec3f(-1.0, -1.0, -1.0));

    // #pragma omp parallel for
    for (int i = 0; i < mvKeylinesUn.size(); i++)
    {
        if (mvKeylinesUn[i].lineLength < 10.0)
        {
            mvKeylinesUn[i].startPointX = 0;
            mvKeylinesUn[i].startPointY = 0;
            mvKeylinesUn[i].endPointX = 0;
            mvKeylinesUn[i].endPointY = 0;
            continue;
        }

         std::pair<cv::Point, cv::Point> pair_pts_2D;
         std::pair<cv::Point3f, cv::Point3f> pair_pts_3D = std::make_pair(cv::Point3f(-1.0, -1.0, -1.0),cv::Point3f(-1.0, -1.0, -1.0));
         cv::Vec3f line_vector;

         if (!mpLSDextractorLeft->computeBest3dLineRepr(ImageGray, imDepth, mvKeylinesUn[i], mK, pair_pts_2D, pair_pts_3D, line_vector))
         {
            mvKeylinesUn[i].startPointX = 0;
            mvKeylinesUn[i].startPointY = 0;
            mvKeylinesUn[i].endPointX = 0;
            mvKeylinesUn[i].endPointY = 0;
        }

        else
        {
            mvLineEq[i] = line_vector;
         
            Eigen::Vector3d st_pt3D(pair_pts_3D.first.x,
                                    pair_pts_3D.first.y,
                                    pair_pts_3D.first.z);

            Eigen::Vector3d e_pt3D(pair_pts_3D.second.x,
                                   pair_pts_3D.second.y,
                                   pair_pts_3D.second.z);

            std::pair<Eigen::Vector3d, Eigen::Vector3d> line_ep_3D(st_pt3D, e_pt3D);
            mvLines3D[i] = line_ep_3D;
        }
    }
}

bool Frame::ComputeDepthEnpoints(const cv::Mat &imDepth, const line_descriptor::KeyLine &keyline, const cv::Mat mK, std::pair<cv::Point3f, cv::Point3f> &end_pts3D)
{
    cv::Point2f st_pt = keyline.getStartPoint(); 

    const float &st_v = st_pt.y;
    const float &st_u = st_pt.x;

    const float st_d = imDepth.at<float>(st_v, st_u);

    if (!(st_d > 0 && st_d < 7.0))
        return false;
    
    cv::Point2f end_pt = keyline.getEndPoint();
    
    const float &end_v = end_pt.y;
    const float &end_u = end_pt.x;

    const float end_d = imDepth.at<float>(end_v, end_u);

    if (!(end_d > 0 && end_d < 7.0))
        return false;  
    
    float st_x = ((st_pt.x - mK.at<float>(0, 2)) * st_d) / mK.at<float>(0, 0);
    float st_y = ((st_pt.y - mK.at<float>(1, 2)) * st_d) / mK.at<float>(1, 1);
    cv::Point3f st_point(st_x, st_y, st_d);

    float end_x = ((end_pt.x - mK.at<float>(0, 2)) * end_d) / mK.at<float>(0, 0);
    float end_y = ((end_pt.y - mK.at<float>(1, 2)) * end_d) / mK.at<float>(1, 1);
    cv::Point3f end_point(end_x, end_y, end_d);

    end_pts3D.first =  st_point;
    end_pts3D.second = end_point;
}


cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

    void Frame::ComputePlanes(const cv::Mat &imDepth, const cv::Mat &Depth, const cv::Mat &imRGB, cv::Mat K, float depthMapFactor) {
        ///读入彩色图和深度图
        planeDetector.readColorImage(imRGB);
        planeDetector.readDepthImage(Depth, K, depthMapFactor);
        planeDetector.runPlaneDetection(imDepth.rows, imDepth.cols);

        for (int i = 0; i < planeDetector.plane_num_; i++) {
            ///indices 存储每个平面包含的像素
            auto &indices = planeDetector.plane_vertices_[i];
            ///像素点作为点云存储到PointCloud
            PointCloud::Ptr inputCloud(new PointCloud());
            for (int j : indices) {
                PointT p;
                p.x = (float) planeDetector.cloud.vertices[j][0];
                p.y = (float) planeDetector.cloud.vertices[j][1];
                p.z = (float) planeDetector.cloud.vertices[j][2];

                inputCloud->points.push_back(p);
            }

            auto extractedPlane = planeDetector.plane_filter.extractedPlanes[i];
            double nx = extractedPlane->normal[0];
            double ny = extractedPlane->normal[1];
            double nz = extractedPlane->normal[2];
            double cx = extractedPlane->center[0];
            double cy = extractedPlane->center[1];
            double cz = extractedPlane->center[2];

            float d = (float) -(nx * cx + ny * cy + nz * cz);

            ///过滤每个平面的点云
            pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize(0.1, 0.1, 0.1);

            PointCloud::Ptr coarseCloud(new PointCloud());
            voxel.setInputCloud(inputCloud);
            voxel.filter(*coarseCloud);

            ///Hessian Plane
            cv::Mat coef = (cv::Mat_<float>(4, 1) << nx, ny, nz, d);

            bool valid = MaxPointDistanceFromPlane(coef, coarseCloud);

            if (!valid) {
                continue;
            }

            mvPlanePoints.push_back(*coarseCloud);

            mvPlaneCoefficients.push_back(coef);
        }
        std::vector<SurfaceNormal> surfaceNormals;

        PointCloud::Ptr inputCloud( new PointCloud() );
        for (int m=0; m<imDepth.rows; m+=3)
        {
            for (int n=0; n<imDepth.cols; n+=3)
            {
                float d = imDepth.ptr<float>(m)[n];
                PointT p;
                p.z = d;
                //cout << "depth:" << d<<endl;
                p.x = ( n - cx) * p.z / fx;
                p.y = ( m - cy) * p.z / fy;

                inputCloud->points.push_back(p);
            }
        }
        inputCloud->height = ceil(imDepth.rows/3.0);
        inputCloud->width = ceil(imDepth.cols/3.0);

        //compute normals
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
        ne.setMaxDepthChangeFactor(0.05f);
        ne.setNormalSmoothingSize(10.0f);
        ne.setInputCloud(inputCloud);
        //计算特征值

        if (inputCloud->size()== 0)
        {
            PCL_ERROR ("Could not estimate a planar model for the given initial plane.\n");
            return;
        }
        ne.compute(*cloud_normals);

        for ( int m=0; m<inputCloud->height; m+=1 ) {
            if(m%2==0) continue;
            for (int n = 0; n < inputCloud->width; n+=1) {
                pcl::Normal normal = cloud_normals->at(n, m);
                SurfaceNormal surfaceNormal;
                if(n%2==0) continue;
                surfaceNormal.normal.x = normal.normal_x;
                surfaceNormal.normal.y = normal.normal_y;
                surfaceNormal.normal.z = normal.normal_z;

                pcl::PointXYZRGB point = inputCloud->at(n, m);
                surfaceNormal.cameraPosition.x = point.x;
                surfaceNormal.cameraPosition.y = point.y;
                surfaceNormal.cameraPosition.z = point.z;
                surfaceNormal.FramePosition.x = n*3;
                surfaceNormal.FramePosition.y = m*3;

                surfaceNormals.push_back(surfaceNormal);
            }
        }

        vSurfaceNormal = surfaceNormals;
    }
    bool Frame::MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud) {
    ///tum 0.05  icl 0.03
        //double disTh = 0.05;
        auto disTh = ORB_SLAM2::Config::Get<double>("Plane.DistanceThreshold");
        bool erased = false;
////        double max = -1;
        double threshold = 0.04;
        int i = 0;
        auto &points = pointCloud->points;
////        std::cout << "points before: " << points.size() << std::endl;
        for (auto &p : points) {
            double absDis = abs(plane.at<float>(0) * p.x +
                                plane.at<float>(1) * p.y +
                                plane.at<float>(2) * p.z +
                                plane.at<float>(3));

            if (absDis > disTh)
                return false;
            i++;
        }

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<PointT> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(disTh);

        seg.setInputCloud(pointCloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            PCL_ERROR ("Could not estimate a planar model for the given initial plane.\n");
            return false;
        }

        float oldVal = plane.at<float>(3);
        float newVal = coefficients->values[3];

        cv::Mat oldPlane = plane.clone();


        plane.at<float>(0) = coefficients->values[0];
        plane.at<float>(1) = coefficients->values[1];
        plane.at<float>(2) = coefficients->values[2];
        plane.at<float>(3) = coefficients->values[3];

        if ((newVal < 0 && oldVal > 0) || (newVal > 0 && oldVal < 0)) {
            plane = -plane;
//                double dotProduct = plane.dot(oldPlane) / sqrt(plane.dot(plane) * oldPlane.dot(oldPlane));
//                std::cout << "Flipped plane: " << plane.t() << std::endl;
//                std::cout << "Flip plane: " << dotProduct << std::endl;
        }
//        }

        return true;
    }
    cv::Mat Frame::ComputePlaneWorldCoeff(const int &idx) {
        cv::Mat temp;
        cv::transpose(mTcw, temp);
        cv::Mat b = -mOw.t();
        return temp * mvPlaneCoefficients[idx];
    }

} //namespace ORB_SLAM
