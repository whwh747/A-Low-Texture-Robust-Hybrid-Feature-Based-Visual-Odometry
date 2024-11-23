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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<chrono>


#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "LineExtractor.h"
#include "Manhattan.h"

#include "auxiliar.h"
#include "MapLine.h"
#include "LSDmatcher.h"
#include "PlaneMatcher.h"

#include <mutex>


namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);

    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageRGBD_wh(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
   
    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    ///use par line to compute vp
    void get_vp(int);
    ///use two par line to match vanish point
    void match_vp(Frame &CurrentFrame,vector<int> match_12);
    ///draw current frame vp
    void draw_vp(Frame &CurrentFrame);
    ///use distance to match vanish point
    void match_vpBydis(Frame &CurrentFrame,const Frame &LastFrame , std::vector<int> &match_12,double distance);
    ///draw two frame match vp
    void draw_2vp(Frame &CurrentFrame , const Frame &LastFrame,vector<int> matches);

    void draw(vector< pair<cv::Point2f,cv::Point2f > > vp);
    ///match two frame vp
    void match2frame_vp(Frame &CurrentFrame , Frame &LastFrame );

    /// match point line structure
    int pOnLMatch(Frame &CurrentFrame);

    cv::Mat TrackManhattanFrame(cv::Mat &mLastRcm,vector<SurfaceNormal> &vSurfaceNormal,vector<FrameLine>&vVanishingDirection);
    axiSNV ProjectSN2Conic(int a,const cv::Mat &R_cm,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection);
    ResultOfMS ProjectSN2MF(int a,const cv::Mat &R_cm,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection,const int numOfSN);
    sMS MeanShift(vector<cv::Point2d> & v2D);

public:
    int match_true = 0;

    // Required MSC-VO Times:
    double mSumMTimeFeatExtract;
    double mSumMTimeEptsLineOpt;
    double mSumTimePoseEstim;
    double mTimeCoarseManh;

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mInitialFrame;

    bool mbIniFirst;

    cv::Mat mImGray;

    cv::Mat mask;

    // Initialization Variables (Monocular)
    ////////////////////////////////////////////////////////////////////////////////////////
    // Points
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;

    ///////////////////////////////////////////////////////////////////////////////////////
    // Lines
    std::vector<int> mvIniLastLineMatches;
    vector<int> mvIniLineMatches;
    vector<cv::Point3f> mvLineS3D; 
    vector<cv::Point3f> mvLineE3D;  

    // Manhattan frames
    Manhattan* mpManh;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    cv::Mat Rotation_cm;
    cv::Mat mLastRcm;
    cv::Mat mRotation_wc;

    ///用ma估计的上一帧到当前帧的旋转粗略值
    cv::Mat coarseRcl;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();
    void CreateInitialMapMonoWithLine();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalLines();
    void FindSimilarLines(std::vector<std::vector<int>> &sim_lines_idx);
    
    // Not used, modify it for the back-end and check results
    void JoinLines(const std::vector<std::vector<int>> &sim_lines_idx);
    float PointToLineDist(const cv::Mat &sp1, const cv::Mat &ep1, const cv::Mat &sp2, const cv::Mat &ep2);
    int FindRepresentDesc(std::vector<int> v_idxs);
    int ComputeRepresentMapLine(const std::vector<int> &v_idxs);

    void UpdateLocalKeyFrames();

    bool TrackLocalMapWithLines();
    void SearchLocalPoints();
    void SearchLocalLines();
    ///modify by wh
    void SearchLocalPlanes();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    bool ExtractCoarseManhAx();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    // Line
    LINEextractor* mpLSDextractorLeft;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    std::vector<MapLine*> mvpLocalMapLines;
    std::vector<MapLine*> mvpLocalMapLines_InFrustum;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    // Monocular sensor
    Mat mUndistX, mUndistY;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;   
    int mnLineMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    // Manhattan parameters
    bool mManhInit;
    bool mCoarseManhInit;
    ///MODIFY BY WH
    float mfDThRef;
    float mfAThRef;

    float mfVerTh;
    float mfParTh;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
