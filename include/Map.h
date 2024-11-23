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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>

#include "MapLine.h"
#include "MapPlane.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class MapLine;
class MapPlane;
class Frame;

class Map
{
public:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud <PointT> PointCloud;
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void EraseKeyFrame(KeyFrame* pKF);
    void InformNewBigChange();
    int GetLastBigChangeIdx();
    //---MapPoint---
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    //---MapLine---
    void AddMapLine(MapLine* pML);
    void EraseMapLine(MapLine* pML);
    void SetReferenceMapLines(const std::vector<MapLine*> &vpMLs);

    // Manhattan Axis 
    void SetWorldManhAxis(cv::Mat worldManhAxis);

    cv::Mat FindManhattan(Frame &pF, const float &verTh, bool out = false);

    std::vector<KeyFrame*> GetAllKeyFrames();

    //---MapPoint---
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();
    long unsigned int MapPointsInMap();
    //---MapLine---
    std::vector<MapLine*> GetAllMapLines();
    std::vector<MapLine*> GetReferenceMapLines();
    long unsigned int MapLinesInMap();

    // ----Manhattan Axis ----
    cv::Mat GetWorldManhAxis();

    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    ///modify by wh
    ///plane
    void EraseMapPlane(MapPlane *pMP);
    std::set<MapPlane*> mspMapPlanes;
    void AddMapPlane(MapPlane* pMP);
    std::vector<MapPlane*> GetAllMapPlanes();
    void FlagMatchedPlanePoints(ORB_SLAM2::Frame &pF, const float &dTh);
    double PointDistanceFromPlane(const cv::Mat& plane, PointCloud::Ptr boundry, bool out = false);

    vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;
    std::mutex mMutexLineCreation;

protected:
    //---MapPoint---
    std::set<MapPoint*> mspMapPoints;
    //---MapLine---
    std::set<MapLine*> mspMapLines;

    std::set<KeyFrame*> mspKeyFrames;

    cv::Mat mWorldManhAxis;

    std::vector<MapPoint*> mvpReferenceMapPoints;
    std::vector<MapLine*> mvpReferenceMapLines;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
