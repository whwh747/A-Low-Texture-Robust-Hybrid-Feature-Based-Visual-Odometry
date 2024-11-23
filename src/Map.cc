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

#include "Map.h"

#include<mutex>

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    for(set<MapLine*>::iterator sit=mspMapLines.begin(), send=mspMapLines.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspMapLines.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpReferenceMapLines.clear();
    mvpKeyFrameOrigins.clear();
}

    void Map::AddMapLine(MapLine *pML)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.insert(pML);
    }

    void Map::EraseMapLine(MapLine *pML)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.erase(pML);
    }

    void Map::SetWorldManhAxis(cv::Mat worldManhAxis)
    {
        mWorldManhAxis = worldManhAxis;
    }

    ///如果有互相垂直的两个平面 则使用它们的法向量作为ma 否则寻找与之垂直的线段作为补充
    cv::Mat Map::FindManhattan(Frame &pF, const float &verTh, bool out) {
        cv::Mat bestP1, bestP2;
        float lverTh = verTh;
        int maxSize = 0;

        if(out)
            //cout << "Matching planes..." << endl;

            ///先根据两个互相垂直的平面  算出一个初始的方向  相机坐标系下
            for (int i = 0; i < pF.mnPlaneNum; ++i) {
                cv::Mat p1 = pF.mvPlaneCoefficients[i];
                if(out)
                    //cout << " plane  " << i << ": " << endl;

                    if(out)
                        //cout << " p1  " << p1.t() << ": " << endl;

                        for (int j = i+1;j < pF.mnPlaneNum; ++j) {
                            cv::Mat p2 = pF.mvPlaneCoefficients[j];

                            float angle = p1.at<float>(0) * p2.at<float>(0) +
                                          p1.at<float>(1) * p2.at<float>(1) +
                                          p1.at<float>(2) * p2.at<float>(2);

                            if(out)
                                //cout << j << ", p2 : " << p2.t() << endl;

                                if(out)
                                    //cout << j << ", angle : " << angle << endl;

                                    // vertical planes
                                    if (angle < lverTh && angle > -lverTh && (pF.mvPlanePoints[i].size() + pF.mvPlanePoints[j].size()) > maxSize) {
                                        if(out)
                                            //cout << "  vertical!" << endl;
                                            maxSize = pF.mvPlanePoints[i].size() + pF.mvPlanePoints[j].size();

                                        if (bestP1.empty() || bestP2.empty()) {
                                            bestP1 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                                            bestP2 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                                        }

                                        bestP1.at<float>(0, 0) = p1.at<float>(0, 0);
                                        bestP1.at<float>(1, 0) = p1.at<float>(1, 0);
                                        bestP1.at<float>(2, 0) = p1.at<float>(2, 0);

                                        bestP2.at<float>(0, 0) = p2.at<float>(0, 0);
                                        bestP2.at<float>(1, 0) = p2.at<float>(1, 0);
                                        bestP2.at<float>(2, 0) = p2.at<float>(2, 0);
                                    }
                        }
            }

//        cout<<"best p1 = "<<bestP1<<endl;
//        cout<<"best p2 = "<<bestP2<<endl;

        ///没有互相垂直的两个平面
        if (bestP1.empty() || bestP2.empty()) {
            if(out)
                cout << "Matching planes and lines..." << endl;

            for (int i = 0; i < pF.mnPlaneNum; ++i) {
                ///计算的是世界坐标系下的平面参数
                cv::Mat p = pF.ComputePlaneWorldCoeff(i);
                cout<<"p = "<<p<<endl;
                if(out)
                    cout << " plane  " << i << ": " << endl;

                for (int j = 0; j < pF.mvLines3D.size(); ++j) {
                    ///相机坐标系下的三维线的两个端点  转换到世界坐标系下
                    Eigen::Vector3d st_3D_w = pF.PtToWorldCoord(pF.mvLines3D[i].first);
                    Eigen::Vector3d e_3D_w = pF.PtToWorldCoord(pF.mvLines3D[i].second);
                    Vector6d lineVector;
                    lineVector << st_3D_w.x(), st_3D_w.y(), st_3D_w.z(),
                            e_3D_w.x(), e_3D_w.y(), e_3D_w.z();

                    //cout<<"???"<<endl;
                    cv::Mat startPoint = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                    cv::Mat endPoint = cv::Mat::eye(cv::Size(1, 3), CV_32F);

                    startPoint.at<float>(0, 0) = lineVector[0];
                    startPoint.at<float>(1, 0) = lineVector[1];
                    startPoint.at<float>(2, 0) = lineVector[2];
                    endPoint.at<float>(0, 0) = lineVector[3];
                    endPoint.at<float>(1, 0) = lineVector[4];
                    endPoint.at<float>(2, 0) = lineVector[5];

                    ///方向向量
                    cv::Mat line = startPoint - endPoint;
                    line /= cv::norm(line);

                    if(out)
                        cout << "line: " << line << endl;

                    float angle = p.at<float>(0, 0) * line.at<float>(0, 0) +
                                  p.at<float>(1, 0) * line.at<float>(1, 0) +
                                  p.at<float>(2, 0) * line.at<float>(2, 0);

                    if(out)
                        cout << j << ", angle : " << angle << endl;

                    if (angle < lverTh && angle > -lverTh) {
                        if(out)
                            cout << "  vertical!" << endl;
                        lverTh = abs(angle);

                        if (bestP1.empty() || bestP2.empty()) {
                            bestP1 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                            bestP2 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                        }

                        bestP1.at<float>(0, 0) = p.at<float>(0, 0);
                        bestP1.at<float>(1, 0) = p.at<float>(1, 0);
                        bestP1.at<float>(2, 0) = p.at<float>(2, 0);

                        bestP2.at<float>(0, 0) = line.at<float>(0, 0);
                        bestP2.at<float>(1, 0) = line.at<float>(1, 0);
                        bestP2.at<float>(2, 0) = line.at<float>(2, 0);
                    }
                }
            }
        }

        if(out)
            cout << "Matching done" << endl;

        cv::Mat Rotation_cm;
        Rotation_cm = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        if (!bestP1.empty() && !bestP2.empty()) {

            int loc1;
            float max1 = 0;
            ///找第一个方向中绝对值最大的下标
            for (int i = 0; i < 3; i++) {
                float val = bestP1.at<float>(i);
                if (val < 0)
                    val = -val;
                if (val > max1) {
                    loc1 = i;
                    max1 = val;
                }
            }

            ///保证最大的为正
            if (bestP1.at<float>(loc1) < 0) {
                bestP1 = -bestP1;
            }

            int loc2;
            float max2 = 0;
            for (int i = 0; i < 3; i++) {
                float val = bestP2.at<float>(i);
                if (val < 0)
                    val = -val;
                if (val > max2) {
                    loc2 = i;
                    max2 = val;
                }
            }

            if (bestP2.at<float>(loc2) < 0) {
                bestP2 = -bestP2;
            }

            cv::Mat p3;

            p3 = bestP1.cross(bestP2);

            int loc3;
            float max3 = 0;
            for (int i = 0; i < 3; i++) {
                float val = p3.at<float>(i);
                if (val < 0)
                    val = -val;
                if (val > max3) {
                    loc3 = i;
                    max3 = val;
                }
            }

            if (p3.at<float>(loc3) < 0) {
                p3 = -p3;
            }

            ///上面保证三个方向都是正的
            if(out) {
//                cout << "p1: " << bestP1 << endl;
//                cout << "p2: " << bestP2 << endl;
//                cout << "p3: " << p3 << endl;
            }

            cv::Mat first, second, third;

            std::map<int, cv::Mat> sort;
            ///modify
            if(loc1 == loc2 || loc1 == loc3 || loc2 == loc3)
            {
                if(loc1 == loc2)
                {
                    sort[loc1] == bestP1;
                    sort[loc3] = p3;
                    if(loc1+loc3 == 1)sort[2] = bestP2;
                    else if(loc1 + loc3 == 2)sort[1] = bestP2;
                    else sort[0] = bestP2;
                }
                sort[loc1] = bestP1;
                sort[loc2] = bestP2;
                if(loc1 + loc2 == 1)sort[2] = p3;
                else if(loc1 + loc2 == 2)sort[1] = p3;
                else sort[0] = p3;
            }

            sort[loc1] = bestP1;
            sort[loc2] = bestP2;
            sort[loc3] = p3;

            first = sort[0];
            second = sort[1];
            third = sort[2];

//            cout<<"first = "<<first<<endl;
//            cout<<"second = "<<second<<endl;
//            cout<<"third = "<<third<<endl;

            // todo: refine this part
            Rotation_cm.at<float>(0, 0) = first.at<float>(0, 0);
            Rotation_cm.at<float>(1, 0) = first.at<float>(1, 0);
            Rotation_cm.at<float>(2, 0) = first.at<float>(2, 0);
            Rotation_cm.at<float>(0, 1) = second.at<float>(0, 0);
            Rotation_cm.at<float>(1, 1) = second.at<float>(1, 0);
            Rotation_cm.at<float>(2, 1) = second.at<float>(2, 0);
            Rotation_cm.at<float>(0, 2) = third.at<float>(0, 0);
            Rotation_cm.at<float>(1, 2) = third.at<float>(1, 0);
            Rotation_cm.at<float>(2, 2) = third.at<float>(2, 0);

            cv::Mat U, W, VT;

            cv::SVD::compute(Rotation_cm, W, U, VT);

            Rotation_cm = U * VT;
        }

        return Rotation_cm;
    }

    void Map::SetReferenceMapLines(const std::vector<MapLine *> &vpMLs)
    {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapLines = vpMLs;
    }

    vector<MapLine*> Map::GetAllMapLines()
    {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapLine*>(mspMapLines.begin(), mspMapLines.end());
    }


    cv::Mat Map::GetWorldManhAxis()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mWorldManhAxis;
    }

    vector<MapLine*> Map::GetReferenceMapLines()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapLines;
    }

    long unsigned int Map::MapLinesInMap()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapLines.size();
    }
    void Map::EraseMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.erase(pMP);
    }
    void Map::AddMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.insert(pMP);
    }
    vector<MapPlane *> Map::GetAllMapPlanes() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPlane *>(mspMapPlanes.begin(), mspMapPlanes.end());
    }
    ////跟踪丢失后这个函数会出问题   ComputePlaneWorldCoeff
    void Map::FlagMatchedPlanePoints(ORB_SLAM2::Frame &pF, const float &dTh) {

        unique_lock<mutex> lock(mMutexMap);
        int nMatches = 0;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);

            if (pF.mvpMapPlanes[i]) {
                for (auto mapPoint : mspMapPoints) {
                    cv::Mat pW = mapPoint->GetWorldPos();

                    double dis = abs(pM.at<float>(0, 0) * pW.at<float>(0, 0) +
                                     pM.at<float>(1, 0) * pW.at<float>(1, 0) +
                                     pM.at<float>(2, 0) * pW.at<float>(2, 0) +
                                     pM.at<float>(3, 0));

                    if (dis < 0.5) {
                        mapPoint->SetAssociatedWithPlaneFlag(true);
                        nMatches++;
                    }
                }
            }
        }
    }

    double Map::PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr boundry, bool out) {
        double res = 100;
        if (out)
            cout << " compute dis: " << endl;
        for (auto p : boundry->points) {
            double dis = abs(plane.at<float>(0, 0) * p.x +
                             plane.at<float>(1, 0) * p.y +
                             plane.at<float>(2, 0) * p.z +
                             plane.at<float>(3, 0));
            if (dis < res)
                res = dis;
        }
        if (out)
            cout << endl << "ave : " << res << endl;
        return res;
    }

} //namespace ORB_SLAM
