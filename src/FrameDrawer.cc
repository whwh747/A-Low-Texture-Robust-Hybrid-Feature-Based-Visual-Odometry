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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state

    vector<KeyLine> vCurrentKeyLines; 
    vector<KeyLine> vIniKeyLines;
    vector<bool> vbLineVO, vbLineMap;

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
            vCurrentKeyLines = mvCurrentKeyLines;
            vIniKeyLines = mvIniKeyLines;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
            vCurrentKeyLines = mvCurrentKeyLines;
            vbLineVO = mvbLineVO;
            vbLineMap = mvbLineMap;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
            vCurrentKeyLines = mvCurrentKeyLines;
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) 
        cvtColor(im,im,CV_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                        cv::Scalar(0,255,0));
            }
        }        
    }
    else if(state==Tracking::OK) 
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }
    // Draw the colours of the lines associated to the Manhattan Axes
    if (vCurrentManhIdx.size() > 0)
    {
        std::vector<cv::line_descriptor::KeyLine> major_axis_lines;
        std::vector<cv::line_descriptor::KeyLine> medium_axis_lines;
        std::vector<cv::line_descriptor::KeyLine> minor_axis_lines;
        std::vector<cv::line_descriptor::KeyLine> other_axis_lines;
        std::vector<cv::line_descriptor::KeyLine> no_depth_lines;
        for (size_t i = 0; i < vCurrentManhIdx.size(); i++)
        {
            if (mvCurrentKeyLines[i].startPointX == 0)
                continue;
            if (vCurrentManhIdx[i] == -1)
            {
                no_depth_lines.push_back(mvCurrentKeyLines[i]);
            }
            else if (vCurrentManhIdx[i] == 0)
            {
                other_axis_lines.push_back(mvCurrentKeyLines[i]);
            }

            else if (vCurrentManhIdx[i] == 1)
            {
                major_axis_lines.push_back(mvCurrentKeyLines[i]);
            }
            else if (vCurrentManhIdx[i] == 2)
            {
                medium_axis_lines.push_back(mvCurrentKeyLines[i]);
            }

            else if (vCurrentManhIdx[i] == 3)
            {
                minor_axis_lines.push_back(mvCurrentKeyLines[i]);
            }
        }
        // Green - MA-X
        drawKeylinesCustom(im, major_axis_lines, im, Scalar(0, 100, 0));
        // Blue - MA-Y
        drawKeylinesCustom(im, medium_axis_lines, im, Scalar(100, 0, 0));
        // Red - MA-Z
        drawKeylinesCustom(im, minor_axis_lines, im, Scalar(0, 0, 255));
        // Orange - No depth extracted
        drawKeylinesCustom(im, no_depth_lines, im, Scalar(23, 127, 245));
        // Line non-associated to a MA  
        drawKeylinesCustom(im, other_axis_lines, im, Scalar(50, 205, 154));
    }
    else
    {
        drawKeylinesCustom(im, mvCurrentKeyLines, im, Scalar(200, 0, 0));
    }

    ///add surface normal
//    for(int i=0;i<vSurfaceNormal.size();i++)
//    {
//        SurfaceNormal surface = vSurfaceNormal[i];
//        if(isnan(surface.normal.x) || isnan(surface.normal.y) || isnan(surface.normal.z))continue;
//        Vector3d dir(surface.normal.x,surface.normal.y,surface.normal.z);
//
//        double value = 0.0;
//        int index  = -1;
//        for(int j=0;j<3;j++)
//        {
//            Vector3d vp = tmp_vps[j];
//            double x = abs( dir.dot(vp) );
//            if(x > value)
//            {
//                value = x;
//                index  = j;
//            }
//        }
//        if(index == 0)
//        {
//            cv::circle(im,surface.FramePosition,1,cv::Scalar(0,0,100),-1);
//        }
//        else if(index == 1)
//        {
//            cv::circle(im,surface.FramePosition,1,cv::Scalar(0,100,0),-1);
//        }
//        else if(index == 2)
//        {
//            cv::circle(im,surface.FramePosition,1,cv::Scalar(100,0,0),-1);
//        }
//    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

// Draw Manhattan axes in the image, extracted from git jstraub/RTMF
void FrameDrawer::projectDirections(cv::Mat& img, const cv::Mat& dirs,
    double f_d)
{
  double scale = 0.12;
  cv::Mat p0 = (cv::Mat_<double>(3,1)<<0.35,0.25,1.0);
  cv::Mat colors = (cv::Mat_<int>(3, 3) << 0, 255, 0,
                    0, 0, 255,
                    255, 0, 0);
  double u0 = p0.at<double>(0)/p0.at<double>(2)*f_d + 320.;
  double v0 = p0.at<double>(1)/p0.at<double>(2)*f_d + 240.;
  for(uint32_t k=0; k < dirs.cols; ++k)
  {
    cv::Mat p1 = p0 + (dirs.col(k)*scale);
    double u1 = p1.at<double>(0)/p1.at<double>(2)*f_d + 320.;
    double v1 = p1.at<double>(1)/p1.at<double>(2)*f_d + 240.;
    cv::line(img, cv::Point(u0,v0), cv::Point(u1,v1),
    CV_RGB(colors.at<int>(k,0),colors.at<int>(k,1),colors.at<int>(k,2)), 2, CV_AA);
    double arrowLen = 10.;
    double angle = atan2(v1-v0,u1-u0);

    double ru1 = u1 - arrowLen*cos(angle + M_PI*0.25);
    double rv1 = v1 - arrowLen*sin(angle + M_PI*0.25);
    cv::line(img, cv::Point(u1,v1), cv::Point(ru1,rv1),
    CV_RGB(colors.at<int>(k,0),colors.at<int>(k,1),colors.at<int>(k,2)), 2, CV_AA);
    ru1 = u1 - arrowLen*cos(angle - M_PI*0.25);
    rv1 = v1 - arrowLen*sin(angle - M_PI*0.25);
    cv::line(img, cv::Point(u1,v1), cv::Point(ru1,rv1),
    CV_RGB(colors.at<int>(k,0),colors.at<int>(k,1),colors.at<int>(k,2)), 2, CV_AA);
  }
  cv::circle(img, cv::Point(u0,v0), 2, CV_RGB(0,0,0), 2, CV_AA);
}


void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "MSC-VO |  ";
        else
            s << "MSC-VO LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::drawKeylinesCustom( const Mat& image, const std::vector<KeyLine>& keylines, Mat& outImage, const Scalar& color)
{
    outImage = image.clone();

  for ( size_t i = 0; i < keylines.size(); i++ )
  {
      //if(isStructLine[i]==false)continue;
    /* decide lines' color  */
    Scalar lineColor;
    if( color == Scalar::all( -1 ) )
    {
      int R = ( rand() % (int) ( 255 + 1 ) );
      int G = ( rand() % (int) ( 255 + 1 ) );
      int B = ( rand() % (int) ( 255 + 1 ) );

      lineColor = Scalar( R, G, B );
    }

    else
      lineColor = color;

    /* get line */
    KeyLine k = keylines[i];

    /* draw line */
    cv::line( outImage, Point2f( k.startPointX, k.startPointY ), Point2f( k.endPointX, k.endPointY ), lineColor, 2);
  }
}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    mvCurrentKeyLines = pTracker->mCurrentFrame.mvKeylinesUn;
    isStructLine = pTracker->mCurrentFrame.isStructLine;
    vCurrentManhIdx = pTracker->mCurrentFrame.vManhAxisIdx;   

    NL = mvCurrentKeyLines.size();  
    mvbLineVO = vector<bool>(NL, false);
    mvbLineMap = vector<bool>(NL, false);

    vSurfaceNormal = pTracker->mCurrentFrame.vSurfaceNormal;
    tmp_vps = pTracker->mCurrentFrame.tmp_vps;

    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
        mvIniKeyLines = pTracker->mInitialFrame.mvKeylinesUn;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;
                }
            }
        }

        //cout<<"NL = "<<NL<<endl;
        for(int i=0; i<NL; i++)
        {

            MapLine* pML = pTracker->mCurrentFrame.mvpMapLines[i];

            if(pML)
            {
                //cout<<"i = "<<i<<endl;
                //cout<<"size1 = "<<pTracker->mCurrentFrame.mvpMapLines.size()<<" size2 = "<<pTracker->mCurrentFrame.mvbLineOutlier.size()<<endl;
                if(!pTracker->mCurrentFrame.mvbLineOutlier[i])
                {
                    //cout<<"pML = "<<pML->mnId<<endl;
                    //cout<<"obs = "<<pML->Observations()<<endl;
                    if(pML->Observations()>0)
                        mvbLineMap[i] = true;
                    else
                        mvbLineVO[i] = true;
                }
            }
        }
    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

} //namespace ORB_SLAM
