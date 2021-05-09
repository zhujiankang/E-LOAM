// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <cmath>
#include <vector>
#include <string>
// #include "aloam_velodyne/common.h"
// #include "aloam_velodyne/tic_toc.h"
// #include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using namespace std;

typedef pcl::PointXYZI PointType;

struct PointXYZIRT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,  
  (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
  (uint16_t, ring, ring) (float, time, time)
)

using std::atan2;
using std::cos;
using std::sin;

// const double scanPeriod = 0.1;

const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;
float cloudCurvature[400000];
float cloudIntensityDiff[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp_corner (int i,int j) { return (cloudCurvature[i] < cloudCurvature[j]); }
bool comp_intensity (int i,int j) { return (cloudIntensityDiff[i] < cloudIntensityDiff[j]); }

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreePlanarCloud(new pcl::KdTreeFLANN<pcl::PointXYZI>());

ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubTexturePointSharp;
ros::Publisher pubTexturePointLessSharp;
ros::Publisher PubNDTPoints;

double MINIMUM_RANGE = 0.1; 

template <typename PointT>
void RemoveNaNFromPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
                              pcl::PointCloud<PointT> &cloud_out,
                              std::vector<int> &index)
{
  // If the clouds are not the same, prepare the output
  if (&cloud_in != &cloud_out)
  {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize (cloud_in.points.size ());
  }
  // Reserve enough space for the indices
  index.resize (cloud_in.points.size ());
  size_t j = 0;

  // If the data is dense, we don't need to check for NaN
  if (cloud_in.is_dense)
  {
    // Simply copy the data
    cloud_out = cloud_in;
    for (j = 0; j < cloud_out.points.size (); ++j)
      index[j] = static_cast<int>(j);
  }
  else
  {
    for (size_t i = 0; i < cloud_in.points.size (); ++i)
    {
      if (!pcl_isfinite (cloud_in.points[i].x) || 
          !pcl_isfinite (cloud_in.points[i].y) || 
          !pcl_isfinite (cloud_in.points[i].z) ||
          !pcl_isfinite (cloud_in.points[i].intensity) ||
          !pcl_isfinite (cloud_in.points[i].ring))
        continue;
      cloud_out.points[j] = cloud_in.points[i];
      index[j] = static_cast<int>(i);
      j++;
    }
    if (j != cloud_in.points.size ())
    {
      // Resize to the correct size
      cloud_out.points.resize (j);
      index.resize (j);
    }

    cloud_out.height = 1;
    cloud_out.width  = static_cast<uint32_t>(j);

    // Removing bad points => dense (note: 'dense' doesn't mean 'organized')
    cloud_out.is_dense = true;
  }
}

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z > 10000)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited)
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    auto t1 = ros::WallTime::now();

    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::PointCloud<PointXYZIRT> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

    std::vector<int> indices;
    RemoveNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    if(laserCloudIn.is_dense == false) 
    {
        cout << "Point cloud is not in dense format, please remove NaN points first!" << endl;
        return;
    }

    static int ringFlag = 0;
    if(ringFlag == 0) 
    {
        ringFlag = -1;
        for (int i = 0; i < (int)laserCloudMsg->fields.size(); ++i)
        {
        if (laserCloudMsg->fields[i].name == "ring")
        {
            ringFlag = 1;
            break;
        }
        }
        if (ringFlag == -1)
        {
        cout << "Point cloud ring channel not available, please configure your point cloud data!" << endl;
        return;
        }
    }
    

    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);
    int cloudSize = laserCloudIn.points.size();
    // 对intensity进行滤波
    for(int i=4; i<(cloudSize-4); i++) {
        laserCloudIn.points[i].intensity =  (laserCloudIn.points[i-4].intensity + laserCloudIn.points[i-3].intensity + laserCloudIn.points[i-2].intensity + 
                                            laserCloudIn.points[i-1].intensity + laserCloudIn.points[i].intensity + laserCloudIn.points[i+1].intensity +
                                            laserCloudIn.points[i+2].intensity + laserCloudIn.points[i+3].intensity + laserCloudIn.points[i+4].intensity) / 9;
    }

    int scanID = 0;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        point.intensity = laserCloudIn.points[i].intensity;

        scanID = laserCloudIn.points[i].ring;
        point.intensity += scanID;
        laserCloudScans[scanID].push_back(point); 

    }

    // int cloudSize = laserCloudIn.points.size();
    // float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    // float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
    //                       laserCloudIn.points[cloudSize - 1].x) +
    //                2 * M_PI;

    // if (endOri - startOri > 3 * M_PI)
    // {
    //     endOri -= 2 * M_PI;
    // }
    // else if (endOri - startOri < M_PI)
    // {
    //     endOri += 2 * M_PI;
    // }
    // //printf("end Ori %f\n", endOri);

    // bool halfPassed = false;
    // int count = cloudSize;
    // PointType point;
    // std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // for (int i = 0; i < cloudSize; i++)
    // {
    //     point.x = laserCloudIn.points[i].x;
    //     point.y = laserCloudIn.points[i].y;
    //     point.z = laserCloudIn.points[i].z;
    //     point.intensity = laserCloudIn.points[i].intensity / 255.0;

    //     float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
    //     int scanID = 0;

    //     if (N_SCANS == 16)
    //     {
    //         scanID = int((angle + 15) / 2 + 0.5);
    //         if (scanID > (N_SCANS - 1) || scanID < 0)
    //         {
    //             count--;
    //             continue;
    //         }
    //     }
    //     else if (N_SCANS == 32)
    //     {
    //         scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
    //         if (scanID > (N_SCANS - 1) || scanID < 0)
    //         {
    //             count--;
    //             continue;
    //         }
    //     }
    //     else if (N_SCANS == 64)
    //     {   
    //         if (angle >= -8.83)
    //             scanID = int((2 - angle) * 3.0 + 0.5);
    //         else
    //             scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

    //         // use [0 50]  > 50 remove outlies 
    //         if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
    //         {
    //             count--;
    //             continue;
    //         }
    //     }
    //     else
    //     {
    //         printf("wrong scan number\n");
    //         ROS_BREAK();
    //     }
    //     //printf("angle %f scanID %d \n", angle, scanID);

    //     float ori = -atan2(point.y, point.x);
    //     if (!halfPassed)
    //     { 
    //         if (ori < startOri - M_PI / 2)
    //         {
    //             ori += 2 * M_PI;
    //         }
    //         else if (ori > startOri + M_PI * 3 / 2)
    //         {
    //             ori -= 2 * M_PI;
    //         }

    //         if (ori - startOri > M_PI)
    //         {
    //             halfPassed = true;
    //         }
    //     }
    //     else
    //     {
    //         ori += 2 * M_PI;
    //         if (ori < endOri - M_PI * 3 / 2)
    //         {
    //             ori += 2 * M_PI;
    //         }
    //         else if (ori > endOri + M_PI / 2)
    //         {
    //             ori -= 2 * M_PI;
    //         }
    //     }

    //     float relTime = (ori - startOri) / (endOri - startOri);
    //     point.intensity += scanID;
    //     laserCloudScans[scanID].push_back(point); 
    // }
    
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    std::vector<int> laserScanSize(N_SCANS);
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5;
        laserScanSize[i] = laserCloudScans[i].points.size();
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
    }

    for (int i = 5; i < cloudSize - 5; i++)
    { 
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        float diffI = laserCloud->points[i - 5].intensity + laserCloud->points[i - 4].intensity + laserCloud->points[i - 3].intensity + laserCloud->points[i - 2].intensity + laserCloud->points[i - 1].intensity - 10 * laserCloud->points[i].intensity + laserCloud->points[i + 1].intensity + laserCloud->points[i + 2].intensity + laserCloud->points[i + 3].intensity + laserCloud->points[i + 4].intensity + laserCloud->points[i + 5].intensity;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudIntensityDiff[i] = diffI * diffI;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
    }


    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;
    pcl::PointCloud<PointType> texturePointSharp;
    pcl::PointCloud<PointType> texturePointLessSharp;
    pcl::PointCloud<PointType> NDTPoints;

    // float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp_intensity);
            int textureNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1 && 
                    cloudIntensityDiff[ind] >= 0.1)
                // if(cloudNeighborPicked[ind] == 0 &&
                //     cloudCurvature[ind] < 0.1)
                {
                    // 对数据进行处理
                    cloudNeighborPicked[ind] = 1; 
                    textureNum++;
                    if (textureNum <= 2)
                    {                        
                        cloudLabel[ind] = 1; 
                        texturePointSharp.push_back(laserCloud->points[ind]);
                        texturePointLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else if (textureNum <= 20)
                    {                        
                        cloudLabel[ind] = 2; 
                        texturePointLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }
                    // float diffI = 0.0;
                    // for (int l = 1; l <= 5; l++)
                    // {
                    //     diffI += laserCloud->points[ind + l].intensity - laserCloud->points[ind - l].intensity;

                    // }
                    // if (diffI * diffI > 0.1)
                    // {
                    
                    // }
                    for (int l = 1; l <= 5; l++)
                    {
                        if (cloudIntensityDiff[ind + l] < 0.01)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        if (cloudIntensityDiff[ind + l] < 0.01)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
 
                }
            }


            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp_corner);
            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k]; 

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {

                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 3;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= 20)
                    {                        
                        cloudLabel[ind] = 4; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; 

                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {

                    cloudLabel[ind] = -1;
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;

                    if (smallestPickedNum >= 4)
                    {                        
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;



                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            
            
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
    
        }

        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS; 

    }
    auto t2 = ros::WallTime::now();
    std::cout << "lidar feature extraction: " << (t2 - t1).toSec() * 1000 << "  ms" << std::endl;


    // NDTPoints += cornerPointsLessSharp;
    // NDTPoints += surfPointsLessFlat;

    kdtreePlanarCloud->setInputCloud(laserCloud);

    for(size_t i=0; i<cornerPointsSharp.points.size(); i++) 
    {
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreePlanarCloud->radiusSearch(cornerPointsSharp.points[i], 1.0, pointSearchIndLoop, pointSearchSqDisLoop, 20);

        for(size_t j=0; j<pointSearchIndLoop.size(); j++) 
        {
            if(cloudLabel[pointSearchIndLoop[j]] <= 0) 
                continue;

            NDTPoints.push_back(laserCloud->points[pointSearchIndLoop[j]]);
            cloudLabel[pointSearchIndLoop[j]] = -2;
        }
    }


    NDTPoints += surfPointsLessFlat;
    cout << NDTPoints.points.size() << endl;

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/map";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/map";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/map";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/map";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    sensor_msgs::PointCloud2 texturePointSharp2;
    pcl::toROSMsg(texturePointSharp, texturePointSharp2);
    texturePointSharp2.header.stamp = laserCloudMsg->header.stamp;
    texturePointSharp2.header.frame_id = "/map";
    pubTexturePointSharp.publish(texturePointSharp2);

    sensor_msgs::PointCloud2 texturePointLessSharp2;
    pcl::toROSMsg(texturePointLessSharp, texturePointLessSharp2);
    texturePointLessSharp2.header.stamp = laserCloudMsg->header.stamp;
    texturePointLessSharp2.header.frame_id = "/map";
    pubTexturePointLessSharp.publish(texturePointLessSharp2);

    sensor_msgs::PointCloud2 NDTPoints2;
    pcl::toROSMsg(NDTPoints, NDTPoints2);
    NDTPoints2.header.stamp = laserCloudMsg->header.stamp;
    NDTPoints2.header.frame_id = "/map";
    PubNDTPoints.publish(NDTPoints2);


}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 64);

    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/points_raw", 5, laserCloudHandler);
    // ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 100, laserCloudHandler);

    // pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 5);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 5);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 5);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 5);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 5);

    pubTexturePointSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_texture_sharp", 5);

    pubTexturePointLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_texture_less_sharp", 5);

    PubNDTPoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_NDT", 5);

    // pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 5);

    ros::spin();

    return 0;
}
