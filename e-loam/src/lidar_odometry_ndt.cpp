#include <queue>
#include <mutex>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h> 

#include<time.h>
#include <pclomp/ndt_omp.h>
#include "e_loam/lidarFactor.hpp"

pcl::PointCloud<pcl::PointXYZI>::Ptr NDTPoints(new pcl::PointCloud<pcl::PointXYZI>());
// pcl::PointCloud<pcl::PointXYZI>::Ptr NDTPointsLess(new pcl::PointCloud<pcl::PointXYZI>());

std::queue<sensor_msgs::PointCloud2ConstPtr> NDTBuf;
// std::queue<sensor_msgs::PointCloud2ConstPtr> NDTLessBuf;

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudNDTLast(new pcl::PointCloud<pcl::PointXYZI>());

pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());


// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

Eigen::Quaterniond q_last_curr_ndt(1, 0, 0, 0);
Eigen::Vector3d t_last_curr_ndt(0, 0, 0);

Eigen::Matrix4f init_guess;

double timeNDT = 0;
// double timeNDTLess = 0;
double score = 0;


ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserPath;
ros::Publisher pubNDTCloud;

nav_msgs::Path laserPath;

bool systemInited = false;

std::mutex mBuf;

void subNDTCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    mBuf.lock();
    NDTBuf.push(point_msgs);
    mBuf.unlock();
}

// void subNDTLessCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
// {
//     mBuf.lock();
//     NDTLessBuf.push(point_msgs);
//     mBuf.unlock();
// }

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "test_odometry_ndt_node");
    ros::NodeHandle node = ros::NodeHandle();

    // ndt->setTransformationEpsilon(1e-6);
    ndt->setResolution(4.0);
    ndt->setStepSize(0.05);
    ndt->setMaximumIterations(35);
    ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
    ndt->setNumThreads(4);


    ros::Subscriber pointcloudNDTSub = node.subscribe("/laser_cloud_NDT", 5, &subNDTCallback);
    // ros::Subscriber pointcloudNDTLessSub = node.subscribe("/NDTPointLess", 5, &subNDTLessCallback);

    pubNDTCloud = node.advertise<sensor_msgs::PointCloud2>("ndt_cloud1", 5);
    pubLaserOdometry = node.advertise<nav_msgs::Odometry>("/laser_odom_ndt", 5);

    pubLaserPath = node.advertise<nav_msgs::Path>("/laser_odom__ndt_path", 5);

    ros::Rate rate(100);

    while(ros::ok())
    {
        ros::spinOnce();

        if(!NDTBuf.empty())
        {
            timeNDT = NDTBuf.front()->header.stamp.toSec();
            // timeNDTLess = NDTLessBuf.front()->header.stamp.toSec();

            // if(timeNDTLess != timeNDT)
            // {
            //     printf("ndt odometry unsync messeage!");
            //     ROS_BREAK();
            // }

            mBuf.lock();
            NDTPoints->clear();
            pcl::fromROSMsg(*NDTBuf.front(), *NDTPoints);
            NDTBuf.pop();

            // NDTPointsLess->clear();
            // pcl::fromROSMsg(*NDTLessBuf.front(), *NDTPointsLess);
            // NDTLessBuf.pop();
            mBuf.unlock();


            auto t1 = ros::WallTime::now();

            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else
            {   
                ndt->setInputCloud(NDTPoints);
                ndt->setInputTarget(laserCloudNDTLast);
                pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
                ndt->align(*aligned);

                score = ndt->getFitnessScore();
                // std::cout << "ndt score : " << score << std::endl;
                Eigen::Quaternionf tmp_q(ndt->getFinalTransformation().topLeftCorner<3, 3>());
                Eigen::Vector3f tmp_t(ndt->getFinalTransformation().topRightCorner<3, 1>());
                q_last_curr_ndt = tmp_q.cast<double>();
                t_last_curr_ndt = tmp_t.cast<double>();
               
                t_w_curr = t_w_curr + q_w_curr * t_last_curr_ndt;
                q_w_curr = q_w_curr * q_last_curr_ndt;

                init_guess = ndt->getFinalTransformation();
            }
            auto t2 = ros::WallTime::now();

            std::cout << "lidar odometry_ndt: " << (t2 - t1).toSec() * 1000 << "  ms" << std::endl;

            pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = NDTPoints;
            NDTPoints = laserCloudNDTLast;
            laserCloudNDTLast = laserCloudTemp;

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/odom";
            laserOdometry.child_frame_id = "/velodyne";
            laserOdometry.header.stamp = ros::Time::now();
            laserOdometry.pose.pose.orientation.x = q_last_curr_ndt.x();
            laserOdometry.pose.pose.orientation.y = q_last_curr_ndt.y();
            laserOdometry.pose.pose.orientation.z = q_last_curr_ndt.z();
            laserOdometry.pose.pose.orientation.w = q_last_curr_ndt.w();
            laserOdometry.pose.pose.position.x = t_last_curr_ndt.x();
            laserOdometry.pose.pose.position.y = t_last_curr_ndt.y();
            laserOdometry.pose.pose.position.z = t_last_curr_ndt.z();
            laserOdometry.twist.twist.linear.x = score;
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose.position.x = t_w_curr.x();
            laserPose.pose.position.y = t_w_curr.y();
            laserPose.pose.position.z = t_w_curr.z();
            laserPose.pose.orientation.x = q_w_curr.x();
            laserPose.pose.orientation.y = q_w_curr.y();
            laserPose.pose.orientation.z = q_w_curr.z();
            laserPose.pose.orientation.w = q_w_curr.w();
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/map";
            pubLaserPath.publish(laserPath);

            sensor_msgs::PointCloud2 ndtcloud;
            pcl::toROSMsg(*NDTPoints, ndtcloud);
            ndtcloud.header.frame_id = "map";
            ndtcloud.header.stamp = ros::Time(timeNDT);
            pubNDTCloud.publish(ndtcloud);

            
        }

        rate.sleep();
    }
    return 0;
}

