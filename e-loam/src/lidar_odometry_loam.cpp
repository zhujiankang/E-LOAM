#include <queue>
#include <mutex>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h> 

#include<time.h>
#include "e_loam/lidarFactor.hpp"

int corner_correspondence = 0, plane_correspondence = 0, texture_correspondence = 0;
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeTextureLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<pcl::PointXYZI>::Ptr EdgePointsSharp(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr EdgePointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr PlanarPointsFlat(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr PlanarPointsLessFlat(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr TexturePointsSharp(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr TexturePointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>());

std::queue<sensor_msgs::PointCloud2ConstPtr> EdgeSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> EdgeLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> TextureSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> TextureLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> PlanarFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> PlanarLessFlatBuf;

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudEdgeLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudPlanarLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTextureLast(new pcl::PointCloud<pcl::PointXYZI>());


// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

double timeEdgePointsSharp = 0;
double timeEdgePointsLessSharp = 0;
double timeTexturePointsSharp = 0;
double timeTexturePointsLessSharp = 0;
double timePlanarPointsFlat = 0;
double timePlanarPointsLessFlat = 0;

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserPath;

nav_msgs::Path laserPath;

bool systemInited = false;

std::mutex mBuf;


// undistort lidar point
void TransformToStart(pcl::PointXYZI const *const pi, pcl::PointXYZI *const po)
{
    //interpolation ratio
    double s = 1.0;

    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

void subEdgeSharpCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    mBuf.lock();
    EdgeSharpBuf.push(point_msgs);
    mBuf.unlock();
}

void subEdgeLessSharpCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    mBuf.lock();
    EdgeLessSharpBuf.push(point_msgs);
    mBuf.unlock();
}

void subTextureSharpCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    mBuf.lock();
    TextureSharpBuf.push(point_msgs);
    mBuf.unlock();
}

void subTextureLessSharpCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    mBuf.lock();
    TextureLessSharpBuf.push(point_msgs);
    mBuf.unlock();
}

void subPlanarFlatCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    mBuf.lock();
    PlanarFlatBuf.push(point_msgs);
    mBuf.unlock();
}

void subPlanarLessFlatCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    mBuf.lock();
    PlanarLessFlatBuf.push(point_msgs);
    mBuf.unlock();
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "test_odometry_loam_node");
    ros::NodeHandle node = ros::NodeHandle();

    ros::Subscriber subEdgePointsSharp = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, &subEdgeSharpCallback);

    ros::Subscriber subEdgePointsLessSharp = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, &subEdgeLessSharpCallback);

    ros::Subscriber subTexturePointsSharp = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_texture_sharp", 100, &subTextureSharpCallback);

    ros::Subscriber subTexturePointsLessSharp = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_texture_less_sharp", 100, &subTextureLessSharpCallback);

    ros::Subscriber subPlanarPointsFlat = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, &subPlanarFlatCallback);

    ros::Subscriber subPlanarPointsLessFlat = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, &subPlanarLessFlatCallback);

    pubLaserOdometry = node.advertise<nav_msgs::Odometry>("/laser_odom_loam", 10);

    pubLaserPath = node.advertise<nav_msgs::Path>("/laser_odom_loam_path", 10);

    ros::Rate rate(100);

    while(ros::ok())
    {
        ros::spinOnce();

        if(!EdgeLessSharpBuf.empty() && !EdgeSharpBuf.empty() &&
           !TextureLessSharpBuf.empty() && !TextureSharpBuf.empty() &&
           !PlanarLessFlatBuf.empty() && !PlanarFlatBuf.empty())
        {
            timeEdgePointsSharp = EdgeSharpBuf.front()->header.stamp.toSec();
            timeEdgePointsLessSharp = EdgeLessSharpBuf.front()->header.stamp.toSec();
            timeTexturePointsSharp = TextureSharpBuf.front()->header.stamp.toSec();
            timeTexturePointsLessSharp = TextureLessSharpBuf.front()->header.stamp.toSec();
            timePlanarPointsFlat = PlanarFlatBuf.front()->header.stamp.toSec();
            timePlanarPointsLessFlat = PlanarLessFlatBuf.front()->header.stamp.toSec();

            if(timeEdgePointsSharp != timeEdgePointsLessSharp || timeTexturePointsSharp != timeEdgePointsLessSharp ||
               timeTexturePointsLessSharp != timeEdgePointsLessSharp || timePlanarPointsFlat != timeEdgePointsLessSharp ||
               timePlanarPointsLessFlat != timeEdgePointsLessSharp)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            EdgePointsSharp->clear();
            pcl::fromROSMsg(*EdgeSharpBuf.front(), *EdgePointsSharp);
            EdgeSharpBuf.pop();

            EdgePointsLessSharp->clear();
            pcl::fromROSMsg(*EdgeLessSharpBuf.front(), *EdgePointsLessSharp);
            EdgeLessSharpBuf.pop();

            TexturePointsSharp->clear();
            pcl::fromROSMsg(*TextureSharpBuf.front(), *TexturePointsSharp);
            TextureSharpBuf.pop();

            TexturePointsLessSharp->clear();
            pcl::fromROSMsg(*TextureLessSharpBuf.front(), *TexturePointsLessSharp);
            TextureLessSharpBuf.pop();

            PlanarPointsFlat->clear();
            pcl::fromROSMsg(*PlanarFlatBuf.front(), *PlanarPointsFlat);
            PlanarFlatBuf.pop();

            PlanarPointsLessFlat->clear();
            pcl::fromROSMsg(*PlanarLessFlatBuf.front(), *PlanarPointsLessFlat);
            PlanarLessFlatBuf.pop();
            mBuf.unlock();
            
            auto t1 = ros::WallTime::now();
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else
            {   
                int cornerPointsSharpNum = EdgePointsSharp->points.size();
                int surfPointsFlatNum = PlanarPointsFlat->points.size();
                int texturePointsSharpNum = TexturePointsSharp->points.size();

                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;
                    texture_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);

                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    // find correspondence for corner features
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {   
                        TransformToStart(&(EdgePointsSharp->points[i]), &pointSel);
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;
                        if (pointSearchSqDis[0] < 25)
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudEdgeLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = 25;
                            // search in the direction of increasing scan line
                            for (size_t j = closestPointInd + 1; j < laserCloudEdgeLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudEdgeLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudEdgeLast->points[j].intensity) > (closestPointScanID + 2.5))
                                    break;

                                double pointSqDis = (laserCloudEdgeLast->points[j].x - pointSel.x) *
                                                    (laserCloudEdgeLast->points[j].x - pointSel.x) +
                                                    (laserCloudEdgeLast->points[j].y - pointSel.y) *
                                                    (laserCloudEdgeLast->points[j].y - pointSel.y) +
                                                    (laserCloudEdgeLast->points[j].z - pointSel.z) *
                                                    (laserCloudEdgeLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudEdgeLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudEdgeLast->points[j].intensity) < (closestPointScanID - 2.5))
                                    break;

                                double pointSqDis = (laserCloudEdgeLast->points[j].x - pointSel.x) *
                                                    (laserCloudEdgeLast->points[j].x - pointSel.x) +
                                                    (laserCloudEdgeLast->points[j].y - pointSel.y) *
                                                    (laserCloudEdgeLast->points[j].y - pointSel.y) +
                                                    (laserCloudEdgeLast->points[j].z - pointSel.z) *
                                                    (laserCloudEdgeLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            Eigen::Vector3d curr_point(EdgePointsSharp->points[i].x,
                                                       EdgePointsSharp->points[i].y,
                                                       EdgePointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudEdgeLast->points[closestPointInd].x,
                                                        laserCloudEdgeLast->points[closestPointInd].y,
                                                        laserCloudEdgeLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudEdgeLast->points[minPointInd2].x,
                                                        laserCloudEdgeLast->points[minPointInd2].y,
                                                        laserCloudEdgeLast->points[minPointInd2].z);

                            double s = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }
                    }

                    // find correspondence for corner features
                    for (int i = 0; i < texturePointsSharpNum; ++i)
                    {   
                        TransformToStart(&(TexturePointsSharp->points[i]), &pointSel);
                        kdtreeTextureLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;
                        if (pointSearchSqDis[0] < 25)
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudTextureLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = 25;
                            // search in the direction of increasing scan line
                            for (size_t j = closestPointInd + 1; j < laserCloudTextureLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudTextureLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudTextureLast->points[j].intensity) > (closestPointScanID + 2.5))
                                    break;

                                double pointSqDis = (laserCloudTextureLast->points[j].x - pointSel.x) *
                                                    (laserCloudTextureLast->points[j].x - pointSel.x) +
                                                    (laserCloudTextureLast->points[j].y - pointSel.y) *
                                                    (laserCloudTextureLast->points[j].y - pointSel.y) +
                                                    (laserCloudTextureLast->points[j].z - pointSel.z) *
                                                    (laserCloudTextureLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudTextureLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudTextureLast->points[j].intensity) < (closestPointScanID - 2.5))
                                    break;

                                double pointSqDis = (laserCloudTextureLast->points[j].x - pointSel.x) *
                                                    (laserCloudTextureLast->points[j].x - pointSel.x) +
                                                    (laserCloudTextureLast->points[j].y - pointSel.y) *
                                                    (laserCloudTextureLast->points[j].y - pointSel.y) +
                                                    (laserCloudTextureLast->points[j].z - pointSel.z) *
                                                    (laserCloudTextureLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            Eigen::Vector3d curr_point(TexturePointsSharp->points[i].x,
                                                        TexturePointsSharp->points[i].y,
                                                        TexturePointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudTextureLast->points[closestPointInd].x,
                                                        laserCloudTextureLast->points[closestPointInd].y,
                                                        laserCloudTextureLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudTextureLast->points[minPointInd2].x,
                                                        laserCloudTextureLast->points[minPointInd2].y,
                                                        laserCloudTextureLast->points[minPointInd2].z);

                            double s = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }
                    }

                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {   
                        TransformToStart(&(PlanarPointsFlat->points[i]), &pointSel);
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < 25)
                        {
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            int closestPointScanID = int(laserCloudPlanarLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = 25, minPointSqDis3 = 25;

                            // search in the direction of increasing scan line
                            for (size_t j = closestPointInd + 1; j < laserCloudPlanarLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudPlanarLast->points[j].intensity) > (closestPointScanID + 2.5))
                                    break;

                                double pointSqDis = (laserCloudPlanarLast->points[j].x - pointSel.x) *
                                                    (laserCloudPlanarLast->points[j].x - pointSel.x) +
                                                    (laserCloudPlanarLast->points[j].y - pointSel.y) *
                                                    (laserCloudPlanarLast->points[j].y - pointSel.y) +
                                                    (laserCloudPlanarLast->points[j].z - pointSel.z) *
                                                    (laserCloudPlanarLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                if (int(laserCloudPlanarLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                else if (int(laserCloudPlanarLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudPlanarLast->points[j].intensity) < (closestPointScanID - 2.5))
                                    break;

                                double pointSqDis = (laserCloudPlanarLast->points[j].x - pointSel.x) *
                                                    (laserCloudPlanarLast->points[j].x - pointSel.x) +
                                                    (laserCloudPlanarLast->points[j].y - pointSel.y) *
                                                    (laserCloudPlanarLast->points[j].y - pointSel.y) +
                                                    (laserCloudPlanarLast->points[j].z - pointSel.z) *
                                                    (laserCloudPlanarLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudPlanarLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudPlanarLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(PlanarPointsFlat->points[i].x,
                                                        PlanarPointsFlat->points[i].y,
                                                        PlanarPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudPlanarLast->points[closestPointInd].x,
                                                            laserCloudPlanarLast->points[closestPointInd].y,
                                                            laserCloudPlanarLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudPlanarLast->points[minPointInd2].x,
                                                            laserCloudPlanarLast->points[minPointInd2].y,
                                                            laserCloudPlanarLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudPlanarLast->points[minPointInd3].x,
                                                            laserCloudPlanarLast->points[minPointInd3].y,
                                                            laserCloudPlanarLast->points[minPointInd3].z);

                                double s = 1.0;
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }

                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                } 

                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;


            }
            auto t2 = ros::WallTime::now();

            std::cout << "lidar odometry_loam: " << (t2 - t1).toSec() * 1000 << "  ms" << std::endl;

            pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = EdgePointsLessSharp;
            EdgePointsLessSharp = laserCloudEdgeLast;
            laserCloudEdgeLast = laserCloudTemp;

            laserCloudTemp = TexturePointsLessSharp;
            TexturePointsLessSharp = laserCloudTextureLast;
            laserCloudTextureLast = laserCloudTemp;

            laserCloudTemp = PlanarPointsLessFlat;
            PlanarPointsLessFlat = laserCloudPlanarLast;
            laserCloudPlanarLast = laserCloudTemp;

            kdtreeCornerLast->setInputCloud(laserCloudEdgeLast);
            kdtreeSurfLast->setInputCloud(laserCloudPlanarLast);
            kdtreeTextureLast->setInputCloud(laserCloudTextureLast);

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/odom";
            laserOdometry.child_frame_id = "/velodyne";
            laserOdometry.header.stamp = ros::Time::now();
            laserOdometry.pose.pose.orientation.x = q_last_curr.x();
            laserOdometry.pose.pose.orientation.y = q_last_curr.y();
            laserOdometry.pose.pose.orientation.z = q_last_curr.z();
            laserOdometry.pose.pose.orientation.w = q_last_curr.w();
            laserOdometry.pose.pose.position.x = t_last_curr.x();
            laserOdometry.pose.pose.position.y = t_last_curr.y();
            laserOdometry.pose.pose.position.z = t_last_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPose.pose.position.x = t_w_curr.x();
            laserPose.pose.position.y = t_w_curr.y();
            laserPose.pose.position.z = t_w_curr.z();
            laserPose.pose.orientation.x = q_w_curr.x();
            laserPose.pose.orientation.y = q_w_curr.y();
            laserPose.pose.orientation.z = q_w_curr.z();
            laserPose.pose.orientation.w = q_w_curr.w();
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/map";
            pubLaserPath.publish(laserPath);

        }

        rate.sleep();
    }

    return 0;
}

