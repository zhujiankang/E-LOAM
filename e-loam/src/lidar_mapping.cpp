#include <ros/ros.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <Eigen/Dense>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include <pclomp/ndt_omp.h>

using namespace std;
using namespace gtsam;

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    double q_x;
    double q_y;
    double q_z;
    double q_w;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (double, q_x, q_x) (double, q_y, q_y) (double, q_z, q_z) (double, q_w, q_w)
                                   (double, time, time))
typedef PointXYZIRPYT  PointTypePose;
typedef pcl::PointXYZI PointType;

NonlinearFactorGraph gtSAMgraph;
Values initialEstimate;
Values optimizedEstimate;
ISAM2 *isam;
Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odometryNoise;
noiseModel::Diagonal::shared_ptr constraintNoise;

nav_msgs::Odometry odomAftMapped;

vector<pcl::PointCloud<PointType>::Ptr> NDTCloudKeyFrames;
// deque<pcl::PointCloud<PointType>::Ptr> recentNDTCloudFrames;
vector<int> surroundingExistingKeyPoseID;
deque<pcl::PointCloud<PointType>::Ptr> surroundingNDTCloudFrames;
deque<pcl::PointCloud<PointType>::Ptr> recentNDTCloudKeyFrames;
pcl::PointCloud<PointType>::Ptr laserCloudNDTLast;

pcl::PointCloud<PointType>::Ptr laserCloudNDTFromMap;
pcl::PointCloud<PointType>::Ptr laserCloudNDTFromMapDS;

pcl::PointCloud<PointType>::Ptr nearHistoryNDTKeyFrameCloud;
pcl::PointCloud<PointType>::Ptr nearHistoryNDTKeyFrameCloudDS;

pcl::PointCloud<PointType>::Ptr latestNDTKeyFrameCloud;

//PointTypePose的XYZI保存和cloudKeyPoses3D一样的内容，另外还保存RPY角度以及一个时间值timeLaserOdometry
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;
pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames;
pcl::VoxelGrid<PointType> downSizeFilterNDT;

pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;

PointType previousRobotPosPoint;
PointType currentRobotPosPoint;

ros::Publisher pubLaserCloudSurround;
ros::Publisher pubOdomAftMappedHighFrec;
ros::Publisher pubLaserPath;

nav_msgs::Path laserPath;
//初始化正态分布变换（NDT）
pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>::Ptr registration;



vector<int> pointSearchInd;
vector<float> pointSearchSqDis;

vector<int> surroundingExistingKeyPosesID;

double timeLaserCloudNDTLast;
double timeLaserOdometry;
double timeLastProcessing;

double odometry_weight = 0.0;

int laserCloudNDTFromMapDSNum;

Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

Eigen::Quaterniond q_wodom_curr_loam(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr_loam(0, 0, 0);

Eigen::Quaterniond q_wodom_curr_ndt(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr_ndt(0, 0, 0);

Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

Eigen::Quaterniond q_w_curr_last(1, 0, 0, 0);
Eigen::Vector3d t_w_curr_last(0, 0, 0);

size_t  surroundingKeyframeSearchNum = 20;
int latestFrameID = 0;
int closestHistoryFrameID;
int latestFrameIDLoopCloure;

bool newLaserCloudNDTLast = false;
bool newLaserOdometry = false;
bool newLaserOdometry_ndt = false;
bool newLaserOdometry_loam = false;
bool loopClosureEnableFlag = true;
bool potentialLoopFlag = false;
bool aLoopIsClosed = false;

std::mutex mtx;

void ndtCloudCallback(const sensor_msgs::PointCloud2ConstPtr& pointMsg) {
    timeLaserCloudNDTLast = pointMsg->header.stamp.toSec();
    laserCloudNDTLast->clear();
    pcl::fromROSMsg(*pointMsg, *laserCloudNDTLast);
    newLaserCloudNDTLast = true;
}

void laserOdometryNDTCallback(const nav_msgs::Odometry::ConstPtr& laserOdometryMSG) {
    // timeLaserOdometry = laserOdometryMSG->header.stamp.toSec();
    q_wodom_curr_ndt.x() = laserOdometryMSG->pose.pose.orientation.x;
    q_wodom_curr_ndt.y() = laserOdometryMSG->pose.pose.orientation.y;
    q_wodom_curr_ndt.z() = laserOdometryMSG->pose.pose.orientation.z;
    q_wodom_curr_ndt.w() = laserOdometryMSG->pose.pose.orientation.w;
    t_wodom_curr_ndt.x() = laserOdometryMSG->pose.pose.position.x;
    t_wodom_curr_ndt.y() = laserOdometryMSG->pose.pose.position.y;
    t_wodom_curr_ndt.z() = laserOdometryMSG->pose.pose.position.z;

    odometry_weight = laserOdometryMSG->twist.twist.linear.x;

    newLaserOdometry_ndt = true;
}

void laserOdometryLoamCallback(const nav_msgs::Odometry::ConstPtr& laserOdometryMSG) {
    timeLaserOdometry = laserOdometryMSG->header.stamp.toSec();
    q_wodom_curr_loam.x() = laserOdometryMSG->pose.pose.orientation.x;
    q_wodom_curr_loam.y() = laserOdometryMSG->pose.pose.orientation.y;
    q_wodom_curr_loam.z() = laserOdometryMSG->pose.pose.orientation.z;
    q_wodom_curr_loam.w() = laserOdometryMSG->pose.pose.orientation.w;
    t_wodom_curr_loam.x() = laserOdometryMSG->pose.pose.position.x;
    t_wodom_curr_loam.y() = laserOdometryMSG->pose.pose.position.y;
    t_wodom_curr_loam.z() = laserOdometryMSG->pose.pose.position.z;

    // high frequence publish

	// Eigen::Quaterniond q_w_curr_tmp = q_wmap_wodom * q_wodom_curr;
	// Eigen::Vector3d t_w_curr_tmp = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	// nav_msgs::Odometry odomAftMapped;
	// odomAftMapped.header.frame_id = "/map";
	// odomAftMapped.child_frame_id = "/aft_mapped";
	// odomAftMapped.header.stamp = laserOdometryMSG->header.stamp;
	// odomAftMapped.pose.pose.orientation.x = q_w_curr_tmp.x();
	// odomAftMapped.pose.pose.orientation.y = q_w_curr_tmp.y();
	// odomAftMapped.pose.pose.orientation.z = q_w_curr_tmp.z();
	// odomAftMapped.pose.pose.orientation.w = q_w_curr_tmp.w();
	// odomAftMapped.pose.pose.position.x = t_w_curr_tmp.x();
	// odomAftMapped.pose.pose.position.y = t_w_curr_tmp.y();
	// odomAftMapped.pose.pose.position.z = t_w_curr_tmp.z();
	// pubOdomAftMappedHighFrec.publish(odomAftMapped);

    // geometry_msgs::PoseStamped laserPose;
    // laserPose.header = odomAftMapped.header;
    // laserPose.pose = odomAftMapped.pose.pose;
    // laserPath.header.stamp = odomAftMapped.header.stamp;
    // laserPath.poses.push_back(laserPose);
    // laserPath.header.frame_id = "/map";
    // pubLaserPath.publish(laserPath);

    newLaserOdometry_loam = true;
}

// set initial guess
void transformAssociateToMap()
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate()
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

Pose3 pclPointTogtsamPose3(PointTypePose thisPoint){
    return Pose3(Rot3::Quaternion(thisPoint.q_w, thisPoint.q_x, thisPoint.q_y, thisPoint.q_z),
                        Point3(double(thisPoint.x),   double(thisPoint.y),    double(thisPoint.z)));
}

Eigen::Affine3f pclPointToAffine3fLidar(PointTypePose thisPoint){

    Eigen::Quaterniond tmp_q(thisPoint.q_w, thisPoint.q_x, thisPoint.q_y, thisPoint.q_z);
    Eigen::Vector3d tmp_t(thisPoint.x, thisPoint.y, thisPoint.z);
    Eigen::Isometry3d tmp_T = Eigen::Isometry3d::Identity();
    tmp_T.rotate(tmp_q);
    tmp_T.pretranslate(tmp_t);

    return Eigen::Affine3f(tmp_T.matrix().cast<float>());
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn){
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType point_To;

    int cloudSize = cloudIn->points.size();
    cloudOut->resize(cloudSize);

    Eigen::Quaterniond tmp_q(transformIn->q_w, transformIn->q_x, transformIn->q_y, transformIn->q_z);
    Eigen::Vector3d tmp_t(transformIn->x, transformIn->y, transformIn->z);
    for (int i = 0; i < cloudSize; ++i){

        Eigen::Vector3d pointFrom(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
        Eigen::Vector3d pointTo =  tmp_q * pointFrom + tmp_t;

        point_To.x = pointTo.x();
        point_To.y = pointTo.y();
        point_To.z = pointTo.z();
        point_To.intensity = cloudIn->points[i].intensity;

        cloudOut->points[i] = point_To;
    }
    return cloudOut;
}

void extractSurroundingKeyFrames(){

    if (cloudKeyPoses3D->points.empty() == true)
        return;	

    // loopClosureEnableFlag 这个变量另外只在loopthread这部分中有用到
    if (loopClosureEnableFlag == true){

        // recentCornerCloudKeyFrames保存的点云数量太少，则清空后重新塞入新的点云直至数量够
        if (recentNDTCloudKeyFrames.size() < surroundingKeyframeSearchNum){
            recentNDTCloudKeyFrames.clear();
            int numPoses = cloudKeyPoses3D->points.size();
            for (int i = numPoses-1; i >= 0; --i){
                // cloudKeyPoses3D的intensity中存的是索引值?
                // 保存的索引值从1开始编号；
                int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                // 进行坐标变换
                recentNDTCloudKeyFrames.push_front(transformPointCloud(NDTCloudKeyFrames[thisKeyInd], &thisTransformation));
                if (recentNDTCloudKeyFrames.size() >= surroundingKeyframeSearchNum)
                    break;
            }

        }else{
            // recentCornerCloudKeyFrames中点云保存的数量较多
            // pop队列最前端的一个，再push后面一个
            if (latestFrameID != cloudKeyPoses3D->points.size() - 1){
                recentNDTCloudKeyFrames.pop_front();
                
                latestFrameID = cloudKeyPoses3D->points.size() - 1;
                PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];

                recentNDTCloudKeyFrames.push_back(transformPointCloud(NDTCloudKeyFrames[latestFrameID], &thisTransformation));
            }
        }

        for (int i = 0; i < recentNDTCloudKeyFrames.size(); ++i){
            
            *laserCloudNDTFromMap += *recentNDTCloudKeyFrames[i];
        }

    } else{ 
        surroundingKeyPoses->clear();
        surroundingKeyPosesDS->clear();

        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);

        // PointType currentRobotPosPoint;
        // currentRobotPosPoint.x = t_w_curr.x();
        // currentRobotPosPoint.y = t_w_curr.y();
        // currentRobotPosPoint.z = t_w_curr.z();


        kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, 6.0, pointSearchInd, pointSearchSqDis, 0);
        for (size_t i = 0; i < pointSearchInd.size(); ++i)
            surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        size_t numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
        for (size_t i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
            bool existingFlag = false;
            for (size_t j = 0; j < numSurroundingPosesDS; ++j){
                // 双重循环，不断对比surroundingExistingKeyPosesID[i]和surroundingKeyPosesDS的点的index
                // 如果能够找到一样的，说明存在相同的关键点(因为surroundingKeyPosesDS从cloudKeyPoses3D中筛选而来)
                if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity){
                    existingFlag = true;
                    break;
                }
            }
            
            if (existingFlag == false){
                // 如果surroundingExistingKeyPosesID[i]对比了一轮的已经存在的关键位姿的索引后（intensity保存的就是size()）
                // 没有找到相同的关键点，那么把这个点从当前队列中删除
                // 否则的话，existingFlag为true，该关键点就将它留在队列中
                surroundingExistingKeyPosesID.   erase(surroundingExistingKeyPosesID.   begin() + i);
                surroundingNDTCloudFrames.       erase(surroundingNDTCloudFrames.       begin() + i);
                --i;
            }
        }

        // 上一个两重for循环主要用于删除数据，此处的两重for循环用来添加数据
        for (size_t i = 0; i < numSurroundingPosesDS; ++i) {
            bool existingFlag = false;
            for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter){
                // *iter就是不同的cloudKeyPoses3D->points.size(),
                // 把surroundingExistingKeyPosesID内没有对应的点放进一个队列里
                // 这个队列专门存放周围存在的关键帧，但是和surroundingExistingKeyPosesID的点没有对应的，也就是新的点
                if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity){
                    existingFlag = true;
                    break;
                }
            }
            if (existingFlag == true){
                continue;
            }else{
                int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];

                surroundingExistingKeyPosesID.   push_back(thisKeyInd);
                surroundingNDTCloudFrames.       push_back(transformPointCloud(NDTCloudKeyFrames[thisKeyInd], &thisTransformation));
            }
        }

        for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
            *laserCloudNDTFromMap += *surroundingNDTCloudFrames[i];
        }

        
    }

    downSizeFilterNDT.setInputCloud(laserCloudNDTFromMap);
    downSizeFilterNDT.filter(*laserCloudNDTFromMapDS);
    laserCloudNDTFromMapDSNum = laserCloudNDTFromMapDS->points.size();

}

void ndtRegistration() {

    if(laserCloudNDTFromMapDS->points.empty()) 
        return;
    registration->setInputTarget(laserCloudNDTFromMapDS);
    registration->setInputSource(laserCloudNDTLast);

    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4d init_guess;
    init_guess.setIdentity();
    init_guess.block<3,3>(0,0) = q_w_curr.toRotationMatrix();
    init_guess.topRightCorner(3,1) = t_w_curr;

    // cout << "init guess" << init_guess << endl;
    // auto t1 = ros::WallTime::now();
    registration->align(*aligned, init_guess.cast<float>());
    // auto t2 = ros::WallTime::now();

    // cout << "ndt past " << registration->getFinalTransformation() << endl;
    // std::cout << "Normal Distributions Transform has converged:" << registration->hasConverged()
            // << " score: " << registration->getFitnessScore() << std::endl;
    Eigen::Quaternionf tmp_q(registration->getFinalTransformation().topLeftCorner<3, 3>());
    Eigen::Vector3f tmp_t(registration->getFinalTransformation().topRightCorner<3, 1>());
    q_w_curr = tmp_q.cast<double>();
    t_w_curr = tmp_t.cast<double>();

    transformUpdate();
}

void saveKeyFramesAndFactor(){

    currentRobotPosPoint.x = t_w_curr.x();
    currentRobotPosPoint.y = t_w_curr.y();
    currentRobotPosPoint.z = t_w_curr.z();

    bool saveThisKeyFrame = true;
    if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
            +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
            +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3){
        saveThisKeyFrame = false;
    }

    if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        return;

    previousRobotPosPoint = currentRobotPosPoint;

    if (cloudKeyPoses3D->points.empty()){
        gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::Quaternion(q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z()),
                                                   Point3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z())), priorNoise));
        // initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
        initialEstimate.insert(0, Pose3(Rot3::Quaternion(q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z()),
                                                    Point3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z())));
        q_w_curr_last = q_w_curr;
        t_w_curr_last = t_w_curr;
    }
    else{
        gtsam::Pose3 poseFrom = Pose3(Rot3::Quaternion(q_w_curr_last.w(), q_w_curr_last.x(), q_w_curr_last.y(), q_w_curr_last.z()),
                                                   Point3(t_w_curr_last.x(), t_w_curr_last.y(), t_w_curr_last.z()));
        gtsam::Pose3 poseTo   = Pose3(Rot3::Quaternion(q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z()),
                                                   Point3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z()));
        
        // 构造函数原型:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
        gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->points.size(), poseTo);
    }

    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    // Compute an estimate from the incomplete linear delta computed during the last update.
    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

    // thisPose3D.x = t_w_curr.x();
    // thisPose3D.y = t_w_curr.y();
    // thisPose3D.z = t_w_curr.z();

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->points.size();
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.q_x = latestEstimate.rotation().toQuaternion().x();
    thisPose6D.q_y = latestEstimate.rotation().toQuaternion().y();
    thisPose6D.q_z = latestEstimate.rotation().toQuaternion().z();
    thisPose6D.q_w = latestEstimate.rotation().toQuaternion().w();
    thisPose6D.time = timeLaserOdometry;
    cloudKeyPoses6D->push_back(thisPose6D);

    if (cloudKeyPoses3D->points.size() > 1){
        q_w_curr.x() = latestEstimate.rotation().toQuaternion().x();
        q_w_curr.y() = latestEstimate.rotation().toQuaternion().y();
        q_w_curr.z() = latestEstimate.rotation().toQuaternion().z();
        q_w_curr.w() = latestEstimate.rotation().toQuaternion().w();
        t_w_curr.x() = latestEstimate.translation().x();
        t_w_curr.y() = latestEstimate.translation().y();
        t_w_curr.z() = latestEstimate.translation().z();

        q_w_curr_last = q_w_curr;
        t_w_curr_last = t_w_curr;

        // for (int i = 0; i < 6; ++i){
        //     transformLast[i] = transformAftMapped[i];
        //     transformTobeMapped[i] = transformAftMapped[i];
        // }
    }

    pcl::PointCloud<PointType>::Ptr thisNDTKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudNDTLast, *thisNDTKeyFrame);

    NDTCloudKeyFrames.push_back(thisNDTKeyFrame);

    // transformUpdate();

}

void correctPoses(){
    if (aLoopIsClosed == true){
        recentNDTCloudKeyFrames.clear();

        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;

            cloudKeyPoses6D->points[i].q_x = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().x();
            cloudKeyPoses6D->points[i].q_y = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().y();
            cloudKeyPoses6D->points[i].q_z = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().z();
            cloudKeyPoses6D->points[i].q_w = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().w();
        }
        transformUpdate();
        aLoopIsClosed = false;
    }
}

void clearCloud() {
    laserCloudNDTFromMap->clear();
    laserCloudNDTFromMapDS->clear();

    nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/map";
	odomAftMapped.child_frame_id = "/aft_mapped";
    odomAftMapped.header.stamp = ros::Time(timeLaserOdometry);
	// odomAftMapped.header.stamp = timeLaserOdometry;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);

    
    static tf::TransformBroadcaster pose_broadcaster;

    tf::StampedTransform odom_trans;
    odom_trans.stamp_ = odomAftMapped.header.stamp;
    odom_trans.frame_id_ = "/map";
    odom_trans.child_frame_id_ = "/base_link";


    odom_trans.setRotation(tf::Quaternion(odomAftMapped.pose.pose.orientation.x, odomAftMapped.pose.pose.orientation.y, odomAftMapped.pose.pose.orientation.z, odomAftMapped.pose.pose.orientation.w));
    odom_trans.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z));

    pose_broadcaster.sendTransform(odom_trans);

    geometry_msgs::PoseStamped laserPose;
    laserPose.header = odomAftMapped.header;
    laserPose.pose = odomAftMapped.pose.pose;
    laserPath.header.stamp = odomAftMapped.header.stamp;
    laserPath.poses.push_back(laserPose);
    laserPath.header.frame_id = "/map";
    pubLaserPath.publish(laserPath);

}

bool detectLoopClosure(){

    
    latestNDTKeyFrameCloud->clear();
    nearHistoryNDTKeyFrameCloud->clear();
    nearHistoryNDTKeyFrameCloudDS->clear();

    // 资源分配时初始化
    // 在互斥量被析构前不解锁
    std::lock_guard<std::mutex> lock(mtx);

    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, 15.0, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    
    closestHistoryFrameID = -1;
    for (int i = 0; i < pointSearchIndLoop.size(); ++i){
        int id = pointSearchIndLoop[i];
        // 两个时间差值大于30秒
        if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0){
            closestHistoryFrameID = id;
            break;
        }
    }
    if (closestHistoryFrameID == -1){
        // 找到的点和当前时间上没有超过30秒的
        return false;
    }
    
    latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
    *latestNDTKeyFrameCloud += *transformPointCloud(NDTCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

    pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
    int cloudSize = latestNDTKeyFrameCloud->points.size();
    for (int i = 0; i < cloudSize; ++i){
        if ((int)latestNDTKeyFrameCloud->points[i].intensity >= 0){
            hahaCloud->push_back(latestNDTKeyFrameCloud->points[i]);
        }
    }
    latestNDTKeyFrameCloud->clear();
    *latestNDTKeyFrameCloud   = *hahaCloud;

    // historyKeyframeSearchNum在utility.h中定义为25，前后25个点进行变换
    for (int j = -25; j <= 25; ++j){
        if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
            continue;
        // 要求closestHistoryFrameID + j在0到cloudKeyPoses3D->points.size()-1之间,不能超过索引
        *nearHistoryNDTKeyFrameCloud += *transformPointCloud(NDTCloudKeyFrames[closestHistoryFrameID+j], &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
    }

    // 下采样滤波减少数据量
    downSizeFilterHistoryKeyFrames.setInputCloud(nearHistoryNDTKeyFrameCloud);
    downSizeFilterHistoryKeyFrames.filter(*nearHistoryNDTKeyFrameCloudDS);

    // if (pubHistoryKeyFrames.getNumSubscribers() != 0){
    //     sensor_msgs::PointCloud2 cloudMsgTemp;
    //     pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
    //     cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    //     cloudMsgTemp.header.frame_id = "/camera_init";
    //     pubHistoryKeyFrames.publish(cloudMsgTemp);
    // }

    return true;
}

void performLoopClosure(){

    if (cloudKeyPoses3D->points.empty() == true)
        return;
    if (potentialLoopFlag == false){
        if (detectLoopClosure() == true){
            potentialLoopFlag = true;
            // timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
        }

        if (potentialLoopFlag == false)
            return;
    }

    cout << "========================================================" << endl;
    potentialLoopFlag = false;
    // TODO:后面可以改成NDT
    // pcl::IterativeClosestPoint<PointType, PointType> icp;
    // icp.setMaxCorrespondenceDistance(100);
    // icp.setMaximumIterations(100);
    // icp.setTransformationEpsilon(1e-6);
    // icp.setEuclideanFitnessEpsilon(1e-6);
    // // 设置RANSAC运行次数
    // icp.setRANSACIterations(0);

    // icp.setInputSource(latestNDTKeyFrameCloud);
    // // 使用detectLoopClosure()函数中下采样刚刚更新nearHistorySurfKeyFrameCloudDS
    // icp.setInputTarget(nearHistoryNDTKeyFrameCloudDS);
    // pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    // // 进行icp点云对齐
    // icp.align(*unused_result);

    // // 为什么匹配分数高直接返回???分数高代表噪声太多
    // if (icp.hasConverged() == false || icp.getFitnessScore() > 0.3)
    //     return; 

/*     pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt_loop(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());
    ndt_loop->setMaxCorrespondenceDistance(100);
    ndt_loop->setMaximumIterations(100);
    ndt_loop->setTransformationEpsilon(1e-6);
    ndt_loop->setEuclideanFitnessEpsilon(1e-6);
    ndt_loop->setRANSACIterations(0);


    ndt_loop->setResolution(5);
    ndt_loop->setStepSize(0.5);
    ndt_loop->setNeighborhoodSearchMethod(pclomp::DIRECT1);
    ndt_loop->setNumThreads(4);
    ndt_loop->setInputCloud(latestNDTKeyFrameCloud);
    ndt_loop->setInputTarget(nearHistoryNDTKeyFrameCloudDS);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    ndt_loop->align(*unused_result);

    std::cout << "fit score: " << ndt_loop->getFitnessScore() << std::endl;
    if(ndt_loop->hasConverged() == false || ndt_loop->getFitnessScore() > 0.5)
        return; */

     pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(100);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
// Align clouds
    icp.setInputSource(latestNDTKeyFrameCloud);
    icp.setInputTarget(nearHistoryNDTKeyFrameCloudDS);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);


    std::cout << "fit score: " << icp.getFitnessScore() << std::endl;
    if (icp.hasConverged() == false || icp.getFitnessScore() > 0.5)
        return;






    // // 以下在点云icp收敛并且噪声量在一定范围内进行
    // if (pubIcpKeyFrames.getNumSubscribers() != 0){
    //     pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
    //     // icp.getFinalTransformation()的返回值是Eigen::Matrix<Scalar, 4, 4>
    //     pcl::transformPointCloud (*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
    //     sensor_msgs::PointCloud2 cloudMsgTemp;
    //     pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
    //     cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    //     cloudMsgTemp.header.frame_id = "/camera_init";
    //     pubIcpKeyFrames.publish(cloudMsgTemp);
    // }   



    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionFrame;
    correctionFrame = icp.getFinalTransformation();
    // correctionFrame = ndt_loop->getFinalTransformation();
    cout << "loop: " << correctionFrame.matrix() << endl;
    // 得到平移和旋转的角度
    pcl::getTranslationAndEulerAngles(correctionFrame, x, y, z, roll, pitch, yaw);
    // Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(x, y, z, roll, pitch, yaw);
    Eigen::Affine3f tWrong = pclPointToAffine3fLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

    Eigen::Affine3f tCorrect = correctionFrame * tWrong;

    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);


    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    // float noiseScore = ndt_loop->getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // cout << "posefrom" << poseFrom.matrix() << endl;
    // cout << "poseto" << poseTo.matrix() << endl;

    std::lock_guard<std::mutex> lock(mtx);
    gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));
    isam->update(gtSAMgraph);
    isam->update();
    gtSAMgraph.resize(0);

    aLoopIsClosed = true;
}

void run() {

    while(ros::ok()) {

        if(newLaserCloudNDTLast && newLaserOdometry) {
            newLaserCloudNDTLast = false; newLaserOdometry = false;
            std::lock_guard<std::mutex> lock(mtx);

            // if(timeLaserCloudNDTLast - timeLastProcessing >= 0.3) {
                timeLastProcessing = timeLaserCloudNDTLast;
                
                auto t1 = ros::WallTime::now();
    
                transformAssociateToMap();


                extractSurroundingKeyFrames();

                // downsampleCurrentScan();

                ndtRegistration();

                saveKeyFramesAndFactor();

                // publishTF();

                // publishKeyPosesAndFrames();

                clearCloud();

                correctPoses();

                auto t2 = ros::WallTime::now();
                // cout << "lidar mapping: " << (t2 - t1).toSec() * 1000 << "  ms" << endl;
            // }

        }
                
    }
}

void loopClosureThread() {
    if(loopClosureEnableFlag == false) 
        return;
    
    // ros::Rate loopRate(1);

    while(ros::ok()) {
        // loopRate.sleep();
        performLoopClosure();
        cout << "sss" << endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

}

void publishGlobalMap(){

    if (pubLaserCloudSurround.getNumSubscribers() == 0)
        return;

    if (cloudKeyPoses3D->points.empty() == true)
        return;
    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, 500.0, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    // extract visualized and downsampled key frames
    for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i){
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        *globalMapKeyFrames += *transformPointCloud(NDTCloudKeyFrames[thisKeyInd],   &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

    sensor_msgs::PointCloud2 cloudMsgTemp;
    // TODO:做了暂时的修改
    pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
    cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    cloudMsgTemp.header.frame_id = "/map";
    pubLaserCloudSurround.publish(cloudMsgTemp);  

    globalMapKeyPoses->clear();
    globalMapKeyPosesDS->clear();
    globalMapKeyFrames->clear();
    globalMapKeyFramesDS->clear();     
}


void visualizeGlobalMapThread(){
    ros::Rate rate(0.2);
    while (ros::ok()){
        publishGlobalMap();
        rate.sleep();
    }

    // Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    // transform.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(1,0,0)));
    // pcl::transformPointCloud(*globalMapKeyFramesDS, *globalMapKeyFramesDS, transform);

    // Eigen::Affine3f transform1 = Eigen::Affine3f::Identity();
    // transform1.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(0,0,1)));
    // pcl::transformPointCloud(*globalMapKeyFramesDS, *globalMapKeyFramesDS, transform1);
    // // save final point cloud
    // pcl::io::savePCDFileASCII(fileDirectory+"finalCloud.pcd", *globalMapKeyFramesDS);

    // string cornerMapString = "cornerMap.pcd";
    // string surfaceMapString = "surfaceMap.pcd";
    // string trajectoryString = "/tmp/trajectory.pcd";

    // pcl::PointCloud<PointType>::Ptr cornerMapCloud(new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr cornerMapCloudDS(new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr surfaceMapCloud(new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr surfaceMapCloudDS(new pcl::PointCloud<PointType>());
    
    // for(int i = 0; i < cornerCloudKeyFrames.size(); i++) {
    // *cornerMapCloud  += *transformPointCloud(cornerCloudKeyFrames[i],   &cloudKeyPoses6D->points[i]);
    // *surfaceMapCloud += *transformPointCloud(surfCloudKeyFrames[i],     &cloudKeyPoses6D->points[i]);
    // *surfaceMapCloud += *transformPointCloud(outlierCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
    // }

    // downSizeFilterCorner.setInputCloud(cornerMapCloud);
    // downSizeFilterCorner.filter(*cornerMapCloudDS);
    // downSizeFilterSurf.setInputCloud(surfaceMapCloud);
    // downSizeFilterSurf.filter(*surfaceMapCloudDS);



    // Eigen::Affine3f transform2 = Eigen::Affine3f::Identity();
    // transform2.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(1,0,0)));
    // pcl::transformPointCloud(*cornerMapCloud, *cornerMapCloud, transform2);

    // Eigen::Affine3f transform3 = Eigen::Affine3f::Identity();
    // transform3.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(0,0,1)));
    // pcl::transformPointCloud(*cornerMapCloud, *cornerMapCloud, transform3);


    // Eigen::Affine3f transform4 = Eigen::Affine3f::Identity();
    // transform4.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(1,0,0)));
    // pcl::transformPointCloud(*surfaceMapCloud, *surfaceMapCloud, transform4);

    // Eigen::Affine3f transform5 = Eigen::Affine3f::Identity();
    // transform5.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(0,0,1)));
    // pcl::transformPointCloud(*surfaceMapCloud, *surfaceMapCloud, transform5);


    // pcl::io::savePCDFileASCII(fileDirectory+"cornerMap.pcd", *cornerMapCloud);
    // pcl::io::savePCDFileASCII(fileDirectory+"surfaceMap.pcd", *surfaceMapCloud);
    // // pcl::io::savePCDFileASCII(fileDirectory+"trajectory.pcd", *cloudKeyPoses3D);
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "test_mapping_node");
    ros::NodeHandle node = ros::NodeHandle();

     // 用于闭环图优化的参数设置，使用gtsam库
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);


    ros::Subscriber subNDTCloud = node.subscribe("/laser_cloud_NDT", 100, &ndtCloudCallback);
    ros::Subscriber subLaserOdometry_Loam = node.subscribe("/laser_odom_loam", 100, &laserOdometryLoamCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber subLaserOdometry_ndt = node.subscribe("/laser_odom_ndt", 100, &laserOdometryNDTCallback, ros::TransportHints().tcpNoDelay());

    pubOdomAftMappedHighFrec = node.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 10);
    pubLaserCloudSurround = node.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
    // pubLaserOdometry = node.advertise<nav_msgs::Odometry>("/laser_after_map_to_init", 10);
    pubLaserPath = node.advertise<nav_msgs::Path>("/laser_after_map_path", 10);

    laserCloudNDTLast.reset(new pcl::PointCloud<PointType>());
    latestNDTKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    laserCloudNDTFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudNDTFromMapDS.reset(new pcl::PointCloud<PointType>());
    nearHistoryNDTKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    nearHistoryNDTKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
    surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());

    kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
    globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
    globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
    globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
    globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

    downSizeFilterNDT.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilterHistoryKeyFrames.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilterSurroundingKeyPoses.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4); // for global map visualization

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
    priorNoise = noiseModel::Diagonal::Variances(Vector6);
    odometryNoise = noiseModel::Diagonal::Variances(Vector6);


    std::thread looprun(&run);
    std::thread loopthread(&loopClosureThread);
    std::thread visualizeMapThread(&visualizeGlobalMapThread);
    

    pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());
    ndt->setMaxCorrespondenceDistance(50);
    ndt->setTransformationEpsilon(0.0001);
    ndt->setEuclideanFitnessEpsilon(0.0001);
    ndt->setResolution(2.0);
    ndt->setStepSize(0.05);
    ndt->setMaximumIterations(100);
    ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
    ndt->setNumThreads(4);
    registration = ndt;

    ros::Rate rate(100);

    
    while(ros::ok()) {

        // run();
        ros::spinOnce();

        if(newLaserOdometry_loam)
        {
            Eigen::Quaterniond q_tmp(1, 0, 0, 0);
            Eigen::Vector3d t_tmp(0, 0, 0);

            // if(odometry_weight > 0.5) {
                q_tmp = q_wodom_curr_loam;
                t_tmp = t_wodom_curr_loam;
            // } 
            // else
            // {
            //     double w = (0.5 - odometry_weight) * 0.4;

            //     q_tmp.x() = w * q_wodom_curr_ndt.x() + (1 - w) * q_wodom_curr_loam.x();
            //     q_tmp.y() = w * q_wodom_curr_ndt.y() + (1 - w) * q_wodom_curr_loam.y();
            //     q_tmp.z() = w * q_wodom_curr_ndt.z() + (1 - w) * q_wodom_curr_loam.z();
            //     q_tmp.w() = w * q_wodom_curr_ndt.w() + (1 - w) * q_wodom_curr_loam.w();
            //     q_tmp.normalize();

            //     t_tmp = w * t_wodom_curr_ndt + (1 - w) * t_wodom_curr_loam;
            // }

            t_wodom_curr = t_wodom_curr + q_wodom_curr * t_tmp;
            q_wodom_curr = q_wodom_curr * q_tmp;
            
            // run();


            // high frequence publish

            // Eigen::Quaterniond q_w_curr_tmp = q_wmap_wodom * q_wodom_curr;
            // Eigen::Vector3d t_w_curr_tmp = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

            // nav_msgs::Odometry odomAftMapped;
            // odomAftMapped.header.frame_id = "/map";
            // odomAftMapped.child_frame_id = "/aft_mapped";
            // odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            // odomAftMapped.pose.pose.orientation.x = q_w_curr_tmp.x();
            // odomAftMapped.pose.pose.orientation.y = q_w_curr_tmp.y();
            // odomAftMapped.pose.pose.orientation.z = q_w_curr_tmp.z();
            // odomAftMapped.pose.pose.orientation.w = q_w_curr_tmp.w();
            // odomAftMapped.pose.pose.position.x = t_w_curr_tmp.x();
            // odomAftMapped.pose.pose.position.y = t_w_curr_tmp.y();
            // odomAftMapped.pose.pose.position.z = t_w_curr_tmp.z();
            // pubOdomAftMappedHighFrec.publish(odomAftMapped);

            // geometry_msgs::PoseStamped laserPose;
            // laserPose.header = odomAftMapped.header;
            // laserPose.pose = odomAftMapped.pose.pose;
            // laserPath.header.stamp = odomAftMapped.header.stamp;
            // laserPath.poses.push_back(laserPose);
            // laserPath.header.frame_id = "/map";
            // pubLaserPath.publish(laserPath);

            newLaserOdometry_loam = false;
            newLaserOdometry_ndt = false;
            newLaserOdometry = true;
        }
        

        rate.sleep();
    }
    looprun.join();
    loopthread.join();
    visualizeMapThread.join();

    return 0;
}

