#include <thread>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv) {
    // Load RGB and depth images
    cv::Mat rgbImage = cv::imread("../rgb_room2.png", cv::IMREAD_COLOR);
    cv::Mat depthImage = cv::imread("../depth_room2.tiff", cv::IMREAD_UNCHANGED);

    if (rgbImage.empty() || depthImage.empty()) {
        std::cerr << "Error loading images" << std::endl;
        return -1;
    }

    // Handling extreme depth values
    float maxReasonableDepth = 30.0f;
    depthImage.forEach<float>([&maxReasonableDepth](float &depth, const int position[]) -> void {
        if (depth > maxReasonableDepth || std::isinf(depth) || std::isnan(depth)) {
            depth = 0; // Setting to 0 to exclude from visualization
        }
    });

    // Camera parameters
    float fx = 1053.02, fy = 1052.72, cx = 894.2, cy = 560.436;

    // Create a Point Cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Generate Point Cloud from the depth and RGB images
    for (int i = 0; i < depthImage.rows; i++) {
        for (int j = 0; j < depthImage.cols; j++) {
            float depthValue = depthImage.at<float>(i, j);
            if (depthValue == 0) continue; // Skip invalid points

            pcl::PointXYZRGB point;
            point.z = depthValue;
            point.x = (j - cx) * depthValue / fx;
            point.y = (i - cy) * depthValue / fy;

            cv::Vec3b rgbPixel = rgbImage.at<cv::Vec3b>(i, j);
            uint32_t rgb = (static_cast<uint32_t>(rgbPixel[2]) << 16 |
                            static_cast<uint32_t>(rgbPixel[1]) << 8 |
                            static_cast<uint32_t>(rgbPixel[0]));
            point.rgb = *reinterpret_cast<float*>(&rgb);

            pointCloud->push_back(point);
        }
    }

// After generating the point cloud, print the number of points
std::cout << "Number of points in the point cloud: " << pointCloud->size() << std::endl;

// Calculate and print the total possible number of points for a 960x540 image
int totalPossiblePoints = 960 * 540; // Assuming the depth image size is 960x540
std::cout << "Total possible number of points for a 960x540 image: " << totalPossiblePoints << std::endl;

// Corrected Visualization section
pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PointCloud Viewer"));
viewer->setBackgroundColor(0, 0, 0);

// Correctly declare the color handler
pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pointCloud);

// Correctly add the point cloud to the viewer
viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud, rgb, "sample cloud");

viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
viewer->addCoordinateSystem(1.0);
viewer->initCameraParameters();

while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}


return 0;
}

