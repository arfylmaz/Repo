#include <thread>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/point_cloud.h>
#include <opencv2/core.hpp>
#include <chrono>


void calculateDimensionsAndVisualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, pcl::visualization::PCLVisualizer::Ptr& viewer) {
    float min_x = std::numeric_limits<float>::max(), max_x = -std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max(), max_y = -std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max(), max_z = std::numeric_limits<float>::min();

    for (const auto& point : cloud->points) {
        if (point.x < min_x) min_x = point.x;
        if (point.x > max_x) max_x = point.x;
        if (point.y < min_y) min_y = point.y;
        if (point.y > max_y) max_y = point.y;
        if (point.z < min_z) min_z = point.z;
        if (point.z > max_z) max_z = point.z;
    }
    
    std::cout << "min_x = " << min_x << "\n";
    std::cout << "max_x = " << max_x << "\n";
    std::cout << "min_y = " << min_y << "\n";
    std::cout << "max_y = " << max_y << "\n";
    std::cout << "min_z = " << min_z << "\n";
    std::cout << "max_z = " << max_z << "\n";

    float width = max_x - min_x;
    float height = max_y - min_y;
    float depth = max_z - min_z;

    std::cout << "Width: " << width << " meters\n";
    std::cout << "Height: " << height << " meters\n";
    std::cout << "Depth: " << depth << " meters\n";

    pcl::PointXYZ minPt(min_x, min_y, min_z);
    pcl::PointXYZ maxPt(max_x, max_y, max_z);
    viewer->addLine(minPt, pcl::PointXYZ(max_x, min_y, min_z), 1, 0, 0, "line_x");
    viewer->addLine(minPt, pcl::PointXYZ(min_x, max_y, min_z), 0, 1, 0, "line_y");
    viewer->addLine(minPt, pcl::PointXYZ(min_x, min_y, max_z), 0, 0, 1, "line_z");
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Mat* image = static_cast<cv::Mat*>(userdata);
        if (image->channels() == 3) {
            cv::Vec3b pixelValue = image->at<cv::Vec3b>(y, x);
            std::cout << "RGB Pixel Value at (" << x << ", " << y << "): "
                      << "B: " << static_cast<int>(pixelValue[0]) << ", "
                      << "G: " << static_cast<int>(pixelValue[1]) << ", "
                      << "R: " << static_cast<int>(pixelValue[2]) << std::endl;
        } 

        if (image->type() == CV_32F) {
            float depthPixel = image->at<float>(y, x);
            std::cout << "Depth Pixel Value at (" << x << ", " << y << "): " << depthPixel << " meters" << std::endl;
        }
    }
}

// Bilinear Interpolation with resize
void bilinearInterpolation(cv::Mat& depthImage) {
    cv::Mat temp;
    cv::resize(depthImage, temp, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::resize(temp, depthImage, depthImage.size(), 0, 0, cv::INTER_LINEAR);
}

// Gaussian Filtering
void gaussianFilterDepthImage(cv::Mat& depthImage) {
    cv::GaussianBlur(depthImage, depthImage, cv::Size(5, 5), 0);
}

// Median Filtering
void medianFilterDepthImage(cv::Mat& depthImage) {
    cv::medianBlur(depthImage, depthImage, 5); // Kernel size is 5, adjust based on your needs
}

// Bilateral Filtering
void bilateralFilterDepthImage(cv::Mat& depthImage) {
    cv::Mat temp;
    cv::bilateralFilter(depthImage, temp, 9, 75, 75); // Diameter = 9, SigmaColor = 75, SigmaSpace = 75
    temp.copyTo(depthImage);
}

// Nearest Neighbor Interpolation
void nearestNeighborInterpolation(cv::Mat& depthImage) {
    cv::Mat temp;
    cv::resize(depthImage, temp, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST); // Downscale using nearest neighbor
    cv::resize(temp, depthImage, depthImage.size(), 0, 0, cv::INTER_NEAREST); // Upscale to original size
}

// Bicubic Interpolation
void bicubicInterpolation(cv::Mat& depthImage) {
    cv::Mat temp;
    cv::resize(depthImage, temp, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC); // Downscale using bicubic interpolation
    cv::resize(temp, depthImage, depthImage.size(), 0, 0, cv::INTER_CUBIC); // Upscale to original size
}


void edgeDirectedInterpolation(cv::Mat& image) {
    // Placeholder for simplicity. Actual EEDI would require a full implementation
    cv::Mat temp;
    cv::GaussianBlur(image, temp, cv::Size(3, 3), 0);
    cv::addWeighted(image, 1.5, temp, -0.5, 0, image);
}

void adaptiveInterpolation(cv::Mat& image) {
    // Ensure the image is in a suitable format; convert to grayscale if necessary.
    cv::Mat grayImage;
    if (image.channels() > 1) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Compute the gradient magnitude of the image to check for high variance areas.
    cv::Mat gradient;
    cv::Sobel(grayImage, gradient, CV_32F, 1, 1);
    
    // Use cv::Scalar to handle mean and stddev, ensuring output is continuous and in the right format.
    cv::Scalar mean, stddev;
    cv::meanStdDev(gradient, mean, stddev);

    // Access the standard deviation as a double, ensuring we handle the correct type.
    double sdValue = stddev.val[0];  // Access the first channel's standard deviation.

    // Adaptive resizing based on the gradient's standard deviation.
    if (sdValue > 5) {  // Hypothetical high variance threshold.
        cv::resize(image, image, cv::Size(), 2, 2, cv::INTER_CUBIC);  // Use bicubic for more detail preservation.
    } else {
        cv::resize(image, image, cv::Size(), 2, 2, cv::INTER_LINEAR);  // Use bilinear for smoother results in low variance areas.
    }
}


void fftInterpolation(cv::Mat& image, double scale_factor) {
    // Ensure image is in floating point format for DFT
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    // Compute DFT of the image
    cv::Mat dftOutput;
    cv::dft(floatImage, dftOutput, cv::DFT_COMPLEX_OUTPUT);

    // Calculate the new size based on the scale factor
    int newRows = cv::getOptimalDFTSize(int(dftOutput.rows * scale_factor));
    int newCols = cv::getOptimalDFTSize(int(dftOutput.cols * scale_factor));

    // Pad the DFT output to the new size: zero padding in frequency domain
    cv::Mat paddedDftOutput = cv::Mat::zeros(newRows, newCols, dftOutput.type());
    cv::Rect roi(cv::Point((newCols - dftOutput.cols) / 2, (newRows - dftOutput.rows) / 2), dftOutput.size());
    dftOutput.copyTo(paddedDftOutput(roi));

    // Inverse DFT to convert back to spatial domain
    cv::Mat inverseDft;
    cv::idft(paddedDftOutput, inverseDft, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    // Resize the image back to the original size
    cv::Mat resizedImage;
    cv::resize(inverseDft, resizedImage, image.size(), 0, 0, cv::INTER_LINEAR);

    // Normalize and convert back to the original type for visualization
    cv::normalize(resizedImage, resizedImage, 0, 1, cv::NORM_MINMAX);
    resizedImage.convertTo(image, image.type());
}

// Closing Function
void applyClosing(cv::Mat& image, int kernelSize = 11) {
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, element);
}



int main(int argc, char** argv) {
    cv::Mat rgbImage = cv::imread("../segmented_rgb.png", cv::IMREAD_COLOR);
    cv::Mat depthImage = cv::imread("../segmented_depth.tiff", cv::IMREAD_UNCHANGED);

    if (rgbImage.empty() || depthImage.empty()) {
        std::cerr << "Error loading images" << std::endl;
        return -1;
    }

    float maxReasonableDepth = 1.0f;
    depthImage.forEach<float>([&maxReasonableDepth](float &depth, const int position[]) -> void {
        if (depth > maxReasonableDepth || std::isinf(depth) || std::isnan(depth)) {
            depth = 0; 
        }
    });

    float fx = 1053.02, fy = 1052.72, cx = 894.2, cy = 560.436;
    float k1 = -0.0361431, k2 = 0.00486255, p1 = -0.000249107, p2 = 4.91599e-06, k3 = -0.00301238;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);

    std::vector<cv::Point2f> imagePoints;
    for (int i = 0; i < depthImage.rows; i++) {
        for (int j = 0; j < depthImage.cols; j++) {
            imagePoints.push_back(cv::Point2f(j, i));
        }
    }

    std::vector<cv::Point2f> undistortedPoints;
    cv::undistortPoints(imagePoints, undistortedPoints, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    float min_depth = 999999;
    float max_depth = -999999;

    for (size_t idx = 0; idx < undistortedPoints.size(); ++idx) {
        int i = static_cast<int>(imagePoints[idx].y);
        int j = static_cast<int>(imagePoints[idx].x);
        float depthValue = depthImage.at<float>(i, j);

        if (depthValue == 0) continue;
        if (depthValue > max_depth) max_depth = depthValue;
        else if (depthValue < min_depth) min_depth = depthValue;

        pcl::PointXYZRGB point;
        point.z = depthValue;
        point.x = (undistortedPoints[idx].x - cx) * depthValue / fx;
        point.y = (undistortedPoints[idx].y - cy) * depthValue / fy;

        cv::Vec3b rgbPixel = rgbImage.at<cv::Vec3b>(i, j);
        uint32_t rgb = (static_cast<uint32_t>(rgbPixel[2]) << 16 |
                        static_cast<uint32_t>(rgbPixel[1]) << 8 |
                        static_cast<uint32_t>(rgbPixel[0]));
               
        if (rgbPixel[0] == 0 && rgbPixel[1] == 0 && rgbPixel[2] == 0) {
            continue;
        }

        point.rgb = *reinterpret_cast<float*>(&rgb);
        pointCloud->push_back(point);
    }

    // Process depth image with different methods
    cv::Mat bilinearDepthImage = depthImage.clone();
    cv::Mat gaussianDepthImage = depthImage.clone();

    cv::Mat medianFilteredDepthImage = depthImage.clone();
    cv::Mat bilateralFilteredDepthImage = depthImage.clone();

    cv::Mat nearestNeighborDepthImage = depthImage.clone();
    cv::Mat bicubicDepthImage = depthImage.clone();
    cv::Mat edgeDirectedInterpolationDepthImage = depthImage.clone();

    cv::Mat fftInterpolationDepthImage = depthImage.clone();
    cv::Mat adaptiveInterpolationDepthImage = depthImage.clone();

     // Apply various filters and interpolations
    cv::Mat ClosingDepthImage = depthImage.clone();


    std::cout << "Max depth is " << max_depth << "\n" << "Min depth is " << min_depth << "\n";

    std::cout << "Number of points in the point cloud: " << pointCloud->size() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    bilinearInterpolation(bilinearDepthImage);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "bilinear Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    gaussianFilterDepthImage(gaussianDepthImage);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "gaussianFilter Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    medianFilterDepthImage(medianFilteredDepthImage);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Median Filter Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    bilateralFilterDepthImage(bilateralFilteredDepthImage);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Bilateral Filter Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    nearestNeighborInterpolation(nearestNeighborDepthImage);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "nearest Neighbor Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    bicubicInterpolation(bicubicDepthImage);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Bicubic Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    edgeDirectedInterpolation(edgeDirectedInterpolationDepthImage);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "edge Directed Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    adaptiveInterpolation(adaptiveInterpolationDepthImage);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Adaptive Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    fftInterpolation(fftInterpolationDepthImage,2);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "FFT Interpolation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    applyClosing(ClosingDepthImage);

    // Display the processed image
    cv::namedWindow("Closed Bilateral Filtered Depth Image");
    cv::imshow("Closed Bilateral Filtered Depth Image", ClosingDepthImage);




    //cv::namedWindow("RGB Image");
    //cv::setMouseCallback("RGB Image", onMouse, &rgbImage);
    //cv::imshow("RGB Image", rgbImage);

    cv::Mat normalizedDepthImage;
    cv::normalize(depthImage, normalizedDepthImage, 0, 65535, cv::NORM_MINMAX, CV_8U);

    cv::namedWindow("Depth Image");
    cv::setMouseCallback("Depth Image", onMouse, &depthImage);
    cv::imshow("Depth Image", depthImage);

    //cv::namedWindow("Bilinear Interpolation Depth Image");
    //cv::imshow("Bilinear Interpolation Depth Image", bilinearDepthImage);

    //cv::namedWindow("Gaussian Filtered Depth Image");
    //cv::imshow("Gaussian Filtered Depth Image", gaussianDepthImage);

    //cv::namedWindow("Median Filtered Depth Image");
    //cv::imshow("Median Filtered Depth Image", medianFilteredDepthImage);

    //cv::namedWindow("Bilateral Filtered Depth Image");
    //cv::imshow("Bilateral Filtered Depth Image", bilateralFilteredDepthImage);

    //cv::namedWindow("Nearest Neighbor Filtered Depth Image");
    //cv::imshow("Nearest Neighbor Filtered Depth Image", nearestNeighborDepthImage);

    //cv::namedWindow("Bicubic Interpolated Depth Image");
    //cv::imshow("Bicubic Interpolated Depth Image", bicubicDepthImage);

    //cv::namedWindow("Edge Directed Interpolated Depth Image");
    //cv::imshow("Edge Directed Interpolated Depth Image", edgeDirectedInterpolationDepthImage);

    cv::namedWindow("FFT Interpolated Depth Image");
    cv::setMouseCallback("FFT Image", onMouse, &fftInterpolationDepthImage);
    cv::imshow("FFT Interpolated Depth Image", fftInterpolationDepthImage);

    //cv::namedWindow("Adaptive Interpolated Depth Image");
    //cv::imshow("Adaptive Interpolated Depth Image", adaptiveInterpolationDepthImage);

    cv::waitKey(0);
/*
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PointCloud Viewer"));

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pointCloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud, rgb, "sample cloud");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "mesh"); // Set mesh color to green

    calculateDimensionsAndVisualize(pointCloud, viewer);

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    */

    return 0;
}
