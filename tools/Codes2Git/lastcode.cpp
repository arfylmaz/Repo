#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void colorBasedSegmentation(const Mat& image) {
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    Scalar lowerGreen(35, 70, 70);
    Scalar upperGreen(85, 255, 255);
    Mat mask;
    inRange(hsvImage, lowerGreen, upperGreen, mask);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourOutput = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > 1500) {
            drawContours(contourOutput, contours, static_cast<int>(i), Scalar(rand() % 255, rand() % 255, rand() % 255), 2);
        }
    }

    imshow("Color-Based Segmentation", contourOutput);
    imwrite("../color_based_segmentation_result.png", contourOutput);
    waitKey(0);
}

void kMeansSegmentation(const Mat& image) {
    int clusters = 3;
    Mat labImage;
    cvtColor(image, labImage, COLOR_BGR2Lab);

    Mat imageFlat = labImage.reshape(1, image.total());
    imageFlat.convertTo(imageFlat, CV_32F);

    Mat labels;
    Mat centers;
    kmeans(imageFlat, clusters, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // Assign distinct colors for each cluster
    vector<Vec3b> colors;
    for (int i = 0; i < clusters; i++) {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Map the clustered labels to the corresponding colors
    Mat segmented(image.size(), image.type());
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int cluster_idx = labels.at<int>(y * image.cols + x);
            segmented.at<Vec3b>(y, x) = colors[cluster_idx];
        }
    }

    cvtColor(segmented, segmented, COLOR_Lab2BGR);

    imshow("K-Means Segmentation", segmented);
    imwrite("../kmeans_segmentation_result.png", segmented);
    waitKey(0);
}


void adaptiveThresholdSegmentation(const Mat& image) {
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    Mat adaptiveThresh;
    adaptiveThreshold(grayImage, adaptiveThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

    imshow("Adaptive Threshold Segmentation", adaptiveThresh);
    imwrite("../adaptive_threshold_segmentation.png", adaptiveThresh);
    waitKey(0);
}

void cannyEdgeDetection(const Mat& image) {
    Mat gray, edges;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    blur(gray, gray, Size(3,3));
    Canny(gray, edges, 100, 200); // You can adjust these thresholds

    imshow("Canny Edges", edges);
    imwrite("../canny_edges.png", edges);
    waitKey(0);
}

void watershedSegmentation(Mat& image) {
    // Convert to binary image
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat binary;
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(binary, dist, DIST_L2, 5);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);

    // Threshold to obtain the peaks
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);

    // Create the CV_8U version of the distance image
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point>> contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);

    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    }

    // Draw the background marker
    circle(markers, Point(5,5), 3, Scalar(255), -1);

    // Perform the watershed algorithm
    watershed(image, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);

    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
        }
    }

    imshow("Watershed Segmentation", dst);
    imwrite("../watershed_segmentation.png", dst);
    waitKey(0);
}

Mat cropCentralPlant(const Mat& image, const Mat& mask) {
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Point imageCenter(image.cols / 2, image.rows / 2);
    Rect bestBoundingBox;
    double minDistanceToCenter = std::numeric_limits<double>::max();
    double maxArea = 1500; // Minimum area threshold to consider a contour

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > maxArea) {
            Rect tempBoundingBox = boundingRect(contour);
            Point contourCenter = 
                Point(tempBoundingBox.x + tempBoundingBox.width / 2, 
                      tempBoundingBox.y + tempBoundingBox.height / 2);
            double distanceToCenter = norm(Mat(contourCenter), Mat(imageCenter));

            // Update if this contour is closer to the center and has a larger area than previous ones
            if (distanceToCenter < minDistanceToCenter) {
                bestBoundingBox = tempBoundingBox;
                minDistanceToCenter = distanceToCenter;
                maxArea = area; // Update the area to the largest found so far
            }
        }
    }

    if (bestBoundingBox.area() > 0) {
        return image(bestBoundingBox);
    } else {
        return Mat();
    }
}

int main() {
    // Load the image
    Mat image = imread("../rgb.png");
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Preprocess and find the HSV mask for green color
    Scalar lowerGreen(35, 70, 70); // Adjusted lower bound for green
    Scalar upperGreen(85, 255, 255); // Adjusted upper bound for green
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);
    Mat mask;
    inRange(hsvImage, lowerGreen, upperGreen, mask);

    // Crop the central plant from the image
    Mat croppedImage = cropCentralPlant(image, mask);
    if (!croppedImage.empty()) {
        imshow("Cropped Plant", croppedImage);
        imwrite("../cropped_plant.png", croppedImage);
        waitKey(0);

        // Apply segmentation to the cropped image
        colorBasedSegmentation(croppedImage);
        kMeansSegmentation(croppedImage);
        cannyEdgeDetection(croppedImage);
        watershedSegmentation(croppedImage);   
        adaptiveThresholdSegmentation(croppedImage);
    } else {
        cout << "No central plant found." << endl;
    }


    return 0;
}



