//
// Created by gp147 on 4/27/2024.
//

#ifndef PADDLEV2_INFERENCE_PREDICT_ALGORITHM_HPP
#define PADDLEV2_INFERENCE_PREDICT_ALGORITHM_HPP

#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace modern_ocr {
namespace algorithm {

inline std::tuple<std::vector<std::vector<cv::Point2f>>,
				  std::vector<std::tuple<std::string, float>>>
postprocess(
	const std::vector<std::vector<cv::Point2f>> &dt_boxes,
	const std::vector<std::tuple<std::string, float>> &rec_res) {
  std::vector<std::vector<cv::Point2f>> filter_boxes;
  std::vector<std::tuple<std::string, float>> filter_rec_res;
  for (size_t i = 0; i < dt_boxes.size(); ++i) {
	const auto &box = dt_boxes[i];
	const auto &rec_result = rec_res[i];
	const std::string &text = std::get<0>(rec_result);
	float score = std::get<1>(rec_result);
	//      if (score >= FLAGS.drop_score) {
	filter_boxes.push_back(box);
	filter_rec_res.push_back(rec_result);
	//      }
	return std::make_tuple(filter_boxes, filter_rec_res);
  }
}

inline cv::Mat
getRotateCropImage(const cv::Mat &img, const cv::Rect &rect) {

  // TODO 结果不对，待优化
  // Calculate the width and height of the cropped image
  //  double img_crop_width = rect.width;
  //  double img_crop_height = rect.height;
  //
  //  // Define the destination points for the perspective transform
  //  std::vector<cv::Point2f> pts_std = {
  //      cv::Point2f(0, 0),
  //      cv::Point2f(img_crop_width, 0),
  //      cv::Point2f(img_crop_width, img_crop_height),
  //      cv::Point2f(0, img_crop_height)
  //  };
  //
  //  // Get the perspective transform matrix
  //  std::vector<cv::Point2f> points = {{static_cast<float>(rect.y), static_cast<float>(rect.x)}};
  //  cv::Mat M = cv::getPerspectiveTransform(points.data(), pts_std.data());
  //
  //
  //
  //  // Apply the perspective transform
  //  cv::Mat dst_img;
  //  cv::warpPerspective(
  //      img,
  //      dst_img,
  //      M,
  //      cv::Size(img_crop_width, img_crop_height),
  //      cv::BORDER_REPLICATE,
  //      cv::INTER_CUBIC);
  //
  //  // Check if the image needs to be rotated
  //  int dst_img_height = dst_img.rows;
  //  int dst_img_width = dst_img.cols;
  //  if (static_cast<double>(dst_img_height) / dst_img_width >= 1.5) {
  //    cv::rotate(dst_img, dst_img, cv::ROTATE_90_CLOCKWISE);
  //  }
  //  cv::imshow("pict", dst_img);
  //  //    cv::imwrite("./result.png", img);

  // 直接使用roi，待优化
  cv::Mat roi = img(rect);
  return roi;
}

inline bool
compareRectByY(const cv::Rect &a, const cv::Rect &b) {
  // 比较两个矩形的y坐标
  return a.y < b.y;
}

// Function to sort boxes
inline std::vector<cv::Rect>
sorted_boxes(const std::vector<cv::Rect> &dt_boxes) {
  std::vector<cv::Rect> sorted_boxes(dt_boxes);
  std::sort(sorted_boxes.begin(), sorted_boxes.end(), compareRectByY);
  return sorted_boxes;
}

// Main function to preprocess boxes
inline std::tuple<std::vector<cv::Rect>, std::vector<cv::Mat>>
preprocess_boxes(
	const std::vector<std::map<std::string, std::vector<cv::Rect>>> &dt_boxes,
	const cv::Mat &ori_im) {
  if (dt_boxes.empty()) {
	throw 1;
  }
  auto boses = dt_boxes[0].at("points");

  std::vector<cv::Mat> img_crop_list;
  auto sorted_dt_boxes = sorted_boxes(boses);

  for (const auto &box : sorted_dt_boxes) {
	cv::Mat img_crop = getRotateCropImage(ori_im, box);
	img_crop_list.push_back(img_crop);
  }
  return std::make_tuple(sorted_dt_boxes, img_crop_list);
}

}
}

#endif // PADDLEV2_INFERENCE_PREDICT_ALGORITHM_HPP
