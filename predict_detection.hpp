//
// Created by gp147 on 2024/4/26.
//

#ifndef PADDLEV2_INFERENCE_PREDICT_DETECTION_HPP
#define PADDLEV2_INFERENCE_PREDICT_DETECTION_HPP

#include "clipper/clipper.hpp"
#include "predict_config.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

namespace modern_ocr {
namespace detection {

class DBPreProcess final {
 public:

  explicit DBPreProcess(config::Args args) : args(std::move(args)) {}

  DBPreProcess() : DBPreProcess(config::init_args()) {}

  ~DBPreProcess() = default;

  std::tuple<cv::Mat, std::vector<float>>
  operator()(const cv::Mat &data) const {
	cv::Mat processed_data;
	std::vector<float> shape_list;
	std::tie(processed_data, shape_list) = DetResizeForTest(
		args.det_limit_side_len, args.det_limit_type, data);
	processed_data = NormalizeImage(processed_data);
	processed_data = ToCHWImage(processed_data);
	return std::tuple<cv::Mat, std::vector<float>>(processed_data, shape_list);
  }

 private:
  config::Args args;

  static cv::Mat
  NormalizeImage(const cv::Mat &data) {
	float scale = 1.0f/255;
	cv::Scalar mean(0.485f, 0.456f, 0.406f);
	cv::Scalar std(0.229f, 0.224f, 0.225f);
	cv::Mat img;
	data.convertTo(img, CV_32F, scale);
	cv::subtract(img, mean, img);
	cv::divide(img, std, img);
	return img;
  }

  static cv::Mat
  ToCHWImage(const cv::Mat &data) {
	std::vector<cv::Mat> channels(3);
	cv::split(data, channels);
	cv::Mat chw;
	cv::merge(channels, chw);
	return chw;
  }

  static std::vector<cv::Mat>
  KeepKeys(const std::vector<std::string> &keep_keys, const cv::Mat &data) {
	// This function would need to be adapted based on how you store 'image' and 'shape'
	// For example, you might return a vector containing the image and a Mat with the
	// shape data
	return {data}; // Placeholder
  }

  static std::tuple<cv::Mat, std::vector<float>>
  DetResizeForTest(
	  float limit_side_len, const std::string &limit_type, const cv::Mat &data) {
	int src_h = data.rows;
	int src_w = data.cols;
	float ratio = 1.0f;

	if (src_h >= limit_side_len || src_w >= limit_side_len) {
	  ratio = std::min(limit_side_len/(float)src_h,
					   limit_side_len/(float)src_w);
	}

	int resize_h =
		std::max(static_cast<int>(std::round(src_h*ratio/32)*32), 32);
	int resize_w =
		std::max(static_cast<int>(std::round(src_w*ratio/32)*32), 32);

	std::cout << "DetResizeForTest: image shape original:[" << src_w << "," << src_h
			  << "] reshape to [" << resize_w << "," << resize_h << "] and ratio is " << ratio << std::endl;

	cv::Mat resized_img;
	cv::resize(data, resized_img, cv::Size(resize_w, resize_h));
	std::vector<float> shape_list{(float)src_h, (float)src_w,
								  (float)resize_h/src_h,
								  (float)resize_w/src_w};
	return std::tuple<cv::Mat, std::vector<float>>{resized_img, shape_list};
  }
};

class DBPostProcess {
 public:

  explicit DBPostProcess(config::Args param_args) : args(std::move(param_args)) {
	dilation_kernel = args.use_dilation ?
					  (cv::Mat_<uchar>(2, 2) << 1, 1, 1, 1) :
					  cv::Mat();
  }

  DBPostProcess() : DBPostProcess(config::init_args()) {}

  ~DBPostProcess() = default;

  DBPostProcess &operator=(const DBPostProcess &other)
  = default;

  std::pair<std::vector<cv::Rect>, std::vector<float>>
  boxes_from_bitmap(
	  const cv::Mat &pred, const cv::Mat &_bitmap, int dest_width, int dest_height) {
	std::cout << "_bitmap shape:["
			  << _bitmap.size[0] << ","
			  << _bitmap.size[1] << ","
			  << _bitmap.size[2] << ","
			  << _bitmap.size[3] << "]" << std::endl;

	cv::Mat bitmap = _bitmap.reshape(0, _bitmap.size[2]);
	cv::Mat pred_ = pred.reshape(0, _bitmap.size[2]);
	int height = bitmap.size[0];
	int width = bitmap.size[1];

	std::cout << "prepare entry boxes_from_bitmap :shape:[" << bitmap.size << "]" << std::endl;

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bitmap*255, contours, cv::RETR_LIST,
					 cv::CHAIN_APPROX_SIMPLE);

	std::cout << "entry boxes_from_bitmap: " << contours.size() << std::endl;
	int num_contours =
		std::min(static_cast<int>(contours.size()), args.max_candidates);

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	for (int index = 0; index < num_contours; ++index) {
	  const std::vector<cv::Point> &contour = contours[index];

	  std::vector<cv::Point2f> points;
	  float sside;

	  std::tie(points, sside) = get_mini_boxes(contour);
	  if (sside < args.det_db_piexl) {
		continue;
	  }
	  cv::Mat points_mat(points);
	  float score = box_score_fast(pred_, points_mat.reshape(2));
	  if (args.det_db_box_thresh > score) {
		continue;
	  }

	  auto points_u = unclip(points);
	  std::tie(points_u, sside) = get_mini_boxes(points_u);
	  if (sside < args.det_db_piexl + 2) {
		continue;
	  }

	  for (auto &p : points_u) {
//                std::cout << "ORI: x:" << p.x << "\ty:" << p.y << std::endl;
		p.x = std::min(int(p.x/width*dest_width), dest_width);
		p.x = std::max(int(p.x), 0);

		p.y = std::min(int(p.y/height*dest_height), dest_height);
		p.y = std::max(int(p.y), 0);

//                std::cout << "POST: x:" << p.x << "\ty:" << p.y << std::endl;
	  }

	  boxes.push_back(cv::boundingRect(points_u));
	  scores.push_back(score);
	}

	return {boxes, scores};
  }

 private:
  config::Args args;
  cv::Mat dilation_kernel;

  static std::tuple<std::vector<cv::Point2f>, float>
  get_mini_boxes(const std::vector<cv::Point2f> &contour) {
	cv::RotatedRect bounding_box = cv::minAreaRect(contour);

//        std::cout << "-------------------------------------------------------------"
//                  << std::endl;
//        for (const auto &pointt : contour)
//        {
//            std::cout << pointt << std::endl;
//        }
//        std::cout << "-------------------------------------------------------------"
//                  << std::endl;

	std::vector<cv::Point2f> points;
	bounding_box.points(points);

	std::sort(points.begin(), points.end(),
			  [](cv::Point2f &p1, cv::Point2f &p2) { return p1.x < p2.x; });

	int index_1, index_2, index_3, index_4;
	if (points[1].y > points[0].y) {
	  index_1 = 0;
	  index_4 = 1;
	} else {
	  index_1 = 1;
	  index_4 = 0;
	}
	if (points[3].y > points[2].y) {
	  index_2 = 2;
	  index_3 = 3;
	} else {
	  index_2 = 3;
	  index_3 = 2;
	}

	std::vector<cv::Point2f> box = {points[index_1], points[index_2],
									points[index_3], points[index_4]};
	return std::make_tuple(
		box, std::min(bounding_box.size.width, bounding_box.size.height));
  }

  static std::tuple<std::vector<cv::Point2f>, float>
  get_mini_boxes(const std::vector<cv::Point> &contour) {
	cv::RotatedRect bounding_box = cv::minAreaRect(contour);
	std::vector<cv::Point2f> points;
	bounding_box.points(points);

	std::sort(points.begin(), points.end(),
			  [](cv::Point2f &p1, cv::Point2f &p2) { return p1.x < p2.x; });

	int index_1, index_2, index_3, index_4;
	if (points[1].y > points[0].y) {
	  index_1 = 0;
	  index_4 = 1;
	} else {
	  index_1 = 1;
	  index_4 = 0;
	}
	if (points[3].y > points[2].y) {
	  index_2 = 2;
	  index_3 = 3;
	} else {
	  index_2 = 3;
	  index_3 = 2;
	}

	std::vector<cv::Point2f> box = {points[index_1], points[index_2],
									points[index_3], points[index_4]};
	return std::make_tuple(
		box, std::min(bounding_box.size.width, bounding_box.size.height));
  }

  static float
  box_score_fast(const cv::Mat &bitmap,
				 const std::vector<cv::Point> &box) {
	int h = bitmap.size[0];
	int w = bitmap.size[1];

//        std::cout << "bitmap size:" << h << " " << w << std::endl;
	// Copy and convert box to a vector of cv::Point
	std::vector<cv::Point> _box(box.begin(), box.end());

//        for (const auto &point_l : _box)
//        {
//            std::cout << point_l.x << " " << point_l.y << std::endl;
//        }

	// Calculate the bounding box's min and max x and y coordinates
	struct {
	  bool
	  operator()(cv::Point a, cv::Point b) const {
		return a.x < b.x;
	  }
	} x_cmp;
	struct {
	  bool
	  operator()(cv::Point a, cv::Point b) const {
		return a.y < b.y;
	  }
	} y_cmp;

//        std::cout << std::max(
//            0, (*std::min_element(_box.begin(), _box.end(), x_cmp)).x)
//                  << std::endl;
	int xmin = std::max(
		0, static_cast<int>(std::floor(
			(*std::min_element(_box.begin(), _box.end(), x_cmp)).x)));
	int xmax = std::min(
		w - 1, static_cast<int>(std::ceil(
			(*std::max_element(_box.begin(), _box.end(), x_cmp)).x)));
	int ymin = std::max(
		0, static_cast<int>(std::floor(
			(*std::min_element(_box.begin(), _box.end(), y_cmp)).y)));
	int ymax = std::min(
		h - 1, static_cast<int>(std::ceil(
			(*std::max_element(_box.begin(), _box.end(), y_cmp)).y)));

//        std::cout << "Mask:[" << xmin << " " << xmax << " " << ymin << " " << ymax
//                  << std::endl;
	// Create a mask with the same size as the bounding box
	cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

	// Offset the box coordinates to the mask's coordinate system
	for (cv::Point &p : _box) {
	  p.x -= xmin;
	  p.y -= ymin;
	}

	// Fill the polygon in the mask
	std::vector<std::vector<cv::Point>> contours{_box};
	cv::fillPoly(mask, contours, cv::Scalar(1));

	// Calculate the mean score within the mask
	return static_cast<float>(
		cv::mean(bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)),
				 mask)[0]);
  }

  std::vector<cv::Point2f>
  unclip(const std::vector<cv::Point2f> &box) const {
	// 计算多边形面积和周长
	double area = 0.0;
	double length = 0.0;
	cv::Point2f last_point = box.back();
	for (const auto &point : box) {
	  area += last_point.x*point.y - point.x*last_point.y;
	  length += cv::norm(point - last_point);
	  last_point = point;
	}
	area = std::abs(area*0.5);

	// 计算扩展距离
	double distance = area*args.det_db_unclip_ratio/length;

	// 使用Clipper库进行路径扩展
	ClipperLib::Path subj;
	ClipperLib::Paths solution;
	for (const auto &point : box) {
	  subj << ClipperLib::IntPoint(static_cast<int>(point.x),
								   static_cast<int>(point.y));
	}

	ClipperLib::ClipperOffset co;
	co.AddPath(subj, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
	co.Execute(solution, distance);

	// 将Clipper路径转换回OpenCV点数组
	std::vector<cv::Point2f> expanded;
	for (const auto &point : solution[0]) {
	  expanded.emplace_back(static_cast<float>(point.X),
							static_cast<float>(point.Y));
	}

	return expanded;
  }

 public:
  std::vector<std::map<std::string, std::vector<cv::Rect>>>
  operator()(
	  const std::map<std::string, cv::Mat> &outs_dict,
	  const std::vector<std::tuple<int, int, float, float>> &shape_list) {
	cv::Mat pred = outs_dict.at("maps");
	std::cout << "pred shape [" << pred.size << "]" << std::endl;
	pred = pred.reshape(1).row(0);
	cv::Mat segmentation = pred > args.det_db_thresh;

	std::vector<std::map<std::string, std::vector<cv::Rect>>> boxes_batch;

	for (int batch_index = 0; batch_index < pred.size[0]; ++batch_index) {
	  int src_h, src_w;
	  float ratio_h, ratio_w;
	  std::tie(src_h, src_w, ratio_h, ratio_w) = shape_list[batch_index];

	  cv::Mat mask;
	  if (!dilation_kernel.empty()) {
		cv::dilate(segmentation.row(batch_index), mask, dilation_kernel);
	  } else {
		mask = segmentation.row(batch_index).row(0);
	  }

	  std::cout << "mask ok size:[" << mask.size << "]" << std::endl;

	  std::vector<cv::Rect> boxes;
	  std::vector<float> scores;
	  std::tie(boxes, scores) =
		  boxes_from_bitmap(pred.row(batch_index), mask, src_w, src_h);

	  boxes_batch.push_back({{"points", boxes}});
	}
	return boxes_batch;
  }
};

class TextDetector {
 public:
  TextDetector(config::Args args,
			   const DBPreProcess &preprocess_op,
			   const DBPostProcess &postprocess_op)
	  : config(std::move(args)),
		preprocess_op(preprocess_op),
		postprocess_op(postprocess_op) {
	try {
	  predictor = cv::dnn::readNetFromONNX(config.det_model_dir);
	  model_load_status = !predictor.empty();
	}
	catch (std::exception &error) {
	  std::cout << error.what() << std::endl;
	  model_load_status = false;
	}
  }

  explicit TextDetector(config::Args args) :
	  config(std::move(args)),
	  preprocess_op(config),
	  postprocess_op(config), model_load_status(false) {
	try {
	  predictor = cv::dnn::readNetFromONNX(config.det_model_dir);
	  model_load_status = !predictor.empty();
	}
	catch (std::exception &error) {
	  std::cout << error.what() << std::endl;
	  model_load_status = false;
	}
  }

  TextDetector() {}

  ~TextDetector() = default;

  bool
  isModelLoadStatus() const {
	return model_load_status;
  }
 private:
  bool model_load_status;
 private:
  static std::vector<cv::Point2f>
  orderPointsClockwise(const std::vector<cv::Point2f> &pts) {
	std::vector<int> xSortedIndices(pts.size());
	std::iota(xSortedIndices.begin(), xSortedIndices.end(), 0);
	//        std::sort(xSortedIndices.begin(), xSortedIndices.end(), []() {
	//                return pts[a].x < pts[b].x;
	//        });

	std::vector<cv::Point2f> leftMost = {pts[xSortedIndices[0]],
										 pts[xSortedIndices[1]]};
	std::vector<cv::Point2f> rightMost = {pts[xSortedIndices[2]],
										  pts[xSortedIndices[3]]};

	std::sort(leftMost.begin(), leftMost.end(),
			  [](cv::Point2f &a, cv::Point2f &b) { return a.y < b.y; });
	cv::Point2f tl = leftMost[0];
	cv::Point2f bl = leftMost[1];

	std::sort(rightMost.begin(), rightMost.end(),
			  [](cv::Point2f &a, cv::Point2f &b) { return a.y < b.y; });
	cv::Point2f tr = rightMost[0];
	cv::Point2f br = rightMost[1];

	return {tl, tr, br, bl};
  }

  static std::vector<cv::Point2f>
  clipDetRes(std::vector<cv::Point2f> points,
			 int img_height, int img_width) {
	for (auto &point : points) {
	  point.x =
		  std::min(std::max(point.x, 0.f), static_cast<float>(img_width - 1));
	  point.y =
		  std::min(std::max(point.y, 0.f), static_cast<float>(img_height - 1));
	}
	return points;
  }

  static std::vector<std::vector<cv::Point2f>>
  filterTagDetRes(const std::vector<std::vector<cv::Point2f>> &dt_boxes,
				  const cv::Size &image_shape) {
	std::vector<std::vector<cv::Point2f>> dt_boxes_new;
	for (const auto &box : dt_boxes) {
	  auto ordered_box = orderPointsClockwise(box);
	  auto clipped_box =
		  clipDetRes(ordered_box, image_shape.height, image_shape.width);
	  float rect_width = cv::norm(clipped_box[0] - clipped_box[1]);
	  float rect_height = cv::norm(clipped_box[0] - clipped_box[3]);
	  if (rect_width <= 3 || rect_height <= 3) {
		continue;
	  }
	  dt_boxes_new.push_back(clipped_box);
	}
	return dt_boxes_new;
  }

 public:
  std::vector<std::map<std::string, std::vector<cv::Rect>>>
  operator()(const cv::Mat &img) {
	cv::Mat ori_im = img.clone();
	cv::Mat second_deal_image;
	std::vector<float> shape_list;
	std::tie(second_deal_image, shape_list) = preprocess_op(img);
	auto size = second_deal_image.size;
//        std::cout << "second_deal_image shape: " << size[0] << " " << size[1] << " "
//                  << second_deal_image.channels() << std::endl;

	cv::Mat blob = cv::dnn::blobFromImage(second_deal_image);
//        std::cout << blob.size << std::endl;

	predictor.setInput(blob);
	std::vector<cv::Mat> outputs;
	predictor.forward(outputs);

	std::map<std::string, cv::Mat> inputs;
	inputs["maps"] = outputs[0];
	std::vector<std::tuple<int, int, float, float>> shape_lists;
	shape_lists.emplace_back((int)shape_list[0], (int)shape_list[1],
							 shape_list[2], shape_list[3]);

	auto post_result = postprocess_op(inputs, shape_lists);
	//        auto dt_boxes = filterTagDetRes(post_result, ori_im.size());

//        std::cout << post_result.size() << std::endl;

	auto &prediction = outputs[0];

	return post_result;
  }

 private:
  config::Args config;
  cv::dnn::Net predictor;
  DBPreProcess preprocess_op;
  DBPostProcess postprocess_op;
};

}
}

#endif // PADDLEV2_INFERENCE_PREDICT_DETECTION_HPP
