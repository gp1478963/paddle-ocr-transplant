//
// Created by gengpeng on 2024/4/26.
//

#ifndef PADDLEV2_INFERENCE_PREDICT_RECOGNTION_HPP
#define PADDLEV2_INFERENCE_PREDICT_RECOGNTION_HPP

#include "predict_config.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <ctime>

namespace modern_ocr {
namespace recogntion {

class BaseRecLabelDecode {
 public:
  explicit BaseRecLabelDecode(const config::Args &param_args, bool use_space_char = false)
	  : args(param_args) {
	beg_str = "sos";
	end_str = "eos";

	if (param_args.rec_char_dict_path.empty()) {
	  std::cout << "Key words path:[" << param_args.rec_char_dict_path
				<< "] not exist, so use digits and alphabet instead." << std::endl;
	  character_str.emplace_back("0123456789abcdefghijklmnopqrstuvwxyz");
	} else {
	  std::ifstream fin(param_args.rec_char_dict_path, std::ios::in);
	  std::string line;
	  while (std::getline(fin, line)) {
		// Remove newline characters
		line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
		line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
		character_str.push_back(line);
	  }

	  if (use_space_char) {
		character_str.push_back(" ");
	  }
	  char_counter = character_str.size();
	  std::cout << "Total read char counter:[" << char_counter << "]\n";

	}

	addSpecialChar();
	for (size_t i = 0; i < character_str.size(); ++i) {
	  dict[character_str[i]] = i;
	}
  }

  config::Args args;

  unsigned int char_counter;

  virtual void
  addSpecialChar() {
	// Add any special characters if needed
  }

  std::vector<std::pair<std::string, float>>
  decode(
	  const std::vector<std::vector<int>> &text_index,
	  const std::vector<std::vector<float>> &text_prob = {},
	  bool is_remove_duplicate = false) {
	std::vector<std::pair<std::string, float>> result_list;
	auto ignored_tokens = getIgnoredTokens();
	size_t batch_size = text_index.size();

	for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
	  std::string char_list;
	  std::vector<float> conf_list;
	  for (size_t idx = 0; idx < text_index[batch_idx].size(); ++idx) {
		if (std::find(ignored_tokens.begin(), ignored_tokens.end(),
					  text_index[batch_idx][idx])!=ignored_tokens.end()) {
		  continue;
		}
		if (is_remove_duplicate && idx > 0 &&
			text_index[batch_idx][idx - 1]==text_index[batch_idx][idx]) {
		  continue;
		}
		char_list += character_str[text_index[batch_idx][idx]];
		if (!text_prob.empty()) {
		  conf_list.push_back(text_prob[batch_idx][idx]);
		} else {
		  conf_list.push_back(1.0f);
		}
	  }
	  float average_conf =
		  std::accumulate(conf_list.begin(), conf_list.end(), 0.0f)/
			  conf_list.size();
	  result_list.emplace_back(char_list, average_conf);
	}
	return result_list;
  }

  std::vector<std::pair<std::string, float>>
  decode(const cv::Mat &test_index, const cv::Mat &text_prob,
		 bool is_remove_duplicate = false) {
	std::vector<std::pair<std::string, float>> result_list;
	auto ignored_tokens = getIgnoredTokens();
	std::vector<int> index;
	std::vector<std::pair<int, float>> index_and_pro;

	for (int index_r = 0; index_r < test_index.rows; index_r++) {
	  for (int index_c = 0; index_c < test_index.cols; index_c++) {
		auto index_word = test_index.at<int>(index_r, index_c);
		auto pro = text_prob.at<float>(index_r, index_c);
		if (pro > args.drop_score && index_word > 0 && index_word <= char_counter) {
		  index_and_pro.emplace_back(index_word, pro);
		}
	  }
	}

	std::string words;
	float score_mean = 0.0;
	for (const auto &in : index_and_pro) {
	  words.append(character_str[in.first - 1]);
	  score_mean += in.second;
	}

	result_list.emplace_back(words, score_mean/(float)index_and_pro.size());
	return result_list;
  }

  std::vector<int>
  getIgnoredTokens() {
	return {0}; // for ctc blank
  }

 protected:
  std::string beg_str;
  std::string end_str;
  std::vector<std::string> character_str;
  std::unordered_map<std::string, int> dict;

 private:
  std::vector<std::string>
  add_special_char(const std::vector<std::string> &dict_character) {
	return dict_character; // Override this method in derived classes if needed
  }
};

class CTCLabelDecode : public BaseRecLabelDecode {
 public:
  explicit CTCLabelDecode(const config::Args &param_args, bool use_space_char = false)
	  : BaseRecLabelDecode(param_args, use_space_char) {
  }

  CTCLabelDecode() : CTCLabelDecode(config::init_args(), false) {}

  std::vector<std::pair<std::string, float>>
  operator()(const cv::Mat &preds, const cv::Mat &label = cv::Mat(),
			 const std::vector<cv::Mat> &args = {}) {

	std::cout << "output shape:" << preds.size << std::endl;
	auto preds_s = preds.reshape(1, 320);

	//    std::cout << cv::format(preds_s.row(0), cv::Formatter::FMT_NUMPY) << std::endl;

	//    for (int y = 0; y < 150; ++y) {
	//      float val = 0;
	//      int index = 0;
	//      for (int x = 0; x < 6625; x ++ ) {
	//         auto tmp = preds_s.at<float>(y, x);
	//         if(tmp - val > 0.0) {
	//           val = tmp;
	//           index = x;
	//         }
	//      }
	//      if(index > 0)
	//        std::cout << "index ->" << index << std::endl;
	//    }

	cv::Mat preds_idx;
	cv::reduceArgMax(preds_s, preds_idx, 1);
	cv::Mat preds_prob;
	cv::reduce(preds_s, preds_prob, 1, cv::REDUCE_MAX);

	std::vector<std::pair<std::string, float>> text =
		decode(preds_idx, preds_prob, true);
	//    if (label.empty()) {
	//      return text;
	//    }
	//    std::vector<std::pair<std::string, float>> label_text = decode(label);
	return text;
  }

 protected:
  void
  addSpecialChar() override {
	character_str.insert(character_str.begin(), "blank");
  }
};

class TextRecognizer {
 public:
  explicit TextRecognizer(const config::Args &args) : model_load_status(false) {
	rec_image_shape = parseImageShape(args.rec_image_shape);
	temp_for_fill = cv::Mat::zeros(rec_image_shape[1], rec_image_shape[2], CV_32FC3);

	ctcLabelDecode = CTCLabelDecode(args, args.use_space_char);
	this->args = args;

	try {
	  predictor = cv::dnn::readNetFromONNX(this->args.rec_model_dir);
	  model_load_status = !predictor.empty();
	}
	catch (std::exception &error) {
	  std::cout << error.what() << std::endl;
	  model_load_status = false;
	}
  }

  TextRecognizer() = default;

  ~TextRecognizer() = default;

  static std::vector<int>
  parseImageShape(const std::string &image_shape_str) {
	std::vector<int> result;
	std::istringstream iss(image_shape_str);
	std::string token;
	while (std::getline(iss, token, ',')) {
	  result.push_back(std::stoi(token));
	}
	return result;
  }

 public:
  bool
  isModelLoadStatus() const {
	return model_load_status;
  }

 private:
  config::Args args;
  cv::dnn::Net predictor;
  CTCLabelDecode ctcLabelDecode;
  std::vector<int> rec_image_shape;
  std::vector<std::pair<std::string, float>> postprocess_op;
  cv::Mat temp_for_fill;
  bool model_load_status{};

 private:
  std::vector<cv::Mat>
  splicing(const std::vector<cv::Mat> &img_list) {
	std::vector<cv::Mat> image_resized_list;
	cv::Mat tempForFill = cv::Mat::zeros(rec_image_shape[1], rec_image_shape[2], CV_32FC3);
	int total_length = 0;
	for (const auto &img : img_list) {
	  auto resized_img = resizeNormImg(img);

	  if ((total_length + resized_img.cols) < rec_image_shape[2]) {
		auto mask = tempForFill(cv::Rect(total_length, 0, resized_img.cols, resized_img.rows));
		resized_img.copyTo(mask);
		total_length += resized_img.cols;
	  } else {
		image_resized_list.emplace_back(tempForFill.clone());
		tempForFill = cv::Mat::zeros(rec_image_shape[1], rec_image_shape[2], CV_32FC3);
		auto mask = tempForFill(cv::Rect(0, 0, resized_img.cols, resized_img.rows));
		resized_img.copyTo(mask);
		total_length = resized_img.cols;
	  }
	}
	image_resized_list.emplace_back(tempForFill);
	return image_resized_list;
  }
 public:

  // this function is dropped out, instead of ( resizeNormImg(const cv::Mat &img)->cv::Mat )
  cv::Mat
  resizeNormImg(const cv::Mat &img, double max_wh_ratio) {
	// TODO channel counter must is 3
	int inputC = rec_image_shape[0];
	int inputH = rec_image_shape[1];
	int inputW = rec_image_shape[2];

	int imageH = img.rows;
	int imageW = img.cols;

	//        std::cout << cv::format(img, cv::Formatter::FMT_NUMPY) << std::endl;

//        std::cout << "-------------" << img.size << std::endl;
	auto ratio = std::max((float)imageW/inputW, (float)imageH/inputH);
	cv::Mat resized_image_;
	cv::resize(img, resized_image_, cv::Size(imageW/ratio, imageH/ratio));
	//        cv::resize(img, resized_image_, cv::Size(inputW, inputH));

	cv::Mat input_tensor = cv::Mat::zeros(inputH, inputW, CV_8UC3);
	cv::Mat mask =
		input_tensor(cv::Rect(0, 0, resized_image_.cols, resized_image_.rows));
	//        resized_image_.copyTo(input_tensor, mask);
	resized_image_.copyTo(mask);
	//        cv::addWeighted(mask, 0, resized_image_, 1,0.0, mask);

	input_tensor.convertTo(input_tensor, CV_32FC3, 1/255.0);
	//        cv::imshow("show", input_tensor);
	//        std::cout << cv::format(input_tensor, cv::Formatter::FMT_NUMPY) << std::endl; cv::waitKey(00);
	return input_tensor;
  }

  cv::Mat
  resizeNormImg(const cv::Mat &img) {
	// TODO channel counter must is 3
	int inputC = rec_image_shape[0];
	int inputH = rec_image_shape[1];
	int inputW = rec_image_shape[2];

	int imageH = img.rows;
	int imageW = img.cols;

	auto ratio = std::max((float)imageW/inputW, (float)imageH/inputH);
	cv::Mat resized_image_;
	cv::resize(img, resized_image_, cv::Size(imageW/ratio, imageH/ratio));

	resized_image_.convertTo(resized_image_, CV_32FC3, 1/255.0);
	return resized_image_;
  }

  std::vector<std::pair<std::string, float>>
  operator()(const std::vector<cv::Mat> &img_list) {
	auto resized_img_list = this->splicing(img_list);
	std::vector<std::pair<std::string, float>> rec_res;

	for (const auto &img : resized_img_list) {
	  cv::Mat blob = cv::dnn::blobFromImage(img);
	  predictor.setInput(blob);
//	  std::cout << blob.size << std::endl;
	  cv::Mat preds = predictor.forward();
	  std::vector<std::pair<std::string, float>> rec_result = ctcLabelDecode(preds);
//            for (const auto &r : rec_result)
//            {
//                std::cout << r.first << std::endl;
//            }
	  if (!rec_result.empty()) {
		rec_res.push_back(rec_result[0]);
	  }

	}

	return rec_res;

  }
};

}
}

#endif // PADDLEV2_INFERENCE_PREDICT_RECOGNTION_HPP
