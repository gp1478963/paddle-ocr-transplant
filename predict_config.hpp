//
// Created by gengpeng on 2024/4/26.
//

#ifndef PADDLEV2_INFERENCE_PREDICT_CONFIG_HPP
#define PADDLEV2_INFERENCE_PREDICT_CONFIG_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

namespace modern_ocr {
namespace config {

struct Args {
  std::string image_path;
  std::string det_algorithm;
  std::string det_model_dir;
  float det_limit_side_len{};
  std::string det_limit_type;
  float det_db_thresh{};
  float det_db_box_thresh{};
  float det_db_unclip_ratio{};
  float det_db_piexl{};

  int max_batch_size{};
  bool use_dilation{};
  std::string det_db_score_mode;
  std::string rec_algorithm;
  std::string rec_model_dir;
  std::string rec_image_shape;
  int max_text_length{};
  std::string rec_char_dict_path;
  bool use_space_char{};
  std::string vis_font_path;
  float drop_score{};

  bool use_angle_cls{};
  std::string cls_model_dir;
  std::string cls_image_shape;
  std::vector<std::string> label_list;
  int cls_batch_num{};
  float cls_thresh{};
  int max_candidates;

  //TODO Not work, why?
  template<typename EleType>
  friend std::ostream &operator<<(std::ostream &Os, const std::vector<EleType> &vecS) {
	for (int i = 0; i < vecS.size(); i++) {
	  Os << "[" << i << "]:" << vecS[i] << "\n";
	}
	return Os;
  }

 private:
  static std::string VecStr2S(const std::vector<std::string> &data) {
	std::string s;
	for (int i = 0; i < data.size(); i++) {
	  s += (std::string("[") + std::to_string(i) + "]:" + data[i] + "\n");
	}
	return s;
  }

//			friend std::ostream & operator << (std::ostream& Os,std::vector<std::string>& vecS)
//			{
//				for (int i = 0; i < vecS.size(); i++)
//				{
//					Os << "[" << i << "]:" << vecS[i] << "\n";
//				}
//				return Os;
//			}

 public:
  friend std::ostream &
  operator<<(std::ostream &Os, const Args &Args) {
	Os << "image_path: [" << Args.image_path
	   << "] \ndet_algorithm: " << Args.det_algorithm
	   << "] \ndet_model_dir: [" << Args.det_model_dir
	   << "] \ndet_limit_side_len: " << Args.det_limit_side_len
	   << "] \ndet_limit_type: [" << Args.det_limit_type
	   << "] \ndet_db_thresh: [" << Args.det_db_thresh
	   << "] \ndet_db_box_thresh: [" << Args.det_db_box_thresh
	   << "] \ndet_db_unclip_ratio: [" << Args.det_db_unclip_ratio
	   << "] \ndet_db_piexl: [" << Args.det_db_piexl
	   << "] \nmax_batch_size: [" << Args.max_batch_size
	   << "] \nuse_dilation: [" << Args.use_dilation
	   << "] \ndet_db_score_mode: [" << Args.det_db_score_mode
	   << "] \nrec_algorithm: [" << Args.rec_algorithm
	   << "] \nrec_model_dir: [" << Args.rec_model_dir
	   << "] \nrec_image_shape: [" << Args.rec_image_shape
	   << "] \nmax_text_length: [" << Args.max_text_length
	   << "] \nrec_char_dict_path: [" << Args.rec_char_dict_path
	   << "] \nuse_space_char: [" << Args.use_space_char
	   << "] \nvis_font_path: [" << Args.vis_font_path
	   << "] \ndrop_score: [" << Args.drop_score
	   << "] \nuse_angle_cls: [" << Args.use_angle_cls
	   << "] \ncls_model_dir: [" << Args.cls_model_dir
	   << "] \ncls_image_shape: [" << Args.cls_image_shape
	   << "] \nlabel_list: [" << VecStr2S(Args.label_list)
	   << "] \ncls_batch_num: [" << Args.cls_batch_num
	   << "] \ncls_thresh: [" << Args.cls_thresh
	   << "] \nmax_candidates: [" << Args.max_candidates << "]";
	return Os;
  }

  void loadFormMap(const std::string &configPath) {
	std::fstream configStream(configPath, std::ios::in);
	std::string line;
	while (getline(configStream, line)) {
	  auto middlePos = line.find('=', 0);
	  if (middlePos==std::string::npos) {
		continue;
	  }

	  auto keyName = line.substr(0, middlePos);
	  auto value = line.substr(middlePos + 1, line.size());

	  if (keyName=="det_model_dir") {
		det_model_dir = value;
	  } else if (keyName=="rec_model_dir") {
		rec_model_dir = value;
	  } else if (keyName=="rec_char_dict_path") {
		rec_char_dict_path = value;
	  }
	}
  }
};

inline Args init_args() {
  Args args;
  args.image_path = "";

  args.det_algorithm = "DB";
//            args.det_model_dir = "../models/inference_det_v4.onnx";

  args.det_model_dir = "../models/inference_det_v4.onnx";

  args.det_limit_side_len = 960;
  args.det_limit_type = "max";
  args.det_db_thresh = 0.3;
  args.det_db_box_thresh = 0.7;
  args.det_db_unclip_ratio = 1.5;
  args.det_db_piexl = 3;
  args.max_candidates = 1000;
  args.det_db_score_mode = "fast";
  args.use_dilation = false;

  args.max_batch_size = 1;
  args.rec_algorithm = "CRNN";
  args.rec_model_dir = "../models/model_reg_v2_1b_mobile-sim-no-shape.onnx";
//  args.rec_image_shape = "3, 32, 1280";
  args.rec_image_shape = "3, 32, 1280";
  args.max_text_length = 25;
  args.rec_char_dict_path = "../words/ppocr_keys_v1.txt";
  args.use_space_char = true;
  args.vis_font_path = "./utils/doc/fonts/simfang.ttf";
  args.drop_score = 0.20;
  args.use_angle_cls = true;
  args.cls_model_dir = "";
  args.cls_image_shape = "3, 48, 192";
  args.label_list = {"0", "180"};
  args.cls_batch_num = 1;
  args.cls_thresh = 0.9;
  return args;
}

inline Args init_args_from_file(const std::string &file_path) {
  Args config = init_args();
  config.loadFormMap(file_path);
  return config;
}
}
}

#endif // PADDLEV2_INFERENCE_PREDICT_CONFIG_HPP
