#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "predict_interface.h"


#if defined(_WIN32) || defined(_MSC_VER)
#include <io.h>                      //C (Windows)    access
#else
#include <unistd.h>
#endif

#include "predict_config.hpp"
#include "predict_detection.hpp"
#include "predict_recogntion.hpp"
#include "predict_algorithm.hpp"

void
read_image_and_show()
{
    std::string image_path = "../imgs/1rt.bmp";
    auto image_mat = cv::imread(image_path);
//    cv::imshow("picture", image_mat);
//    cv::waitKey(0);

}

void
read_net_form_onnx_format()
{
    auto net = cv::dnn::readNetFromONNX("../models/model_e_reg_600_1b.onnx");
    auto layers = net.getLayerNames();
    for (const auto &single_layer : layers)
    {
        OCR_OUT_STREAM << single_layer.c_str() << std::endl;
    }

    OCR_OUT_STREAM << "Model load end and success" << std::endl;

    std::string image_path = "../imgs/1rt.bmp";
    auto image_mat = cv::imread(image_path);
    cv::resize(image_mat, image_mat, cv::Size(600, 32));
    auto blob = cv::dnn::blobFromImage(image_mat);
    net.setInput(blob);
    auto prediction = net.forward();
    OCR_OUT_STREAM << "shibaile" << std::endl;
}

struct ModernOcr {
	std::iostream* outStream;
	modern_ocr::config::Args args;
	modern_ocr::detection::DBPreProcess preProcess;
	modern_ocr::detection::DBPostProcess dbPostProcess;
	modern_ocr::detection::TextDetector textDetector;
	modern_ocr::recogntion::TextRecognizer textRecognizer;
};

int  PaddleOCRInit(const char* model_det_path, const char* model_cls_path, const char* model_rec_path,
								  const char* model_key_words_path, const char* model_super_param_path, void** handle) {
	ModernOcr* ocr_implement = new ModernOcr();
	if(model_super_param_path)
	{
		if(access(model_super_param_path,0) == 0)
		{
			OCR_OUT_STREAM << "Load config form [" << model_super_param_path << "]" << std::endl;
			ocr_implement->args = modern_ocr::config::init_args_from_file(model_super_param_path);
		}
		else
		{
			OCR_OUT_STREAM << "Load config form [" << model_super_param_path << "] failure, because file not exist." << OCR_OUT_STREAM_END;
			goto HANDLE_FAILURE_ARGUMENT_INVALID;
		}
	}
	else
	{
		ocr_implement->args = modern_ocr::config::init_args();
		OCR_OUT_STREAM << "Load default config." << std::endl;
	}

	if(model_det_path)
	{
		ocr_implement->args.det_model_dir = model_det_path;
	}

	if(model_cls_path)
	{
		ocr_implement->args.cls_model_dir = model_cls_path;
	}

	if(model_rec_path)
	{
		ocr_implement->args.rec_model_dir = model_rec_path;
	}

	if(model_key_words_path)
	{
		ocr_implement->args.rec_char_dict_path = model_key_words_path;
	}

	OCR_OUT_STREAM<< "*****************************PARAM*****************************" << OCR_OUT_STREAM_END;
	OCR_OUT_STREAM << ocr_implement->args << OCR_OUT_STREAM_END;
	OCR_OUT_STREAM<< "***************************************************************" << OCR_OUT_STREAM_END;
	// load pre process operator
	ocr_implement->preProcess = modern_ocr::detection::DBPreProcess(ocr_implement->args);
	// load post process operator
	ocr_implement->dbPostProcess = modern_ocr::detection::DBPostProcess(ocr_implement->args);

	// load detection from args.rec_model_dir
	if(access(ocr_implement->args.rec_model_dir.data(), 0) == 0)
	{
		ocr_implement->textDetector = modern_ocr::detection::TextDetector(ocr_implement->args,
																		  ocr_implement->preProcess,
																		  ocr_implement->dbPostProcess);

		if(ocr_implement->textDetector.isModelLoadStatus())
		{
			OCR_OUT_STREAM << "Load detection model form [" << ocr_implement->args.det_model_dir << "] done." << std::endl;
		}
		else
		{
			OCR_OUT_STREAM << "Load detection model form [" << ocr_implement->args.det_model_dir << "] failure." << std::endl;
			goto HANDLE_FAILURE;
		}
	}
	else
	{
		OCR_OUT_STREAM << "Load detection model form [" << ocr_implement->args.det_model_dir << "] failure, because file not exist." << OCR_OUT_STREAM_END;
		goto HANDLE_FAILURE_ARGUMENT_INVALID;
	}

	// load recognizer from args.rec_model_dir
	if(access(ocr_implement->args.rec_model_dir.data(), 0) == 0)
	{
		ocr_implement->textRecognizer = modern_ocr::recogntion::TextRecognizer(ocr_implement->args);
		if(ocr_implement->textRecognizer.isModelLoadStatus())
		{
			OCR_OUT_STREAM << "Load detection model form [" << ocr_implement->args.rec_model_dir << "] done." << std::endl;
		}
		else
		{
			OCR_OUT_STREAM << "Load detection model form [" << ocr_implement->args.rec_model_dir << "] failure." << std::endl;
			goto HANDLE_FAILURE;
		}
	}
	else
	{
		OCR_OUT_STREAM << "Load detection model form [" << ocr_implement->args.rec_model_dir << "] failure, because file not exist." << OCR_OUT_STREAM_END;
		goto HANDLE_FAILURE_ARGUMENT_INVALID;
	}

	*handle = ocr_implement;
	return 0;

	HANDLE_FAILURE_ARGUMENT_INVALID:
	delete ocr_implement;
	handle = NULL;
	return -2;

	HANDLE_FAILURE:
	delete ocr_implement;
	handle = NULL;
	return -1;
}


void textCopy(const std::vector<std::pair<std::string, float>>& ori_text_list, PaddleForwardOut& paddleForwardOut)
{
	paddleForwardOut.length = 0;
	for (const auto& line: ori_text_list)
	{
		paddleForwardOut.length += line.first.size();
	}

	paddleForwardOut.content = new char[paddleForwardOut.length+10]();
	paddleForwardOut.length = 0;
	for (const auto& line: ori_text_list)
	{
		memcpy(paddleForwardOut.content + paddleForwardOut.length, line.first.data(), line.first.size());
		paddleForwardOut.length += line.first.size();
	}
}

int PaddleOCRForwardImplement(PaddleForwardOut& paddleForwardOut,ModernOcr* modernOCR) try
{
	// read image form disk, must 3 channels;
	cv::Mat img = cv::imread(modernOCR->args.image_path,cv::IMREAD_COLOR);
	std::vector<std::map<std::string, std::vector<cv::Rect>>> boxes;
	std::vector<std::pair<std::string, float>> text_list;
	std::vector<cv::Rect> ori_rect;
	std::vector<cv::Mat> crop_image;
	if(img.empty())
	{
		OCR_OUT_STREAM << "Img " << modernOCR->args.image_path <<  " read error." << OCR_OUT_STREAM_END;
		goto PaddleOCRForward_ARGUMENT_INVALID;
	}

	boxes = modernOCR->textDetector(img);
	if(boxes.empty())
	{
		return 0;
	}

	std::tie(ori_rect, crop_image) = modern_ocr::algorithm::preprocess_boxes(boxes, img);
	text_list = modernOCR->textRecognizer(crop_image);
	textCopy(text_list, paddleForwardOut);
	return 0;

PaddleOCRForward_ARGUMENT_INVALID:
	delete paddleForwardOut.content;
	paddleForwardOut.content = 0;
	paddleForwardOut.length = 0;
	paddleForwardOut.error_code = -2;
	return -1;

}
catch (std::exception& error)
{
	OCR_OUT_STREAM << error.what() << OCR_OUT_STREAM_END;
	delete paddleForwardOut.content;
	paddleForwardOut.content = 0;
	paddleForwardOut.length = 0;
	paddleForwardOut.error_code = -2;
	return -1;
}
catch (...)
{
	OCR_OUT_STREAM << "Recvied a exception from PaddleOCRForwardImplement, and exception type unknown." << OCR_OUT_STREAM_END;
	delete paddleForwardOut.content;
	paddleForwardOut.content = 0;
	paddleForwardOut.length = 0;
	paddleForwardOut.error_code = -2;
	return -1;
}

PaddleForwardOut PaddleOCRForward(void* handle, const char* image_path) {
	PaddleForwardOut prediction;
	prediction.content = 0;
	ModernOcr* ocr_implement = nullptr;

	if (handle == NULL)
	{
		OCR_OUT_STREAM << "Handle is NULL, argument invalid" << OCR_OUT_STREAM_END;
		goto PaddleOCRForward_ARGUMENT_INVALID;
	}
	else
	{
		ocr_implement = (ModernOcr*)handle;
	}

	if(image_path == NULL)
	{
		OCR_OUT_STREAM << "img is NULL, argument invalid" << OCR_OUT_STREAM_END;
		goto PaddleOCRForward_ARGUMENT_INVALID;
	}
	else
	{
		if(access(image_path, 0) != 0)
		{
			OCR_OUT_STREAM << "img is not exist, argument invalid" << OCR_OUT_STREAM_END;
			goto PaddleOCRForward_ARGUMENT_INVALID;
		}
	}

	ocr_implement->args.image_path = image_path;
	PaddleOCRForwardImplement(prediction, ocr_implement);
	return prediction;

	PaddleOCRForward_ARGUMENT_INVALID:
	prediction.content = 0;
	prediction.length = 0;
	prediction.error_code = -1;
	return prediction;
}

void PaddleOCRUnInit(void* handle)
{
	if (handle)
	{
		delete (ModernOcr*)handle;
	}
}



#ifdef PADDLE_OCR_EXPORT_MAIN

int
main()
{
    modern_ocr::config::Args args = modern_ocr::config::init_args();
    modern_ocr::detection::DBPreProcess preProcess(args);
    modern_ocr::detection::DBPostProcess dbPostProcess(args);

    args.image_path = "../imgs/samples1/07_图片测试.tiff";
    auto img = cv::imread(args.image_path);

	if(img.empty()) {
		std::cout << "Image read error, Mat is empty, maybe image type unsupported!" << std::endl;
		return 0;
	}
    modern_ocr::detection::TextDetector textDetector(args, preProcess, dbPostProcess);
    auto boxes = textDetector(img);
    OCR_OUT_STREAM << "***************************************************************************************" << OCR_OUT_STREAM_END;

#if 0
    for (auto& mm: boxes) {
        for (const auto& bb: mm["points"]) {
            cv::rectangle(img, bb, (0,0,255), 2);
            OCR_OUT_STREAM << "bb shape [" << bb << "]" << std::endl;
        }

    }
	cv::imshow("pict", img);
	//    cv::imwrite("./result.png", img);
	cv::waitKey(0);
#endif

    std::vector<cv::Rect> ori_rect;
    std::vector<cv::Mat> crop_image;
    std::tie(ori_rect, crop_image) = modern_ocr::algorithm::preprocess_boxes(boxes, img);
//    for (const auto& img_c : crop_image) {
//      cv::imshow("pict", img_c);
//      //    cv::imwrite("./result.png", img);
//      cv::waitKey(0);
//    }
//    cv::imshow("pict", img);
//    cv::imwrite("./result.png", img);
//    cv::waitKey(0);
    OCR_OUT_STREAM << "***************************************************************************************" << OCR_OUT_STREAM_END;
    modern_ocr::recogntion::TextRecognizer textRecognizer(args);
    auto text_list = textRecognizer(crop_image);

    for (const auto& line : text_list) {
      OCR_OUT_STREAM << line.first << "\t" << line.second << std::endl;
    }

    OCR_OUT_STREAM << "inference end" << std::endl;

	return 0;
}

#endif
