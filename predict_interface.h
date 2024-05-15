//
// Created by gengpeng on 2024/4/28.
//

#ifndef PADDLEV2_INFERENCE__PREDICT_INTERFACE_H
#define PADDLEV2_INFERENCE__PREDICT_INTERFACE_H

#ifdef USE_STDCOUT_STREAM
#include <iostream>
#define OCR_OUT_STREAM std::cout
#define OCR_OUT_STREAM_END std::endl
#else
#define OCR_OUT_STREAM std::cout
#define OCR_OUT_STREAM_END ""
#endif

#if defined(_WIN32) || defined(_MSC_VER)
#ifdef PADDLE_OCR_EXPORT
#define PADDLE_OCR_API __declspec(dllexport)
#else
#define PADDLE_OCR_API __declspec(dllimport)
#endif
#elif defined(__linux__) || defined(__APPLE__)
#ifdef DEEPNETAPI_EXPORT
#define PADDLE_OCR_API __attribute__ ((visibility ("default")))
#else
#define PADDLE_OCR_API
#endif // DEEPNETAPI_EXPORT
#endif


struct PADDLE_OCR_API PaddleForwardOut{
	char* content;
	unsigned int length;
	unsigned int error_code;
};

/**
 * OCR 初始化操作，在此期间，OCR会加载三个神经网络模型， 从model_super_param_path加载超参等。
 * 如果指定model_super_param_path，将从model_super_param_path加载超参，其中包含模型路径，
 * 如果指定model_det_path model_cls_path model_rec_path，则这三个路径将会覆盖model_super_param_path
 * 中设定的路径； 如果不设定model_super_param_path， 则会加载默认的配置参数，如果指定三个模型的路径，则会覆盖
 * 模型的默认加载路径。
 * @param model_det_path 指向框选推理模型的路径
 * @param model_cls_path 指向分类推理模型的路径
 * @param model_rec_path 指向识别推理模型的路径
 * @param model_super_param_path 指向超参配置文件路径
 * @return 返回handle，类型void*，  包含三个推理模型和超参配置等信息
 */
int  PADDLE_OCR_API PaddleOCRInit(const char* model_det_path,
								  const char* model_cls_path,
								  const char* model_rec_path,
								  const char* model_key_words_path,
								  const char* model_super_param_path,
								  void** handle);

/**
 *
 * @param handle
 * @param image_path
 * @return
 */
PADDLE_OCR_API PaddleForwardOut PaddleOCRForward(void* handle,
												 const char* image_path);

/**
 *
 * @param handle
 */
PADDLE_OCR_API void PaddleOCRUnInit(void* handle);







#endif //PADDLEV2_INFERENCE__PREDICT_INTERFACE_H
