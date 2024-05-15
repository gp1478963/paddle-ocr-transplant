//
// Created by gp147 on 2024/4/29.
//

#include "predict_interface.h"
#if defined(_WIN32) || defined(_MSC_VER)
#include <io.h>                      //C (Windows)    access
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <stdlib.h>
int main(int argc, char** argv)
{
	OCR_OUT_STREAM << "Hello, World!" << OCR_OUT_STREAM_END;
	std::string img_path;
	if(argc > 1 && argv[1])
	{
		img_path = argv[1];
	}
	else
	{
		OCR_OUT_STREAM << "Please input a image." <<OCR_OUT_STREAM_END;
		return -1;
	}
	OCR_OUT_STREAM << "Image:" << img_path.data() <<OCR_OUT_STREAM_END;

	std::string configPath;
	if(argc > 2 && argv[2])
	{
		configPath = argv[2];
	}

	const char* config_path = nullptr;
	if(!configPath.empty() && access(configPath.data(), 0) == 0)
	{
		config_path = configPath.data();
	}
	if(config_path == nullptr)
	{
		// 此行代码用于clion调试用
		if(access("../predict_config.ini",0) == 0)
		{
			config_path = "../predict_config.ini";
		}
		else if(access("./predict_config.ini",0) == 0)
		{
			config_path = "./predict_config.ini";
		}
	}

	if(config_path !=nullptr)
	{
		OCR_OUT_STREAM << "Config :[" << config_path << "]" << OCR_OUT_STREAM_END;
	}

	void* OCR_handle;
 	int retVal = PaddleOCRInit(NULL, NULL, NULL,NULL, config_path, &OCR_handle);
	switch (retVal)
	{
		case -1:
			OCR_OUT_STREAM << "Model init failure, may be model format incompatible, please use onnx model." << OCR_OUT_STREAM_END;
			break;
		case -2:
			OCR_OUT_STREAM << "Argument fetch failure, params incompatible, please use correct params." << OCR_OUT_STREAM_END;
			break;
		default:
			OCR_OUT_STREAM << "Model init properly." << OCR_OUT_STREAM_END;
			break;
	}

	PaddleForwardOut predict = PaddleOCRForward(OCR_handle, img_path.data());
	if(predict.length && predict.content)
	{
		OCR_OUT_STREAM << predict.content << std::endl;
	}
	OCR_OUT_STREAM << "Hello, World End!" << OCR_OUT_STREAM_END;
//	system("pause");
//	getchar();
	return 0;
}



