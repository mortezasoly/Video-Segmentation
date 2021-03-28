#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>
#include <ATen/ATen.h>


using namespace cv;

std::vector<double> norm_mean = {0.485, 0.456, 0.406};

std::vector<double> norm_std = {0.229, 0.224, 0.225};


int main(int argc, char** argv)
{

	auto model = torch::jit::load(argv[1]);
	std::vector<Vec3b> colors;
    for (size_t i = 0; i < 21; i++)
    {
       	int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

	VideoCapture capture(argv[2]);
	Mat image;
	int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	VideoWriter video("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(frame_width, frame_height));
	std::cout << "Eat my shit" << '\n';
	while (true)
	{
		capture >> image;
		if(image.empty())
			break;
		cvtColor(image, image, COLOR_BGR2RGB);
		image.convertTo(image, CV_32FC3, 1.0f/255.0f);
		torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows		,image.cols, 3}, c10::kFloat);

		tensor_image = tensor_image.permute({0,3,1,2});
		//image_tensor.unsqueeze_(0);
		tensor_image[0][0] = tensor_image[0][0].sub(0.485).div(0.229); 
    	tensor_image[0][1] = tensor_image[0][1].sub(0.456).div(0.224);
    	tensor_image[0][2] = tensor_image[0][2].sub(0.406).div(0.225);
		/*
		tensor_image = torch::data::transforms::Normalize<>(norm_mean, norm_std)(image_tensor);
		*/
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor_image);
		model.eval();
    	auto output = model.forward(inputs).toGenericDict();
		auto output_tensor = output.at("out").toTensor();
	
		auto results = output_tensor[0];
		auto predictions = results.argmax(0);

		
   		Mat markers(image.rows, image.cols, CV_8UC3);
		
		for (int i = 0; i < markers.rows; i++)
   		{
        	for (int j = 0; j < markers.cols; j++)
        	{
            	int index = predictions[i][j].item<int>();
            	markers.at<Vec3b>(i,j) = colors[index-1];
           		
       		}
    	}

    	video.write(markers);
	}
	
	capture.release();
	video.release();
	destroyAllWindows();
	
    std::cout << "Done!\n";	


	return 0;
}


