//////////////////////////////////////////////////////////////////////////////
// OpenCL project: Group 4, Topics - dilation, erosion, opening, closing, Gaussian blurr ( mask dimension - 3x3 and 5x5 ) 
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>
#include<algorithm>
#include <stdlib.h>


#include <boost/lexical_cast.hpp>


//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////

const int msize1 = 3; // size of gaussian mask 1 
const int msize2 = 5; // size of gaussian mask 2

float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[j * countX + i];
}
int getValueMask(const std::vector<int>& b, std::size_t row, std::size_t column, int i, int j)
{
	if (i < 0 || (size_t)i >= row || j < 0 || (size_t)j >= column)
		return 0;
	else
		return b[j * row + i];

}
//function - compares the result of CPU and GPU 
bool compareResult(const std::vector<float>& hOutputCPU, const std::string& outputCPU, const std::vector<float>& hOutputGPU, const std::string& outputGPU, std::size_t countX, std::size_t countY) {
	std::size_t errorCount = 0;
	for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			size_t index = i + j * countX;
			// Allow small differences between results (due to rounding)
			if (!(std::abs(hOutputCPU[index] - hOutputGPU[index]) <= 1e-2)) {
				if (errorCount < 15)
					std::cout << "Result for " << i << "," << j << " is incorrect: " << outputCPU << " value is " << hOutputCPU[index] << ", " << outputGPU << " value is " << hOutputGPU[index] << std::endl;
				else if (errorCount == 15)
					std::cout << "..." << std::endl;
				errorCount++;
			}
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return false;
	}
	return true;
}

// function - convert 2D matrix into 1D array- required for a gaussian mask
void twoDTOoneD(float *array2d, float* array1d, int row, int column)
{
	int i, j;

	for (i = 0; i < row; i++)
	{
		for (j = 0; j < column; j++)
		{
			array1d[i * row + j] = *((array2d + i * column) + j);
		}
	
	}
	
}
// Function - This function perfoms the dilation operation
void dilation(const std::vector<float>& h_input, 
	std::vector<int> const& h_structure_element,
	std::vector<float>& h_outputDilationCpu, 
	std::size_t countX, 
	std::size_t countY) {


	float maximum = 0.0f;
	std::size_t mask_size = 3;
	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			maximum = 0.0f;
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					float pixelValue = getValueGlobal(h_input, countX, countY, x + i, y + j);
					int mask_element = getValueMask(h_structure_element, mask_size, mask_size, x + 1, y + 1);
					if (mask_element == 1)
					{

						if (maximum > pixelValue)
							maximum = maximum;
						else
							maximum = pixelValue;
					}

				}
			}
			h_outputDilationCpu[j * countX + i] = maximum;
		}
	}
}
// Function - This function perfoms the erosion operation 
void erosion(const std::vector<float>& h_input, 
	std::vector<int>& h_structure_element,
	std::vector<float>& h_outputErosionCpu, 
	std::size_t countX, 
	std::size_t countY) {
	float minimum = 1.0f;
	std::size_t mask_size = 3;
	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			minimum = 1.0f;
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					float pixelValue = getValueGlobal(h_input, countX, countY, x + i, y + j);
					int mask_element = getValueMask(h_structure_element, mask_size, mask_size, x + 1, y + 1);
					if (mask_element == 1)
					{

						if (minimum < pixelValue)
							minimum = minimum;
						else
							minimum = pixelValue;
					}

				}
			}
			h_outputErosionCpu[j * countX + i] = minimum;
		}
	}
}
// Function - This function perfoms the opening operation.
void opening(const std::vector<float>& h_input, 
	std::vector<int>& h_structure_element, 
	std::vector<float>& h_outputOpeningCpu, 
	std::size_t countX, 
	std::size_t countY)
{
	std::size_t count1 = countX * countY;
	std::vector<float> h_temp(count1);
	erosion(h_input, h_structure_element,h_temp, countX, countY);
	dilation(h_temp, h_structure_element, h_outputOpeningCpu, countX, countY);
}
// Function - This function perfoms the closing operation.
void closing(const std::vector<float>& h_input, 
	std::vector<int>& h_structure_element, 
	std::vector<float>& h_outputClosingCpu, 
	std::size_t countX, 
	std::size_t countY)
{
	std::size_t count1 = countX * countY;
	std::vector<float> h_temp1(count1);
	dilation(h_input, h_structure_element, h_temp1, countX, countY);
	erosion(h_temp1, h_structure_element, h_outputClosingCpu, countX, countY);
}

// Function - This function creates the Gaussian mask for mask size = 3x3 & 5x5.
void Gaussian_mask(float* mask, float sigma, int msize)
{

	float r = 0.0;
	float s = 2.0 * sigma * sigma;
	float sum = 0.0;
	float sigma_sqr = (sigma) * (sigma);
	int a = (msize - 1) / 2;
	int b = (msize - 1) / 2;

    for (int i = -a; i <= a; i++)
	{
		for (int j = -b; j <= b; j++)
		{
			r = (float)(i) * (float)(i)+(float)(j) * (float)(j);
			*((mask + (i+a) * msize) + (j+b)) = (float)((exp(-(r) / (2 * sigma_sqr))) / (2 * 3.14 * sigma_sqr));
			sum = sum + *((mask + (i + a) * msize) + (j + b));
		}
	}
	// normalize the mask 
	for (int i = 0; i < msize; ++i)
	{
		for (int j = 0; j < msize; ++j)
		{
			*((mask + (i) * msize) + (j)) /= sum;
		}
	}


}
// Function - This function performs the Gaussian alogrithm (Filter operation)
void Gaussian_filter(const std::vector<float>& h_input,
	std::vector<float>& h_GaussianOut,
	std::vector<float>& h_GaussianMask,
	std::size_t countX,
	std::size_t countY,
	int msize
	)
{   
	float sum = 0.0;
	int a = (msize - 1) / 2;
	int b = (msize - 1) / 2;
	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			sum = 0.0;
			for (int x = -a; x <=a; x++)
			{
				for (int y = -b; y <=b; y++)
				{   
					float pixelValue = getValueGlobal(h_input, countX, countY, x+i, y+j);
					sum += pixelValue * getValueGlobal(h_GaussianMask,(size_t)msize, (size_t)msize, x + a, y + b);
					
				}
			}
			h_GaussianOut[j * countX + i] = sum;
		}

	}
}
//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
using namespace std;
int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);


	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT(deviceNr > 0);
	ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "F:/gpu/Opencl-ex1/Opencl-ex1/src/OpenCLProject_Group4.cl");
	//cl::Program program = OpenCL::loadProgramSource(context, "../Opencl-ex1/Opencl-ex1/src/OpenCLProject_Group4.cl");

	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values .
	std::size_t wgSizeX = 20; // Number of work items per work group in X direction
	std::size_t wgSizeY = 20;
	std::size_t countX = wgSizeX; // Overall number of work items in X direcstion = Number of elements in X direction
	std::size_t countY = wgSizeY;
	std::size_t count = countX * countY; // Overall number of elements
	// Number of elements in structuring element(structuring element is used for dilation, erosion, opeing and closing)
	std::size_t countA =(size_t)(msize1 * msize1);
	// Number of elements in Gaussian mask of dimension 3x3
	std::size_t countB = (size_t)(msize1 * msize1);
	// Number of elements in Gaussian mask of dimension 5x5
    std:size_t countD = (size_t)(msize2 * msize2);
	std::size_t size = count * sizeof(float); // Size of data in bytes
	std::size_t sizeA = countA * sizeof(int);
	std::size_t sizeB = countB * sizeof(float);
	std::size_t sizeD = countD * sizeof(float);
	
	// structuring element of size 3x3 for dilation, erosion, opening and closing (2D)
	std::vector<std::vector<int>> structure_element_2d{
				{ 0, 1, 0},
				{ 1, 1, 1},
				{ 0, 1, 0}
	};
	// structuring element of size 3x3 for dilation, erosion, opening and closing (1D)
	int structure_element_1d[msize1 * msize1];
	// Gaussian mask of size 3x3 for (1D)
	float gaussian_mask1_1d[msize1 * msize1];
	// Gaussian mask of size 5x5 for (1D)
	float gaussian_mask2_1d[msize2 * msize2];

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input(count);
	std::vector<float> h_outputDilationCpu(count);
	std::vector<float> h_outputErosionCpu(count);

	std::vector<float> h_outputOpeningCpu(count);
	std::vector<float> h_outputClosingCpu(count);

	std::vector<float> h_outputDilationGpu(count);
	std::vector<float> h_outputErosionGpu(count);

	std::vector<float> h_outputOpeningGpu(count);
	std::vector<float> h_outputClosingGpu(count);

	std::vector<float> h_temp1(count);
	std::vector<float> h_temp2(count);

	std::vector<float> h_Gaussian1Out(count);
	std::vector<float> h_Gaussian1OutGpu(count);

	std::vector<float> h_Gaussian2Out(count);
	std::vector<float> h_Gaussian2OutGpu(count);

	std::vector<float> h_GaussianMask1(countB);
	std::vector<float> h_GaussianMask2(countD);

	std::vector<int> h_structure_element(countA);
	
	
	float mask1[msize1][msize1]; //declare gaussian mask of size 3x3
	float mask2[msize2][msize2]; //declare gaussian mask of size 5x5


	// Allocate space for input and output data on the device
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_structure_element(context, CL_MEM_READ_WRITE, sizeA);
	cl::Buffer d_gaussianMask1(context, CL_MEM_READ_WRITE, sizeB);
	cl::Buffer d_gaussianMask2(context, CL_MEM_READ_WRITE, sizeD);
	cl::Buffer d_outputDilation(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputErosion(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputOpening(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputClosing(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_temp1(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_temp2(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputGaussian1Gpu(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputGaussian2Gpu(context, CL_MEM_READ_WRITE, size);
	
	
	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
    memset(h_structure_element.data(), 255, sizeA);

	memset(h_GaussianMask1.data(), 255, sizeB);
	memset(h_GaussianMask2.data(), 255, sizeD);

	memset(h_outputDilationCpu.data(), 255, size);
	memset(h_outputErosionCpu.data(), 255, size);

	memset(h_outputOpeningCpu.data(), 255, size);
	memset(h_outputClosingCpu.data(), 255, size);

	memset(h_outputDilationGpu.data(), 255, size);
	memset(h_outputErosionGpu.data(), 255, size);

	memset(h_outputOpeningGpu.data(), 255, size);
	memset(h_outputClosingGpu.data(), 255, size);

	memset(h_temp2.data(), 255, size);
	memset(h_temp1.data(), 255, size);
	
	memset(h_Gaussian1Out.data(), 255, size);
	memset(h_Gaussian1OutGpu.data(), 255, size);

	memset(h_Gaussian2Out.data(), 255, size);
	memset(h_Gaussian2OutGpu.data(), 255, size);

	
	//GPU
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_structure_element, true, 0, sizeA, h_structure_element.data());

	queue.enqueueWriteBuffer(d_gaussianMask1, true, 0, sizeB, h_GaussianMask1.data());
	queue.enqueueWriteBuffer(d_gaussianMask2, true, 0, sizeD, h_GaussianMask2.data());

	queue.enqueueWriteBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data());
	queue.enqueueWriteBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data());
	queue.enqueueWriteBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data());
	queue.enqueueWriteBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data());

	queue.enqueueWriteBuffer(d_temp1, true, 0, size, h_temp1.data());
	queue.enqueueWriteBuffer(d_temp2, true, 0, size, h_temp2.data());

	queue.enqueueWriteBuffer(d_outputGaussian1Gpu, true, 0, size, h_Gaussian1OutGpu.data());
	queue.enqueueWriteBuffer(d_outputGaussian2Gpu, true, 0, size, h_Gaussian2OutGpu.data());

	

	//////// Load input data ////////////////////////////////
	
	std::vector<float> inputData;
	std::size_t inputWidth, inputHeight;
	Core::readImagePGM("F:/gpu/Opencl-ex1/Opencl-ex1/data.pgm", inputData, inputWidth, inputHeight);
	for (size_t j = 0; j < countY; j++) {
		for (size_t i = 0; i < countX; i++) {
			h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
		}
	}
	

	// convert 2D structuring element into 1D structing element. 
    for (std::size_t i = 0; i < (countA)/3; i++)
	{
		for (std::size_t j = 0; j < (countA)/3; j++)
		{
			h_structure_element[i * ((countA) / 3 )+ j] = structure_element_2d[i][j];
		}

	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// calling Gaussian_mask function to create 3x3 and 5x5 mask
	float sigma = 1.5f; // init value of sigma for gaussian filter operation. 
	// 3x3
	Gaussian_mask((float *)mask1,sigma, msize1);
	twoDTOoneD((float*)mask1, gaussian_mask1_1d, msize1, msize1); // output is 1d array 
	h_GaussianMask1.assign(begin(gaussian_mask1_1d), end(gaussian_mask1_1d)); // filling entries of a vector 
	// 5x5
	Gaussian_mask((float*)mask2, sigma, msize2);
	twoDTOoneD((float*)mask2, gaussian_mask2_1d, msize2, msize2);  // output is 1d array 
	h_GaussianMask2.assign(begin(gaussian_mask2_1d), end(gaussian_mask2_1d)); // filling entries of a vector 
	///////////////////////////////////////////////////////////////////////////////////////////////

	
	// Do calculation on the host side
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	dilation(h_input, h_structure_element, h_outputDilationCpu, countX, countY);
	erosion(h_input, h_structure_element, h_outputErosionCpu, countX, countY);
	opening(h_input, h_structure_element, h_outputOpeningCpu, countX, countY);
	closing(h_input, h_structure_element, h_outputClosingCpu, countX, countY);
	Gaussian_filter(h_input, h_Gaussian1Out, h_GaussianMask1, countX, countY, msize1);
	Gaussian_filter(h_input, h_Gaussian2Out, h_GaussianMask2, countX, countY, msize2);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();

	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("output_dilation_cpu_group4.pgm", h_outputDilationCpu, countX, countY);
	Core::writeImagePGM("output_erosion_cpu_group4.pgm", h_outputErosionCpu, countX, countY);
	Core::writeImagePGM("output_opening_cpu_group4.pgm", h_outputOpeningCpu, countX, countY);
	Core::writeImagePGM("output_closing_cpu_group4.pgm", h_outputClosingCpu, countX, countY); 
	Core::writeImagePGM("output_gaussian1_cpu_group4.pgm", h_Gaussian1Out, countX, countY);
	Core::writeImagePGM("output_gaussian2_cpu_group4.pgm", h_Gaussian2Out, countX, countY);
	std::cout << std::endl;
	// Iterate over all implementations
	for(int impl =1 ; impl <=2; impl++)
	{ 

		std::cout << "Implementation #" << impl << ":" << std::endl;
		//Reinitialize output memory to 0xff
		memset(h_outputDilationGpu.data(), 255, size);
		memset(h_outputErosionGpu.data(), 255, size);
		memset(h_outputOpeningGpu.data(), 255, size);
		memset(h_outputClosingGpu.data(), 255, size);
		memset(h_Gaussian1OutGpu.data(), 255, size);
		memset(h_Gaussian2OutGpu.data(), 255, size);


		//GPU
		queue.enqueueWriteBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data());
		queue.enqueueWriteBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data());
		queue.enqueueWriteBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data());
		queue.enqueueWriteBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data());
		queue.enqueueWriteBuffer(d_outputGaussian1Gpu, true, 0, size, h_Gaussian1OutGpu.data());
		queue.enqueueWriteBuffer(d_outputGaussian2Gpu, true, 0, size, h_Gaussian2OutGpu.data());

		// Copy input data to device
		cl::Event copy1[3];
		cl::Image2D image;

		image = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
		cl::size_t<3> origin;
		origin[0] = origin[1] = origin[2] = 0;
		cl::size_t<3> region;
		region[0] = countX;
		region[1] = countY;
		region[2] = 1;
		queue.enqueueWriteImage(image, true, origin, region, countX * sizeof(float), 0, h_input.data(), NULL, &copy1[0]);
	
		queue.enqueueWriteBuffer(d_structure_element, true, 0, sizeA, h_structure_element.data(), NULL, &copy1[1]);
	
		if(impl == 1)
			queue.enqueueWriteBuffer(d_gaussianMask1, true, 0, sizeB, h_GaussianMask1.data(), NULL, &copy1[2]);
		else
			queue.enqueueWriteBuffer(d_gaussianMask2, true, 0, sizeD, h_GaussianMask2.data(), NULL, &copy1[2]);


		// Create a kernel object
		std::string kernelName1 = "dilation";
		std::string kernelName2 = "erosion";
		std::string kernelName3 = "opening";
		std::string kernelName4 = "closing";
		std::string kernelName5 = "gaussian" + boost::lexical_cast<std::string> (impl);
	
		cl::Kernel dilation(program, kernelName1.c_str());
		cl::Kernel erosion(program, kernelName2.c_str());
		cl::Kernel opening(program, kernelName3.c_str());
		cl::Kernel closing(program, kernelName4.c_str());
		cl::Kernel gaussian(program, kernelName5.c_str());

		// Launch kernel on the device
		cl::Event execution[5];

		dilation.setArg<cl::Image2D>(0, image);
		dilation.setArg<cl::Buffer>(1, d_outputDilation);
		dilation.setArg<cl::Buffer>(2, d_structure_element);
		queue.enqueueNDRangeKernel(dilation, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[0]);

		erosion.setArg<cl::Image2D>(0, image);
		erosion.setArg<cl::Buffer>(1, d_outputErosion);
		erosion.setArg<cl::Buffer>(2, d_structure_element);
		queue.enqueueNDRangeKernel(erosion, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[1]);

		opening.setArg<cl::Image2D>(0, image);
		opening.setArg<cl::Buffer>(1, d_outputOpening);
		opening.setArg<cl::Buffer>(2, d_temp1);
		opening.setArg<cl::Buffer>(3, d_structure_element);
		queue.enqueueNDRangeKernel(opening, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[2]);

		closing.setArg<cl::Image2D>(0, image);
		closing.setArg<cl::Buffer>(1, d_outputClosing);
		closing.setArg<cl::Buffer>(2, d_temp2);
		closing.setArg<cl::Buffer>(3, d_structure_element);
		queue.enqueueNDRangeKernel(closing, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[3]);

		gaussian.setArg<cl::Image2D>(0, image);
		if (impl == 1) // 3x3 implementation
		{
			gaussian.setArg<cl::Buffer>(1, d_outputGaussian1Gpu);
			gaussian.setArg<cl::Buffer>(2, d_gaussianMask1);
		}
		if (impl == 2) // 5x5 implementation
		{
			gaussian.setArg<cl::Buffer>(1, d_outputGaussian2Gpu);
			gaussian.setArg<cl::Buffer>(2, d_gaussianMask2);
		}
		queue.enqueueNDRangeKernel(gaussian, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution[4]);

	

		// Copy output data back to host
		cl::Event copy2[5];
		queue.enqueueReadBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data(), NULL, &copy2[0]);
		queue.enqueueReadBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data(), NULL, &copy2[1]);
		queue.enqueueReadBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data(), NULL, &copy2[2]);
		queue.enqueueReadBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data(), NULL, &copy2[3]);
		if(impl ==1 )
			queue.enqueueReadBuffer(d_outputGaussian1Gpu, true, 0, size, h_Gaussian1OutGpu.data(), NULL, &copy2[4]);
		if(impl == 2)
			queue.enqueueReadBuffer(d_outputGaussian2Gpu, true, 0, size, h_Gaussian2OutGpu.data(), NULL, &copy2[4]);
		// Print performance data
		Core::TimeSpan cpuTime = cpuEnd - cpuStart;

		Core::TimeSpan gpuTime = Core::TimeSpan::fromSeconds(0);
		for (std::size_t i = 0; i < sizeof(execution) / sizeof(*execution); i++)
			gpuTime = gpuTime + OpenCL::getElapsedTime(execution[i]);

		Core::TimeSpan copytime2 = Core::TimeSpan::fromSeconds(0);
		for (std::size_t i = 0; i < sizeof(copy2) / sizeof(*copy2); i++)
			copytime2 = copytime2 + OpenCL::getElapsedTime(copy2[i]);

		Core::TimeSpan copytime1 = Core::TimeSpan::fromSeconds(0);
		for (std::size_t i = 0; i < sizeof(copy1) / sizeof(*copy1); i++)
			copytime1 = copytime1 + OpenCL::getElapsedTime(copy1[i]);

		Core::TimeSpan copyTime = copytime1 + copytime2;
		Core::TimeSpan overallGpuTime = gpuTime + copyTime;
		std::cout << "CPU Time: " << cpuTime.toString() << ", " << (count / cpuTime.getSeconds() / 1e6) << " MPixel/s" << std::endl;;
		std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
		std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ", " << (count / gpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;
		std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ", " << (count / overallGpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;

		//////// Store GPU output image ///////////////////////////////////
		Core::writeImagePGM("output_dilation_gpu_group4.pgm", h_outputDilationGpu, countX, countY);
		Core::writeImagePGM("output_erosion_gpu_group4.pgm", h_outputErosionGpu, countX, countY);
		Core::writeImagePGM("output_opening_gpu_group4.pgm", h_outputOpeningGpu, countX, countY);
		Core::writeImagePGM("output_closing_gpu_group4.pgm", h_outputClosingGpu, countX, countY);
		if(impl == 1)
			Core::writeImagePGM("output_gaussian1_gpu_group4.pgm", h_Gaussian1OutGpu, countX, countY);
		if(impl == 2)
			Core::writeImagePGM("output_gaussian2_gpu_group4.pgm", h_Gaussian2OutGpu, countX, countY);

		// compare the result of CPU and GPU
		if (!compareResult(h_outputDilationCpu, "DilationCPU", h_outputDilationGpu, "DilationGpu", countX, countY)) return 1;
		if (!compareResult(h_outputErosionCpu, "ErosionCPU", h_outputErosionGpu, "ErosionGpu", countX, countY)) return 1;
		if (!compareResult(h_outputOpeningCpu, "OpeningCPU", h_outputOpeningGpu, "OpeningGpu", countX, countY)) return 1;
		if (!compareResult(h_outputClosingCpu, "ClosingCPU", h_outputClosingGpu, "ClosingGpu", countX, countY)) return 1;
		if (impl == 1)
			if (!compareResult(h_Gaussian1Out, "GaussianCPU3x3", h_Gaussian1OutGpu, "GaussianGpu3x3", countX, countY)) return 1;
		if (impl == 2)
			if (!compareResult(h_Gaussian2Out, "GaussianCPU5x5", h_Gaussian2OutGpu, "GaussianGpu5x5", countX, countY)) return 1;

	
	}
	std::cout << "Success" << std::endl;
	
	return 0;
}