Binarization_CUDA: main.cu
	nvcc main.cu -o binarization_gpu `pkg-config --cflags --libs opencv`