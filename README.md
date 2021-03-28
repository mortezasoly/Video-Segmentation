#Video-Segmentation.

First make sure that LibTorch and OpenCV exist.

Second run cmake in following way to get Makefile:
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .

Third run make to get commandline application:
make

Fourth run ./segment for segmenting a video:
./segment <model_name.pt> <video_file>

for segmentation task you need a segmenting torch model saved as *.pt file.
