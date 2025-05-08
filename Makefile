# Compiler
NVCC = nvcc
CPP=g++

OPT?=-O3

CFLAGS= --std=c++11 -g -ggdb -gdwarf-3 $(OPT) -fsanitize=address
MODULE          := conv1 conv2 class1 class2
MODULE_NEW	 := new_conv_1 new_conv_2 new_conv_3 new_conv_4 new_conv_5 new_conv_6 new_conv_7 new_conv_8 new_conv_9

.PHONY: all clean

all: $(MODULE)
all_new: $(MODULE_NEW)

HEADERS=dnn.hpp

# Source and output files
SRC_CONV = convolution_optimized.cu
SRC_CONV_OLD = convolution.cu
OUT_CONV1 = conv1_optimized
OUT_CONV2 = conv2_optimized
OUT_CONV_OLD = conv_old


SRC_CLASS = classifier.cu
OUT_CLASS1 = class1
OUT_CLASS2 = class2
# Preprocessor definitions
DEFINES_CONV1 = -DNx=224 -DNy=224 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=4 -DTy=4 -DBLOCK_X=6 -DBLOCK_Y=6 -DBLOCK_Z=16

DEFINES_CONV2 = -DNx=14 -DNy=14 -DNi=512 -DKx=3 -DKy=3 -DNn=512 -DTii=64 -DTi=8 -DTnn=64 -DTn=8 -DTx=2 -DTy=2 -DBLOCK_X=4 -DBLOCK_Y=4 -DBLOCK_Z=32


DEFINES_CONV_OLD = -DNx=224 -DNy=224 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=16 -DTi=8 -DTnn=8 -DTn=2 -DTx=16 -DTy=16

DEFINES_CLASS1 = -DNi=25088 -DNn=4096  -DTii=512 -DTi=128     -DTnn=32  -DTn=16

DEFINES_CLASS2 = -DNi=4096 -DNn=1024   -DTii=512 -DTi=32     -DTnn=32  -DTn=16


DEFINES_NEW_CONV1 = -DNx=18 -DNy=18 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=2 -DTy=2 -DBLOCK_X=4 -DBLOCK_Y=4 -DBLOCK_Z=16

DEFINES_NEW_CONV2 = -DNx=18 -DNy=18 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=3 -DTy=3 -DBLOCK_X=5 -DBLOCK_Y=5 -DBLOCK_Z=16

DEFINES_NEW_CONV3 = -DNx=18 -DNy=18 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=18 -DTy=18 -DBLOCK_X=20 -DBLOCK_Y=20 -DBLOCK_Z=2

DEFINES_NEW_CONV4 = -DNx=70 -DNy=70 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=2 -DTy=2 -DBLOCK_X=4 -DBLOCK_Y=4 -DBLOCK_Z=16

DEFINES_NEW_CONV5 = -DNx=70 -DNy=70 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=7 -DTy=7 -DBLOCK_X=9 -DBLOCK_Y=9 -DBLOCK_Z=4

DEFINES_NEW_CONV6 = -DNx=70 -DNy=70 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=1 -DTi=8 -DTnn=8 -DTn=8 -DTx=10 -DTy=10 -DBLOCK_X=12 -DBLOCK_Y=12 -DBLOCK_Z=4

DEFINES_NEW_CONV7 = -DNx=198 -DNy=198 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=2 -DTy=2 -DBLOCK_X=4 -DBLOCK_Y=4 -DBLOCK_Z=16

DEFINES_NEW_CONV8 = -DNx=198 -DNy=198 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=3 -DTy=3 -DBLOCK_X=5 -DBLOCK_Y=5 -DBLOCK_Z=8

DEFINES_NEW_CONV9 = -DNx=198 -DNy=198 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=6 -DTy=6 -DBLOCK_X=8 -DBLOCK_Y=8 -DBLOCK_Z=4

DEFINES_NEW_CONV10 = -DNx=318 -DNy=318 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=2 -DTy=2 -DBLOCK_X=4 -DBLOCK_Y=4 -DBLOCK_Z=16

DEFINES_NEW_CONV11 = -DNx=318 -DNy=318 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=3 -DTy=3 -DBLOCK_X=5 -DBLOCK_Y=5 -DBLOCK_Z=8

DEFINES_NEW_CONV12 = -DNx=318 -DNy=318 -DNi=64 -DKx=3 -DKy=3 -DNn=64 -DTii=8 -DTi=8 -DTnn=8 -DTn=8 -DTx=6 -DTy=6 -DBLOCK_X=8 -DBLOCK_Y=8 -DBLOCK_Z=4


conv1:
	$(NVCC) $(SRC_CONV) -o $(OUT_CONV1) $(DEFINES_CONV1)

conv2:
	$(NVCC) $(SRC_CONV) -o $(OUT_CONV2) $(DEFINES_CONV2)

class1:
	$(NVCC) $(SRC_CLASS) -o $(OUT_CLASS1) $(DEFINES_CLASS1)

class2:
	$(NVCC) $(SRC_CLASS) -o $(OUT_CLASS2) $(DEFINES_CLASS2)


new_conv_1:
	$(NVCC) $(SRC_CONV) -o new_conv1 $(DEFINES_NEW_CONV1)

new_conv_2:
	$(NVCC) $(SRC_CONV) -o new_conv2 $(DEFINES_NEW_CONV2)

new_conv_3:
	$(NVCC) $(SRC_CONV) -o new_conv3 $(DEFINES_NEW_CONV3)

new_conv_4:
	$(NVCC) $(SRC_CONV) -o new_conv4 $(DEFINES_NEW_CONV4)

new_conv_5:
	$(NVCC) $(SRC_CONV) -o new_conv5 $(DEFINES_NEW_CONV5)

new_conv_6:
	$(NVCC) $(SRC_CONV) -o new_conv6 $(DEFINES_NEW_CONV6)

new_conv_7:
	$(NVCC) $(SRC_CONV) -o new_conv7 $(DEFINES_NEW_CONV7)

new_conv_8:
	$(NVCC) $(SRC_CONV) -o new_conv8 $(DEFINES_NEW_CONV8)

new_conv_9:
	$(NVCC) $(SRC_CONV) -o new_conv9 $(DEFINES_NEW_CONV9)

new_conv_10:
	$(NVCC) $(SRC_CONV) -o new_conv10 $(DEFINES_NEW_CONV10)

new_conv_11:
	$(NVCC) $(SRC_CONV) -o new_conv11 $(DEFINES_NEW_CONV11)

new_conv_12:
	$(NVCC) $(SRC_CONV) -o new_conv12 $(DEFINES_NEW_CONV12)


# Clean up
clean:
	rm -f $(OUT)
