NVIDIA_COMPILER := $(shell nvcc --version 2>/dev/null)
CLIBS   	:=-lv4l2 -lv4lconvert -lavcodec -lavutil -lavformat -lswresample -lswscale

ifdef NVIDIA_COMPILER
$(info Using Nvidia Compiler - CUDA)
CC 		:=nvcc
CLIBS		:=-lnppc -lnppicc $(CLIBS)
CFLAGS		:=-g -O0 -DDEBUG $(CLIBS)
CCOBJFLAGS 	:= $(CFLAGS) -c
else
$(info Using GCC - No CUDA optimizations)
CC 		:=gcc
CFLAGS		:=-g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined $(CLIBS)
CCOBJFLAGS 	:= $(CFLAGS) -x c -c
endif

DBGFLAGS 	:= -g

SRC_PATH= src
OBJ_PATH= obj
BIN_PATH= bin
DBG_PATH := debug
TARGET 	= $(BIN_PATH)/main
.PHONY : all clean release

SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.c*)))
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
OBJ_DEBUG := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

default: makedir debug
all: $(TARGET)
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH) $(DBG_PATH)

debug: $(TARGET)

release: clean
# release: CFLAGS := -DNDEBUG -O2 $(CLIBS)
release: CFLAGS := -O2 $(CLIBS)
release: CCOBJFLAGS	:= $(CFLAGS) -x c -c
release: $(TARGET)


$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAGS) -o $@ $<

$(DBG_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAGS) $(DBGFLAGS) -o $@ $<

$(TARGET_DEBUG): $(OBJ_DEBUG)
	$(CC) $(CFLAGS) $(DBGFLAGS) $(OBJ_DEBUG) -o $@

# clean files list
DISTCLEAN_LIST := $(OBJ) \
                  $(OBJ_DEBUG)
CLEAN_LIST := $(TARGET) \
			  $(TARGET_DEBUG) \
			  $(DISTCLEAN_LIST)

clean:
	rm -f $(CLEAN_LIST)

