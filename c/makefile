CC 			:= gcc
CLIBS   	:= -lv4l2 -lv4lconvert
CFLAGS		:= -fsanitize=address -fsanitize=undefined -g -O0 -Wall $(CLIBS)
CCOBJFLAGS 	:= $(CFLAGS) -c
DBGFLAGS 	:= -g

SRC_PATH= .
OBJ_PATH= obj
BIN_PATH= bin
DBG_PATH := debug
TARGET 	= $(BIN_PATH)/capture


SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.c*)))
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
OBJ_DEBUG := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

default: makedir debug
all: $(TARGET)
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH) $(DBG_PATH)

debug: $(TARGET)

release: clean
release: CFLAGS :=-O2 -Wall $(CLIBS)
release: CCOBJFLAGS	:= $(CFLAGS) -c
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
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

