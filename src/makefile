TARGET = neo

CC = gcc
CPP = g++
MACHINE = $(shell $(CC) -dumpmachine)
# machine should be as follow:
#   x86_64-linux-gnu
#   mingw32
ifneq (,$(or $(findstring mingw, $(MACHINE)), $(findstring cygwin, $(MACHINE))))
	PLATFORM = WIN
	LIBS = -L../opencv_win64/mingw/lib -lopencv_highgui246 -lopencv_core246 -lopencv_imgproc246 -lopencv_nonfree246 -lopencv_features2d246 -lopencv_flann246 -lopencv_imgproc246 -lm
	#-lgdi32 -luser32 -Wl,--subsystem,windows
	RM = del /f/s/q
	RMDIR = rmdir /s/q
	#RM = rm
else
	PLATFORM = LINUX
	LIBS = -lm
	RM = rm
endif

CFLAGS = -I../opencv_include

SRCS := $(wildcard *.c)
SRCPPS := $(wildcard *.cpp)
OBJCS := $(SRCS:.c=.o)
OBJCPPS := $(SRCPPS:.cpp=.o)

OBJS := $(OBJCS) $(OBJCPPS)

all:$(TARGET)

%.o: %.c
	@echo --[CC]-- $@
	$(CC) -c -o"$@" "$<" $(CFLAGS)

%.o: %.cpp
	@echo --[CPP]-- $@
	$(CPP) -c -o"$@" "$<" $(CFLAGS)

$(TARGET): $(OBJS)
	@echo --[TARGET]-- $@
	$(CPP) -o "$(TARGET)" $(OBJS) $(LIBS) $(CFLAGS)
#$(CC) -o "$(TARGET)" $(OBJS) $(LIBS) $(CFLAGS)
	@echo [Finished]

clean:
	-$(RM) ..\\execute\\*
	-$(RM) $(OBJS) $(TARGET).exe
#$(shell $(RM) ,$(OBJS), $(TARGET))

run:
	@echo --[RUN]-- 
	copy $(TARGET).exe ..\\execute\\$(TARGET).exe
	copy ..\\opencv_win64\mingw\\bin\\libopencv_highgui246.dll ..\\execute\\
	copy ..\\opencv_win64\mingw\\bin\\libopencv_core246.dll ..\\execute\\
	copy ..\\opencv_win64\mingw\\bin\\libopencv_imgproc246.dll ..\\execute\\
	copy ..\\opencv_win64\mingw\\bin\\libopencv_features2d246.dll ..\\execute\\
	copy ..\\opencv_win64\mingw\\bin\\libopencv_nonfree246.dll ..\\execute\\
	copy ..\\opencv_win64\mingw\\bin\\libopencv_flann246.dll ..\\execute\\
	copy ..\\opencv_win64\mingw\\bin\\libopencv_imgproc246.dll ..\\execute\\
	copy .\\I1_000000.png ..\\execute\\
	copy .\\I2_000000.png ..\\execute\\
	copy .\\ar1.jpg ..\\execute\\
#$(TARGET)
	@echo --[Finished RUN]--

.PHONY: all clean run


#@echo --[RUN]--
#$(shell $(TARGET))
#@echo --[Finished RUN]--