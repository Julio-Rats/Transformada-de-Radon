#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define paths:
ifeq ($(origin SRCDIR), undefined)
	SRCDIR := $(shell pwd)/src
endif
ifeq ($(origin TEMPDIR), undefined)
	TEMPDIR := $(shell pwd)/tmp
endif
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ARCHS := -gencode=arch=compute_50,code=sm_50
$(TEMPDIR)/cuda_radon.so : $(TEMPDIR)/cuda_radon.o
	cd $(TEMPDIR) && gcc $(TEMPDIR)/cuda_radon.o -shared -o $(TEMPDIR)/cuda_radon.so
$(TEMPDIR)/cuda_radon.o : $(SRCDIR)/c/raft_cuda/cuda_radon.cu
	mkdir -p $(TEMPDIR)
	nvcc $(CCBIN) -c -O3 --compiler-options '-fPIC' -o $(TEMPDIR)/cuda_radon.o -m64 $(ARCHS) $(SRCDIR)/c/raft_cuda/cuda_radon.cu -lstdc++ -lpthread -lm
	cp $(SRCDIR)/python/* $(TEMPDIR)
