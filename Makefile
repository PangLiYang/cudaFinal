NVCC = nvcc
CFLAGS = -arch=compute_75

EXEC1 = maxpool
EXEC2 = topk

SRC1 = maxpool_kernel.cu
SRC2 = topk_kernel.cu

all: $(EXEC1) $(EXEC2)

$(EXEC1): $(SRC1)
	$(NVCC) -o $(EXEC1) $(SRC1) $(CFLAGS)

$(EXEC2): $(SRC2)
	$(NVCC) -o $(EXEC2) $(SRC2) $(CFLAGS)

clean:
	rm -f $(EXEC1) $(EXEC2)