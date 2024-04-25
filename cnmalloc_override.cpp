#include "/torch/neuware_home/include/cn_api.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static cn_uint64_t total_allocated_memory = 0;

CNresult custom_cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes);

extern "C" {
    // 导出symbol
    CNresult cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes) __attribute__((visibility("default")));
}

CNresult cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes) {
    // 调用自建的custom_cnMalloc函数
    return custom_cnMalloc(pmluAddr, bytes);
}

CNresult custom_cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes) {
    if (bytes > 1000000 || (bytes + total_allocated_memory) > 1000000) {
        printf("Out of Memory Error: Attempted to allocate more than 1,000,000 bytes of MLU memory.\n");
        return CN_MEMORY_ERROR_OUT_OF_MEMORY;
    }
    CNresult (*original_cnMalloc)(CNaddr *, cn_uint64_t) = reinterpret_cast<CNresult (*)(CNaddr *, cn_uint64_t)>(dlsym(RTLD_NEXT, "cnMalloc"));
    if (original_cnMalloc == nullptr) {
        fprintf(stderr, "Error: Unable to find original cnMalloc function.\n");
        return CN_ERROR_INVALID_VALUE;
    }
    CNresult result = original_cnMalloc(pmluAddr, bytes);
    if (result == CN_SUCCESS) {
        total_allocated_memory += bytes;
    }
    return result;
}

//使用静态初始化来模拟constructor的功能
namespace {
    struct Init {
        Init() {
            // 将cnMalloc的地址分配给custom_cnMalloc的地址
            *(void **)(&cnMalloc) = reinterpret_cast<void *>(custom_cnMalloc);
        }
    };
    Init init; // 静态初始化以确保它在main函数之前运行
}