#include "/torch/neuware_home/include/cn_api.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static cn_uint64_t total_allocated_memory = 0;

CNresult custom_cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes);

extern "C" {
    // ����symbol
    CNresult cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes) __attribute__((visibility("default")));
}

CNresult cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes) {
    // �����Խ���custom_cnMalloc����
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

//ʹ�þ�̬��ʼ����ģ��constructor�Ĺ���
namespace {
    struct Init {
        Init() {
            // ��cnMalloc�ĵ�ַ�����custom_cnMalloc�ĵ�ַ
            *(void **)(&cnMalloc) = reinterpret_cast<void *>(custom_cnMalloc);
        }
    };
    Init init; // ��̬��ʼ����ȷ������main����֮ǰ����
}