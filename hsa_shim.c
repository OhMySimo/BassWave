// hsa_shim.c
#include <stdint.h>

typedef int hsa_status_t;
typedef struct { uint64_t handle; } hsa_agent_t;

hsa_status_t hsa_amd_memory_get_preferred_copy_engine(
    hsa_agent_t src_agent,
    hsa_agent_t dst_agent,
    hsa_agent_t *engine)
{
    if (engine) engine->handle = 0;
    return 0;
}
