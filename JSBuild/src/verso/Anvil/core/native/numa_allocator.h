#pragma once

#include <cstddef>

namespace anvil::native {

enum class HugePageMode : int {
    Off = 0,
    THPAdvice = 1,
    THPCollapse = 2,
    Hugetlb2MB = 3,
};

enum class BindPolicy : int {
    None = 0,
    Preferred = 1,
    BindStrict = 2,
    Interleave = 3,
};

struct AllocOptions {
    std::size_t alignment = 64;
    int preferred_node = -1;
    BindPolicy bind_policy = BindPolicy::None;
    HugePageMode huge_mode = HugePageMode::Off;
    bool first_touch = false;
    bool zero_init = false;
};

void* anvil_alloc_local_cpp(std::size_t bytes, const AllocOptions& opt);
void anvil_free_local_cpp(void* ptr, std::size_t bytes, const AllocOptions& opt);
int anvil_query_page_nodes_cpp(void* ptr, std::size_t bytes, int* out_nodes, int max_pages);
void anvil_numa_advise_region(void* ptr, std::size_t bytes);
AllocOptions anvil_alloc_options_from_env();

}  // namespace anvil::native

extern "C" {

void* anvil_alloc_local(
    std::size_t bytes,
    std::size_t alignment,
    int preferred_node,
    int bind_policy,
    int huge_mode,
    int first_touch,
    int zero_init
);
void anvil_free_local(
    void* ptr,
    std::size_t bytes,
    std::size_t alignment,
    int preferred_node,
    int bind_policy,
    int huge_mode,
    int first_touch,
    int zero_init
);
int anvil_query_page_nodes(void* ptr, std::size_t bytes, int* out_nodes, int max_pages);

}
