#include "numa_allocator.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#if __has_include(<linux/mempolicy.h>)
#include <linux/mempolicy.h>
#endif
#endif

namespace anvil::native {
namespace {

std::mutex g_mmap_alloc_mu;
std::unordered_map<void*, std::size_t> g_mmap_alloc_sizes;

inline bool env_flag_enabled(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    std::string normalized(raw);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (
        normalized == "0"
        || normalized == "false"
        || normalized == "off"
        || normalized == "no"
    ) {
        return false;
    }
    return true;
}

inline int env_int_or(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || errno != 0) {
        return fallback;
    }
    if (value < static_cast<long>(std::numeric_limits<int>::min())) {
        return std::numeric_limits<int>::min();
    }
    if (value > static_cast<long>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(value);
}

inline void touch_pages(void* ptr, std::size_t bytes, bool zero_init) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }
#ifdef __linux__
    const long page_size = sysconf(_SC_PAGESIZE);
    const std::size_t stride = static_cast<std::size_t>(
        page_size > 0 ? page_size : 4096
    );
#else
    const std::size_t stride = 4096;
#endif
    auto* data = reinterpret_cast<std::uint8_t*>(ptr);
    if (zero_init) {
        std::memset(ptr, 0, bytes);
        return;
    }
    for (std::size_t offset = 0; offset < bytes; offset += stride) {
        data[offset] = static_cast<std::uint8_t>(0);
    }
    data[bytes - 1] = static_cast<std::uint8_t>(0);
}

#ifdef __linux__
inline int mbind_policy_from_bind_policy(BindPolicy policy) {
#ifdef MPOL_BIND
    switch (policy) {
        case BindPolicy::Preferred:
#ifdef MPOL_PREFERRED
            return MPOL_PREFERRED;
#else
            return MPOL_DEFAULT;
#endif
        case BindPolicy::BindStrict:
            return MPOL_BIND;
        case BindPolicy::Interleave:
#ifdef MPOL_INTERLEAVE
            return MPOL_INTERLEAVE;
#else
            return MPOL_DEFAULT;
#endif
        case BindPolicy::None:
        default:
            return MPOL_DEFAULT;
    }
#else
    (void)policy;
    return 0;
#endif
}

inline void apply_madvise_policy(void* ptr, std::size_t bytes, HugePageMode mode) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }
    if (mode == HugePageMode::THPAdvice || mode == HugePageMode::THPCollapse) {
#ifdef MADV_HUGEPAGE
        (void)madvise(ptr, bytes, MADV_HUGEPAGE);
#endif
    }
    if (mode == HugePageMode::THPCollapse) {
#ifdef MADV_COLLAPSE
        (void)madvise(ptr, bytes, MADV_COLLAPSE);
#endif
    }
}

inline void apply_mbind_policy(
    void* ptr,
    std::size_t bytes,
    BindPolicy bind_policy,
    int preferred_node
) {
#if defined(SYS_mbind) && defined(MPOL_DEFAULT)
    if (ptr == nullptr || bytes == 0 || bind_policy == BindPolicy::None || preferred_node < 0) {
        return;
    }
    if (!env_flag_enabled("ANVIL_NUMA_STRICT", false)) {
        return;
    }
    unsigned long nodemask[16] = {0UL};
    const int word_bits = static_cast<int>(sizeof(unsigned long) * 8U);
    const int word = preferred_node / word_bits;
    const int bit = preferred_node % word_bits;
    if (word >= 0 && word < static_cast<int>(sizeof(nodemask) / sizeof(nodemask[0]))) {
        nodemask[word] = (1UL << bit);
    }
    const int policy = mbind_policy_from_bind_policy(bind_policy);
    const unsigned long maxnode = static_cast<unsigned long>(
        sizeof(nodemask) / sizeof(nodemask[0]) * word_bits
    );
    (void)syscall(
        SYS_mbind,
        ptr,
        bytes,
        policy,
        nodemask,
        maxnode,
#ifdef MPOL_MF_MOVE
        MPOL_MF_MOVE
#else
        0
#endif
    );
#else
    (void)ptr;
    (void)bytes;
    (void)bind_policy;
    (void)preferred_node;
#endif
}

inline void* try_hugetlb_alloc(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
#if defined(MAP_HUGETLB)
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;
#if defined(MAP_HUGE_2MB)
    flags |= MAP_HUGE_2MB;
#endif
    void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(g_mmap_alloc_mu);
        g_mmap_alloc_sizes[ptr] = bytes;
    }
    return ptr;
#else
    return nullptr;
#endif
}
#endif

}  // namespace

AllocOptions anvil_alloc_options_from_env() {
    AllocOptions options{};
    options.alignment = static_cast<std::size_t>(std::max(64, env_int_or("ANVIL_NUMA_ALIGNMENT", 64)));
    options.preferred_node = env_int_or("ANVIL_NUMA_PREFERRED_NODE", -1);

    const char* bind_raw = std::getenv("ANVIL_NUMA_BIND_POLICY");
    const std::string bind = bind_raw == nullptr ? "" : std::string(bind_raw);
    if (bind == "preferred") {
        options.bind_policy = BindPolicy::Preferred;
    } else if (bind == "strict") {
        options.bind_policy = BindPolicy::BindStrict;
    } else if (bind == "interleave") {
        options.bind_policy = BindPolicy::Interleave;
    } else {
        options.bind_policy = BindPolicy::None;
    }

    const char* huge_raw = std::getenv("ANVIL_NUMA_HUGEPAGE");
    const std::string huge = huge_raw == nullptr ? "" : std::string(huge_raw);
    if (huge == "thp") {
        options.huge_mode = HugePageMode::THPAdvice;
    } else if (huge == "collapse") {
        options.huge_mode = HugePageMode::THPCollapse;
    } else if (huge == "hugetlb2m") {
        options.huge_mode = HugePageMode::Hugetlb2MB;
    } else {
        options.huge_mode = HugePageMode::Off;
    }

    options.first_touch = env_flag_enabled("ANVIL_NUMA_FIRST_TOUCH", false);
    options.zero_init = env_flag_enabled("ANVIL_NUMA_ZERO_INIT", false);
    return options;
}

void* anvil_alloc_local_cpp(std::size_t bytes, const AllocOptions& opt) {
    if (bytes == 0) {
        return nullptr;
    }
    const std::size_t alignment = std::max<std::size_t>(64, opt.alignment);
    const std::size_t rounded = (bytes + (alignment - 1)) & ~(alignment - 1);

    void* ptr = nullptr;
#ifdef __linux__
    if (opt.huge_mode == HugePageMode::Hugetlb2MB) {
        ptr = try_hugetlb_alloc(rounded);
    }
#endif
    if (ptr == nullptr) {
        if (posix_memalign(&ptr, alignment, rounded) != 0) {
            return nullptr;
        }
    }

#ifdef __linux__
    apply_madvise_policy(ptr, rounded, opt.huge_mode);
    apply_mbind_policy(ptr, rounded, opt.bind_policy, opt.preferred_node);
#endif

    if (opt.first_touch || opt.zero_init) {
        touch_pages(ptr, rounded, opt.zero_init);
    }
    return ptr;
}

void anvil_free_local_cpp(void* ptr, std::size_t bytes, const AllocOptions& opt) {
    (void)bytes;
    (void)opt;
    if (ptr == nullptr) {
        return;
    }
#ifdef __linux__
    {
        std::lock_guard<std::mutex> lock(g_mmap_alloc_mu);
        auto it = g_mmap_alloc_sizes.find(ptr);
        if (it != g_mmap_alloc_sizes.end()) {
            const std::size_t mapped_bytes = it->second;
            g_mmap_alloc_sizes.erase(it);
            (void)munmap(ptr, mapped_bytes);
            return;
        }
    }
#endif
    std::free(ptr);
}

int anvil_query_page_nodes_cpp(void* ptr, std::size_t bytes, int* out_nodes, int max_pages) {
    if (ptr == nullptr || bytes == 0 || out_nodes == nullptr || max_pages <= 0) {
        return 0;
    }
#ifdef __linux__
#if defined(SYS_move_pages)
    const long page_size = sysconf(_SC_PAGESIZE);
    const std::size_t stride = static_cast<std::size_t>(
        page_size > 0 ? page_size : 4096
    );
    int page_count = static_cast<int>((bytes + stride - 1) / stride);
    page_count = std::min(page_count, max_pages);
    std::vector<void*> pages(static_cast<std::size_t>(page_count));
    for (int i = 0; i < page_count; ++i) {
        pages[static_cast<std::size_t>(i)] =
            reinterpret_cast<std::uint8_t*>(ptr) + static_cast<std::size_t>(i) * stride;
    }
    std::vector<int> status(static_cast<std::size_t>(page_count), -1);
    const long rc = syscall(
        SYS_move_pages,
        0,
        page_count,
        pages.data(),
        nullptr,
        status.data(),
        0
    );
    if (rc != 0) {
        return 0;
    }
    for (int i = 0; i < page_count; ++i) {
        out_nodes[i] = status[static_cast<std::size_t>(i)];
    }
    return page_count;
#else
    return 0;
#endif
#else
    (void)out_nodes;
    (void)max_pages;
    return 0;
#endif
}

void anvil_numa_advise_region(void* ptr, std::size_t bytes) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }
#ifdef __linux__
    const AllocOptions opt = anvil_alloc_options_from_env();
    apply_madvise_policy(ptr, bytes, opt.huge_mode);
    apply_mbind_policy(ptr, bytes, opt.bind_policy, opt.preferred_node);
#else
    (void)ptr;
    (void)bytes;
#endif
}

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
) {
    anvil::native::AllocOptions opt{};
    opt.alignment = alignment;
    opt.preferred_node = preferred_node;
    opt.bind_policy = static_cast<anvil::native::BindPolicy>(bind_policy);
    opt.huge_mode = static_cast<anvil::native::HugePageMode>(huge_mode);
    opt.first_touch = first_touch != 0;
    opt.zero_init = zero_init != 0;
    return anvil::native::anvil_alloc_local_cpp(bytes, opt);
}

void anvil_free_local(
    void* ptr,
    std::size_t bytes,
    std::size_t alignment,
    int preferred_node,
    int bind_policy,
    int huge_mode,
    int first_touch,
    int zero_init
) {
    anvil::native::AllocOptions opt{};
    opt.alignment = alignment;
    opt.preferred_node = preferred_node;
    opt.bind_policy = static_cast<anvil::native::BindPolicy>(bind_policy);
    opt.huge_mode = static_cast<anvil::native::HugePageMode>(huge_mode);
    opt.first_touch = first_touch != 0;
    opt.zero_init = zero_init != 0;
    anvil::native::anvil_free_local_cpp(ptr, bytes, opt);
}

int anvil_query_page_nodes(void* ptr, std::size_t bytes, int* out_nodes, int max_pages) {
    return anvil::native::anvil_query_page_nodes_cpp(ptr, bytes, out_nodes, max_pages);
}

}  // extern "C"
