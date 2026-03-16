#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "numa_topology.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace {
std::atomic<int> g_runtime_thread_mode{-1};  // -1=unset, 0=batch, 1=decode

enum AffinityMode : int {
    kAffinityModeOff = 0,
    kAffinityModeLegacy = 1,
    kAffinityModeCompact = 2,
    kAffinityModeSplit = 3,
};

struct AffinityPlan {
    int mode = kAffinityModeLegacy;
    int preferred_decode_l3_domain = -1;
    int decode_primary_l3_domain = -1;
    std::vector<int> orchestrator_cpus;
    std::vector<int> decode_worker_cpus;
    std::vector<int> batch_worker_cpus;
    std::vector<int> visible_l3_domains;
    std::vector<int> batch_l3_domains;
};

thread_local int g_thread_migration_count = 0;
thread_local int g_last_sampled_cpu = -1;

std::mutex g_affinity_plan_mu;
AffinityPlan g_affinity_plan;

inline std::string normalize_env_token(const char* raw) {
    if (raw == nullptr) {
        return {};
    }
    std::string normalized(raw);
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
    );
    while (!normalized.empty() && std::isspace(static_cast<unsigned char>(normalized.back()))) {
        normalized.pop_back();
    }
    std::size_t start = 0;
    while (start < normalized.size()
           && std::isspace(static_cast<unsigned char>(normalized[start]))) {
        ++start;
    }
    return normalized.substr(start);
}

inline int read_env_threads(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return 0;
    }
    const int value = std::atoi(env);
    return value > 0 ? value : 0;
}

inline bool read_env_int(const char* name, int* out_value) {
    if (out_value == nullptr) {
        return false;
    }
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(env, &end, 10);
    if (end == env || errno != 0) {
        return false;
    }
    if (value < static_cast<long>(std::numeric_limits<int>::min())) {
        value = static_cast<long>(std::numeric_limits<int>::min());
    } else if (value > static_cast<long>(std::numeric_limits<int>::max())) {
        value = static_cast<long>(std::numeric_limits<int>::max());
    }
    *out_value = static_cast<int>(value);
    return true;
}

inline bool read_env_bool(const char* name, bool default_value) {
    const std::string normalized = normalize_env_token(std::getenv(name));
    if (normalized.empty()) {
        return default_value;
    }
    if (normalized == "0"
        || normalized == "false"
        || normalized == "off"
        || normalized == "no") {
        return false;
    }
    return true;
}

inline int read_preferred_decode_l3_domain() {
    int value = -1;
    if (!read_env_int("ANVIL_NUMA_DECODE_L3_DOMAIN", &value)) {
        return -1;
    }
    return value;
}

inline int read_affinity_mode_from_env() {
    const std::string mode = normalize_env_token(
        std::getenv("ANVIL_NUMA_AFFINITY_MODE")
    );
    if (mode == "off") {
        return kAffinityModeOff;
    }
    if (mode == "compact") {
        return kAffinityModeCompact;
    }
    if (mode == "split") {
        return kAffinityModeSplit;
    }
    return kAffinityModeLegacy;
}

inline int decode_headroom_reserve(int logical, int default_reserve) {
    logical = std::max(1, logical);
    int configured = 0;
    if (read_env_int("ANVIL_NUM_THREADS_HEADROOM", &configured)) {
        return std::max(0, std::min(logical, configured));
    }
    return std::max(0, std::min(logical, default_reserve));
}

inline int threads_from_reserve(int logical, int minimum, int reserve) {
    logical = std::max(1, logical);
    minimum = std::max(1, std::min(logical, minimum));
    reserve = std::max(0, reserve);
    const int candidate = logical - reserve;
    return std::max(minimum, std::min(logical, candidate));
}

inline long read_long_file(const std::string& path) {
    std::ifstream input(path);
    long value = 0;
    if (!input.is_open()) {
        return 0;
    }
    input >> value;
    return input.fail() ? 0 : value;
}

inline std::vector<int> parse_cpu_range_list(const std::string& raw) {
    std::vector<int> values;
    std::size_t start = 0;
    while (start < raw.size()) {
        std::size_t end = raw.find(',', start);
        if (end == std::string::npos) {
            end = raw.size();
        }
        const std::string token = raw.substr(start, end - start);
        const std::size_t dash = token.find('-');
        if (dash == std::string::npos) {
            const int cpu = std::atoi(token.c_str());
            if (cpu >= 0) {
                values.push_back(cpu);
            }
        } else {
            const int lo = std::atoi(token.substr(0, dash).c_str());
            const int hi = std::atoi(token.substr(dash + 1).c_str());
            if (lo >= 0 && hi >= lo) {
                for (int cpu = lo; cpu <= hi; ++cpu) {
                    values.push_back(cpu);
                }
            }
        }
        start = end + 1;
    }
    return values;
}

inline std::vector<int> parse_env_cpu_list(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return {};
    }
    return parse_cpu_range_list(env);
}

inline bool contains_cpu(const std::vector<int>& cpus, int cpu) {
    return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end();
}

inline std::vector<int> filtered_cpus(
    const std::vector<int>& source,
    const std::vector<int>& allowed
) {
    if (source.empty() || allowed.empty()) {
        return {};
    }
    std::vector<int> filtered;
    filtered.reserve(source.size());
    for (int cpu : source) {
        if (contains_cpu(allowed, cpu)) {
            filtered.push_back(cpu);
        }
    }
    return filtered;
}

inline std::vector<int> exclude_cpus(
    const std::vector<int>& source,
    const std::vector<int>& excluded
) {
    if (source.empty() || excluded.empty()) {
        return source;
    }
    std::vector<int> out;
    out.reserve(source.size());
    for (int cpu : source) {
        if (!contains_cpu(excluded, cpu)) {
            out.push_back(cpu);
        }
    }
    return out;
}

inline int cached_physical_core_count();

inline const char* affinity_mode_name(int mode) {
    switch (mode) {
        case kAffinityModeOff:
            return "off";
        case kAffinityModeCompact:
            return "compact";
        case kAffinityModeSplit:
            return "split";
        case kAffinityModeLegacy:
        default:
            return "legacy";
    }
}

inline std::string cpu_list_json(const std::vector<int>& cpus) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < cpus.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        out << cpus[index];
    }
    out << ']';
    return out.str();
}

#ifdef __linux__
inline std::vector<int> unique_sorted_cpu_list(std::vector<int> cpus) {
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

inline bool read_text_line(const std::string& path, std::string* out) {
    if (out == nullptr) {
        return false;
    }
    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }
    std::getline(input, *out);
    return !input.fail();
}

inline std::vector<int> visible_cpus() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        const long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        std::vector<int> all;
        for (int cpu = 0; cpu < std::max<long>(1, nproc); ++cpu) {
            all.push_back(cpu);
        }
        return all;
    }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

inline std::vector<std::vector<int>> discover_l3_domains_linux(
    const std::vector<int>& candidate_cpus
) {
    if (candidate_cpus.empty()) {
        return {};
    }
    const std::set<int> candidate_set(candidate_cpus.begin(), candidate_cpus.end());
    std::set<std::vector<int>> unique_domains;
    bool discovered = false;
    for (int cpu : candidate_cpus) {
        for (int index = 0; index < 16; ++index) {
            const std::string base =
                "/sys/devices/system/cpu/cpu"
                + std::to_string(cpu)
                + "/cache/index"
                + std::to_string(index)
                + "/";
            if (read_long_file(base + "level") != 3) {
                continue;
            }
            std::string cache_type;
            if (!read_text_line(base + "type", &cache_type)) {
                continue;
            }
            if (normalize_env_token(cache_type.c_str()) != "unified") {
                continue;
            }
            std::string shared_cpu_list;
            if (!read_text_line(base + "shared_cpu_list", &shared_cpu_list)) {
                continue;
            }
            std::vector<int> domain;
            for (int member : parse_cpu_range_list(shared_cpu_list)) {
                if (candidate_set.count(member) != 0) {
                    domain.push_back(member);
                }
            }
            domain = unique_sorted_cpu_list(std::move(domain));
            if (domain.empty()) {
                continue;
            }
            unique_domains.insert(domain);
            discovered = true;
            break;
        }
    }
    if (!discovered) {
        return {};
    }
    return std::vector<std::vector<int>>(unique_domains.begin(), unique_domains.end());
}

inline std::vector<int> collapse_smt_siblings_linux(const std::vector<int>& cpus) {
    const std::set<int> cpu_set(cpus.begin(), cpus.end());
    std::set<int> representatives;
    for (int cpu : cpus) {
        const std::string siblings_path =
            "/sys/devices/system/cpu/cpu"
            + std::to_string(cpu)
            + "/topology/thread_siblings_list";
        std::ifstream input(siblings_path);
        int representative = cpu;
        if (input.is_open()) {
            std::string raw;
            std::getline(input, raw);
            for (int sibling : parse_cpu_range_list(raw)) {
                if (cpu_set.count(sibling) != 0) {
                    representative = std::min(representative, sibling);
                }
            }
        }
        representatives.insert(representative);
    }
    return std::vector<int>(representatives.begin(), representatives.end());
}

inline std::vector<int> flatten_split_domains_linux(
    const std::vector<std::vector<int>>& domains
) {
    std::size_t max_domain_size = 0;
    for (const auto& domain : domains) {
        max_domain_size = std::max(max_domain_size, domain.size());
    }
    std::vector<int> ordered;
    for (std::size_t index = 0; index < max_domain_size; ++index) {
        for (const auto& domain : domains) {
            if (index < domain.size()) {
                ordered.push_back(domain[index]);
            }
        }
    }
    return ordered;
}

inline int detect_physical_cores_linux() {
    const std::vector<int> visible = visible_cpus();
    if (visible.empty()) {
        return 0;
    }
    std::set<int> visible_set(visible.begin(), visible.end());
    std::set<int> unique_cores;
    for (int cpu : visible) {
        const std::string siblings_path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
            + "/topology/thread_siblings_list";
        std::ifstream input(siblings_path);
        if (!input.is_open()) {
            unique_cores.insert(cpu);
            continue;
        }
        std::string raw;
        std::getline(input, raw);
        std::vector<int> siblings = parse_cpu_range_list(raw);
        int representative = cpu;
        for (int sibling : siblings) {
            if (visible_set.count(sibling) != 0) {
                representative = std::min(representative, sibling);
            }
        }
        unique_cores.insert(representative);
    }
    return static_cast<int>(unique_cores.size());
}

inline std::vector<int> detect_high_frequency_cores_linux() {
    const std::vector<int> visible = visible_cpus();
    if (visible.empty()) {
        return {};
    }
    const std::set<int> visible_set(visible.begin(), visible.end());
    std::vector<std::pair<int, long>> cpu_freqs;
    cpu_freqs.reserve(visible.size());
    long max_freq = 0;
    for (int cpu : visible) {
        const std::string base =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/";
        long freq = read_long_file(base + "base_frequency");
        if (freq <= 0) {
            freq = read_long_file(base + "cpuinfo_max_freq");
        }
        if (freq <= 0) {
            continue;
        }
        cpu_freqs.emplace_back(cpu, freq);
        max_freq = std::max(max_freq, freq);
    }
    if (cpu_freqs.empty() || max_freq <= 0) {
        return {};
    }
    const long threshold = static_cast<long>(static_cast<double>(max_freq) * 0.90);
    std::set<int> p_core_representatives;
    for (const auto& [cpu, freq] : cpu_freqs) {
        if (freq < threshold) {
            continue;
        }
        const std::string siblings_path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
            + "/topology/thread_siblings_list";
        std::ifstream input(siblings_path);
        int representative = cpu;
        if (input.is_open()) {
            std::string raw;
            std::getline(input, raw);
            for (int sibling : parse_cpu_range_list(raw)) {
                if (visible_set.count(sibling) != 0) {
                    representative = std::min(representative, sibling);
                }
            }
        }
        p_core_representatives.insert(representative);
    }
    std::vector<int> p_cores(
        p_core_representatives.begin(),
        p_core_representatives.end()
    );
    const int physical = std::max(1, detect_physical_cores_linux());
    // Avoid mistaking opportunistic turbo bins on symmetric CPUs for a hybrid topology.
    if (static_cast<int>(p_cores.size()) < std::max(1, physical / 2)) {
        return {};
    }
    if (static_cast<int>(p_cores.size()) >= physical) {
        return {};
    }
    return p_cores;
}

inline std::vector<int> legacy_affinity_cpus_linux(int use_p_cores_only) {
    std::vector<int> cpus;
    if (use_p_cores_only != 0) {
        cpus = detect_high_frequency_cores_linux();
    }
    if (cpus.empty()) {
        cpus = visible_cpus();
        if (!cpus.empty()) {
            const int physical = std::max(1, cached_physical_core_count());
            if (static_cast<int>(cpus.size()) > physical) {
                cpus.resize(static_cast<std::size_t>(physical));
            }
        }
    }
    return cpus;
}

inline std::vector<int> policy_affinity_cpus_linux(
    int use_p_cores_only,
    int affinity_mode
) {
    if (affinity_mode == kAffinityModeLegacy) {
        return legacy_affinity_cpus_linux(use_p_cores_only);
    }
    std::vector<int> candidate_cpus;
    if (use_p_cores_only != 0) {
        candidate_cpus = detect_high_frequency_cores_linux();
    }
    if (candidate_cpus.empty()) {
        candidate_cpus = visible_cpus();
    }
    candidate_cpus = unique_sorted_cpu_list(std::move(candidate_cpus));
    if (candidate_cpus.empty()) {
        return {};
    }
    const std::vector<std::vector<int>> l3_domains = discover_l3_domains_linux(
        candidate_cpus
    );
    if (affinity_mode == kAffinityModeCompact) {
        if (l3_domains.empty()) {
            return legacy_affinity_cpus_linux(use_p_cores_only);
        }
        std::vector<int> compact = l3_domains.front();
        const bool enable_smt = read_env_bool("ANVIL_NUMA_ENABLE_SMT", false);
        if (!enable_smt) {
            compact = collapse_smt_siblings_linux(compact);
        }
        return compact.empty() ? l3_domains.front() : compact;
    }
    if (affinity_mode == kAffinityModeSplit) {
        if (l3_domains.empty()) {
            return legacy_affinity_cpus_linux(use_p_cores_only);
        }
        return flatten_split_domains_linux(l3_domains);
    }
    return legacy_affinity_cpus_linux(use_p_cores_only);
}

inline AffinityPlan build_affinity_plan_linux(
    int affinity_mode,
    const std::vector<int>& cpus
) {
    AffinityPlan plan;
    plan.mode = affinity_mode;
    if (cpus.empty()) {
        return plan;
    }
    const std::vector<int> requested_orchestrator = parse_env_cpu_list(
        "ANVIL_NUMA_ORCHESTRATOR_CPUS"
    );
    plan.orchestrator_cpus = filtered_cpus(requested_orchestrator, cpus);
    if (plan.orchestrator_cpus.empty()) {
        plan.orchestrator_cpus.push_back(cpus.front());
    }

    const std::vector<std::vector<int>> l3_domains = discover_l3_domains_linux(cpus);
    plan.preferred_decode_l3_domain = read_preferred_decode_l3_domain();
    for (std::size_t idx = 0; idx < l3_domains.size(); ++idx) {
        plan.visible_l3_domains.push_back(static_cast<int>(idx));
    }

    std::vector<int> decode_domain;
    if (!l3_domains.empty()) {
        std::size_t decode_domain_index = 0;
        if (
            plan.preferred_decode_l3_domain >= 0
            && plan.preferred_decode_l3_domain
                < static_cast<int>(l3_domains.size())
        ) {
            decode_domain_index = static_cast<std::size_t>(
                plan.preferred_decode_l3_domain
            );
        } else {
            std::size_t largest_domain_index = 0;
            std::size_t largest_domain_size = 0;
            for (std::size_t idx = 0; idx < l3_domains.size(); ++idx) {
                if (l3_domains[idx].size() > largest_domain_size) {
                    largest_domain_index = idx;
                    largest_domain_size = l3_domains[idx].size();
                }
            }
            decode_domain_index = largest_domain_index;
        }
        plan.decode_primary_l3_domain = static_cast<int>(decode_domain_index);
        decode_domain = l3_domains[decode_domain_index];
        for (std::size_t idx = 0; idx < l3_domains.size(); ++idx) {
            if (idx == decode_domain_index) {
                continue;
            }
            plan.batch_l3_domains.push_back(static_cast<int>(idx));
            plan.batch_worker_cpus.insert(
                plan.batch_worker_cpus.end(),
                l3_domains[idx].begin(),
                l3_domains[idx].end()
            );
        }
    }

    std::vector<int> workers = exclude_cpus(
        decode_domain.empty() ? cpus : decode_domain,
        plan.orchestrator_cpus
    );
    if (workers.empty()) {
        workers = decode_domain.empty() ? cpus : decode_domain;
    }
    plan.decode_worker_cpus = workers;
    if (plan.batch_worker_cpus.empty()) {
        plan.batch_worker_cpus = cpus;
    } else {
        plan.batch_worker_cpus = unique_sorted_cpu_list(
            std::move(plan.batch_worker_cpus)
        );
    }
    return plan;
}

inline int bind_current_thread_to_cpu_linux(int cpu) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        return 0;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    const int pthread_rc = pthread_setaffinity_np(
        pthread_self(),
        sizeof(mask),
        &mask
    );
    if (pthread_rc == 0) {
        return 1;
    }
    return sched_setaffinity(0, sizeof(mask), &mask) == 0 ? 1 : 0;
}

inline int apply_affinity_linux(const std::vector<int>& cpus) {
    if (cpus.empty()) {
        return 0;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(cpu, &mask);
        }
    }
    return sched_setaffinity(0, sizeof(mask), &mask) == 0 ? 1 : 0;
}
#endif

inline int clamp_to_runtime_threads(int requested) {
#ifdef _OPENMP
    const int procs = std::max(1, omp_get_num_procs());
    if (requested > 0) {
        return std::min(requested, procs);
    }
    return std::min(std::max(1, omp_get_max_threads()), procs);
#else
    return std::max(1, requested);
#endif
}

inline int cached_physical_core_count() {
    static std::atomic<int> cached{0};
    int current = cached.load(std::memory_order_acquire);
    if (current > 0) {
        return current;
    }
    int detected = 0;
#ifdef __linux__
    detected = detect_physical_cores_linux();
#endif
    if (detected <= 0) {
#ifdef _OPENMP
        detected = std::max(1, omp_get_num_procs());
#else
        detected = 1;
#endif
    }
    cached.store(detected, std::memory_order_release);
    return detected;
}

inline int logical_core_count() {
#ifdef _OPENMP
    return std::max(1, omp_get_num_procs());
#else
    return 1;
#endif
}

inline int default_threads_for_path(bool decode_path) {
    const int env_mode_threads = read_env_threads(
        decode_path ? "ANVIL_NUM_THREADS_DECODE" : "ANVIL_NUM_THREADS_BATCH"
    );
    const int env_threads = env_mode_threads > 0
        ? env_mode_threads
        : read_env_threads("ANVIL_NUM_THREADS");
    if (env_threads > 0) {
        return clamp_to_runtime_threads(env_threads);
    }
    const int logical = logical_core_count();
    int min_auto = read_env_threads("ANVIL_AUTO_MIN_THREADS");
    if (min_auto <= 0) {
        min_auto = 4;
    }
    min_auto = std::max(1, std::min(min_auto, logical));
    if (decode_path) {
        const int physical = std::max(
            1,
            std::min(logical, cached_physical_core_count())
        );
        const int default_reserve = std::max(0, logical - physical);
        const int reserve = decode_headroom_reserve(logical, default_reserve);
        return clamp_to_runtime_threads(
            threads_from_reserve(logical, min_auto, reserve)
        );
    }
    return clamp_to_runtime_threads(std::max(min_auto, logical));
}

}  // namespace

extern "C" {

const char* anvil_native_build_id() {
#ifdef ANVIL_NATIVE_BUILD_ID
    return ANVIL_NATIVE_BUILD_ID;
#else
    return __DATE__ " " __TIME__;
#endif
}

void anvil_set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) {
        omp_set_num_threads(clamp_to_runtime_threads(n));
    }
#else
    (void)n;
#endif
}

int anvil_get_num_threads() {
    return clamp_to_runtime_threads(0);
}

int anvil_get_num_threads_for_path(int decode_path) {
    return default_threads_for_path(decode_path != 0);
}

void anvil_set_thread_mode(int decode_path) {
    const int normalized = decode_path != 0 ? 1 : 0;
    g_runtime_thread_mode.store(normalized, std::memory_order_release);
}

int anvil_get_thread_mode() {
    return g_runtime_thread_mode.load(std::memory_order_acquire);
}

int anvil_get_num_procs() {
    return logical_core_count();
}

int anvil_detect_physical_cores() {
    return cached_physical_core_count();
}

int anvil_get_p_core_count() {
#ifdef __linux__
    return static_cast<int>(detect_high_frequency_cores_linux().size());
#else
    return 0;
#endif
}

int anvil_set_thread_affinity(int use_p_cores_only) {
    const char* disable = std::getenv("ANVIL_DISABLE_OMP_AFFINITY_DEFAULTS");
    if (disable == nullptr || std::strcmp(disable, "1") != 0) {
        if (std::getenv("OMP_PLACES") == nullptr) {
            setenv("OMP_PLACES", "cores", 0);
        }
        if (std::getenv("OMP_PROC_BIND") == nullptr) {
            setenv("OMP_PROC_BIND", "close", 0);
        }
        if (std::getenv("OMP_DYNAMIC") == nullptr) {
            setenv("OMP_DYNAMIC", "FALSE", 0);
        }
        if (std::getenv("OMP_MAX_ACTIVE_LEVELS") == nullptr) {
            setenv("OMP_MAX_ACTIVE_LEVELS", "1", 0);
        }
    }
#ifdef __linux__
    const int affinity_mode = read_affinity_mode_from_env();
    if (affinity_mode == kAffinityModeOff) {
        std::lock_guard<std::mutex> lock(g_affinity_plan_mu);
        g_affinity_plan = AffinityPlan{};
        g_affinity_plan.mode = kAffinityModeOff;
        return 0;
    }
    const std::vector<int> cpus = policy_affinity_cpus_linux(
        use_p_cores_only,
        affinity_mode
    );
    {
        std::lock_guard<std::mutex> lock(g_affinity_plan_mu);
        g_affinity_plan = build_affinity_plan_linux(affinity_mode, cpus);
    }
    return apply_affinity_linux(cpus);
#else
    (void)use_p_cores_only;
    return 0;
#endif
}

int anvil_get_affinity_mode() {
#ifdef __linux__
    return read_affinity_mode_from_env();
#else
    return kAffinityModeLegacy;
#endif
}

int anvil_configure_affinity_mode(int mode) {
    const int normalized = (mode < kAffinityModeOff || mode > kAffinityModeSplit)
        ? kAffinityModeLegacy
        : mode;
    switch (normalized) {
        case kAffinityModeOff:
            setenv("ANVIL_NUMA_AFFINITY_MODE", "off", 1);
            break;
        case kAffinityModeCompact:
            setenv("ANVIL_NUMA_AFFINITY_MODE", "compact", 1);
            break;
        case kAffinityModeSplit:
            setenv("ANVIL_NUMA_AFFINITY_MODE", "split", 1);
            break;
        case kAffinityModeLegacy:
        default:
            setenv("ANVIL_NUMA_AFFINITY_MODE", "legacy", 1);
            break;
    }
    return normalized;
}

int anvil_get_l3_domain_count() {
    const auto& topology = anvil::native::anvil_get_topology();
    return static_cast<int>(topology.l3_domains.size());
}

int anvil_affinity_plan_export_json(char* out, int out_len) {
    if (out == nullptr || out_len <= 0) {
        return 0;
    }
    AffinityPlan plan;
    {
        std::lock_guard<std::mutex> lock(g_affinity_plan_mu);
        plan = g_affinity_plan;
    }
    std::ostringstream json;
    json
        << '{'
        << "\"mode\":" << plan.mode << ','
        << "\"mode_name\":\"" << affinity_mode_name(plan.mode) << "\","
        << "\"preferred_decode_l3_domain\":" << plan.preferred_decode_l3_domain << ','
        << "\"decode_primary_l3_domain\":" << plan.decode_primary_l3_domain << ','
        << "\"visible_l3_domains\":" << cpu_list_json(plan.visible_l3_domains) << ','
        << "\"batch_l3_domains\":" << cpu_list_json(plan.batch_l3_domains) << ','
        << "\"orchestrator_cpus\":" << cpu_list_json(plan.orchestrator_cpus) << ','
        << "\"decode_worker_cpus\":" << cpu_list_json(plan.decode_worker_cpus) << ','
        << "\"batch_worker_cpus\":" << cpu_list_json(plan.batch_worker_cpus) << ','
        << "\"decode_domain_reserved\":"
        << ((plan.decode_primary_l3_domain >= 0 && !plan.batch_l3_domains.empty())
                ? "true"
                : "false")
        << '}';
    const std::string payload = json.str();
    if (static_cast<int>(payload.size()) + 1 > out_len) {
        return 0;
    }
    std::memcpy(out, payload.c_str(), payload.size() + 1);
    return 1;
}

int anvil_bind_worker_thread(int worker_tid, int role_decode);

int anvil_bind_worker_thread(int worker_tid, int role_decode) {
#ifdef __linux__
    if (worker_tid < 0) {
        return 0;
    }
    std::vector<int> cpus;
    {
        std::lock_guard<std::mutex> lock(g_affinity_plan_mu);
        cpus = role_decode != 0
            ? g_affinity_plan.decode_worker_cpus
            : g_affinity_plan.batch_worker_cpus;
    }
    if (cpus.empty()) {
        return 0;
    }
    const int cpu = cpus[static_cast<std::size_t>(worker_tid) % cpus.size()];
    const int bound = bind_current_thread_to_cpu_linux(cpu);
    if (bound == 1) {
        g_last_sampled_cpu = cpu;
    }
    return bound;
#else
    (void)worker_tid;
    (void)role_decode;
    return 0;
#endif
}

int anvil_sample_thread_cpu() {
#ifdef __linux__
    const int current_cpu = sched_getcpu();
    if (current_cpu < 0) {
        return -1;
    }
    if (g_last_sampled_cpu >= 0 && g_last_sampled_cpu != current_cpu) {
        ++g_thread_migration_count;
        if (read_env_bool("ANVIL_NUMA_MIGRATION_WATCHDOG", false)) {
#ifdef _OPENMP
            const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
#else
            const int tid = 0;
#endif
            const int decode_mode = g_runtime_thread_mode.load(std::memory_order_acquire) == 1
                ? 1
                : 0;
            (void)anvil_bind_worker_thread(tid, decode_mode);
        }
    }
    g_last_sampled_cpu = current_cpu;
    return current_cpu;
#else
    return -1;
#endif
}

int anvil_get_thread_migration_count() {
    return g_thread_migration_count;
}

int anvil_get_last_cpu() {
    return g_last_sampled_cpu;
}

int anvil_openmp_enabled() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int anvil_get_omp_max_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

int anvil_get_omp_dynamic() {
#ifdef _OPENMP
    return omp_get_dynamic() != 0 ? 1 : 0;
#else
    return 0;
#endif
}

int anvil_get_omp_active_levels() {
#ifdef _OPENMP
    return std::max(0, omp_get_max_active_levels());
#else
    return 0;
#endif
}

int anvil_compiled_with_avx2() {
#ifdef __AVX2__
    return 1;
#else
    return 0;
#endif
}

int anvil_compiled_with_avx512() {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

}  // extern "C"
