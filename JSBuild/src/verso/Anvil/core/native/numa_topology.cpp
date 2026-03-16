#include "numa_topology.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace anvil::native {
namespace {

std::mutex g_topology_mu;
TopologyMap g_topology_cache;
std::atomic<bool> g_topology_initialized{false};

inline std::string trim_ascii(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) {
        s.pop_back();
    }
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
        ++start;
    }
    return s.substr(start);
}

inline std::vector<int> parse_cpu_range_list(const std::string& raw) {
    std::vector<int> values;
    std::size_t start = 0;
    while (start < raw.size()) {
        std::size_t end = raw.find(',', start);
        if (end == std::string::npos) {
            end = raw.size();
        }
        const std::string token = trim_ascii(raw.substr(start, end - start));
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
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

inline bool read_text_file(const std::string& path, std::string* out) {
    if (out == nullptr) {
        return false;
    }
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    *out = buffer.str();
    return true;
}

inline int read_int_file(const std::string& path, int fallback = -1) {
    std::ifstream in(path);
    int value = fallback;
    if (!in.is_open()) {
        return fallback;
    }
    in >> value;
    return in.fail() ? fallback : value;
}

inline std::vector<int> discover_allowed_cpus() {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        std::vector<int> cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &mask)) {
                cpus.push_back(cpu);
            }
        }
        return cpus;
    }
    const long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    std::vector<int> cpus;
    for (int cpu = 0; cpu < std::max<long>(1, nproc); ++cpu) {
        cpus.push_back(cpu);
    }
    return cpus;
#else
    return {0};
#endif
}

inline std::unordered_map<int, int> discover_cpu_to_node() {
    std::unordered_map<int, int> cpu_to_node;
#ifdef __linux__
    for (int node = 0; node < 512; ++node) {
        const std::string cpulist_path =
            "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
        std::string raw;
        if (!read_text_file(cpulist_path, &raw)) {
            continue;
        }
        for (int cpu : parse_cpu_range_list(raw)) {
            cpu_to_node[cpu] = node;
        }
    }
#endif
    return cpu_to_node;
}

inline TopologyMap discover_topology_linux() {
    TopologyMap map;
    map.allowed_cpus = discover_allowed_cpus();
    const std::set<int> allowed_set(map.allowed_cpus.begin(), map.allowed_cpus.end());
    const std::unordered_map<int, int> cpu_to_node = discover_cpu_to_node();

    std::unordered_map<std::string, int> l3_key_to_id;
    std::vector<std::vector<int>> domain_members;

    for (int cpu : map.allowed_cpus) {
        CpuInfo info;
        info.cpu_id = cpu;
        info.core_id = read_int_file(
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id",
            cpu
        );
        info.package_id = read_int_file(
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                + "/topology/physical_package_id",
            -1
        );
        info.die_id = read_int_file(
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/die_id",
            -1
        );
        info.cluster_id = read_int_file(
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/cluster_id",
            -1
        );
        info.numa_node = -1;
        auto node_it = cpu_to_node.find(cpu);
        if (node_it != cpu_to_node.end()) {
            info.numa_node = node_it->second;
        }
        info.online = read_int_file(
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/online",
            1
        ) != 0;

        std::string l3_key;
        for (int cache_idx = 0; cache_idx < 16; ++cache_idx) {
            const std::string cache_base =
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index"
                + std::to_string(cache_idx) + "/";
            const int level = read_int_file(cache_base + "level", -1);
            if (level != 3) {
                continue;
            }
            std::string type;
            if (!read_text_file(cache_base + "type", &type)) {
                continue;
            }
            type = trim_ascii(type);
            std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (type != "unified") {
                continue;
            }
            std::string shared;
            if (!read_text_file(cache_base + "shared_cpu_list", &shared)) {
                continue;
            }
            std::vector<int> members;
            for (int candidate : parse_cpu_range_list(shared)) {
                if (allowed_set.count(candidate) != 0) {
                    members.push_back(candidate);
                }
            }
            if (members.empty()) {
                continue;
            }
            std::sort(members.begin(), members.end());
            members.erase(std::unique(members.begin(), members.end()), members.end());
            std::ostringstream oss;
            for (std::size_t i = 0; i < members.size(); ++i) {
                if (i != 0) {
                    oss << ',';
                }
                oss << members[i];
            }
            l3_key = oss.str();
            auto id_it = l3_key_to_id.find(l3_key);
            if (id_it == l3_key_to_id.end()) {
                const int domain_id = static_cast<int>(domain_members.size());
                l3_key_to_id.emplace(l3_key, domain_id);
                domain_members.push_back(std::move(members));
                info.l3_domain_id = domain_id;
            } else {
                info.l3_domain_id = id_it->second;
            }
            break;
        }

        map.cpus.push_back(info);
    }

    if (domain_members.empty() && !map.allowed_cpus.empty()) {
        domain_members.push_back(map.allowed_cpus);
        for (auto& cpu : map.cpus) {
            cpu.l3_domain_id = 0;
        }
    }

    std::unordered_map<int, int> node_counts;
    std::unordered_map<int, std::set<int>> core_to_siblings;
    std::unordered_map<int, int> core_seen_per_domain;

    for (const auto& cpu : map.cpus) {
        if (cpu.numa_node >= 0) {
            node_counts[cpu.numa_node] += 1;
        }
        core_to_siblings[cpu.core_id].insert(cpu.cpu_id);
    }

    map.single_numa = node_counts.size() <= 1;
    map.symmetric_smt = true;
    int sibling_count_ref = -1;
    for (const auto& [_, siblings] : core_to_siblings) {
        const int count = static_cast<int>(siblings.size());
        if (sibling_count_ref < 0) {
            sibling_count_ref = count;
        } else if (sibling_count_ref != count) {
            map.symmetric_smt = false;
            break;
        }
    }

    map.l3_domains.reserve(domain_members.size());
    for (std::size_t domain_id = 0; domain_id < domain_members.size(); ++domain_id) {
        L3Domain domain;
        domain.id = static_cast<int>(domain_id);
        domain.logical_cpus = domain_members[domain_id];
        std::sort(domain.logical_cpus.begin(), domain.logical_cpus.end());
        int domain_node = -1;
        std::set<int> seen_cores;
        for (int cpu_id : domain.logical_cpus) {
            auto cpu_it = std::find_if(
                map.cpus.begin(),
                map.cpus.end(),
                [cpu_id](const CpuInfo& cpu) { return cpu.cpu_id == cpu_id; }
            );
            if (cpu_it == map.cpus.end()) {
                continue;
            }
            if (domain_node < 0 && cpu_it->numa_node >= 0) {
                domain_node = cpu_it->numa_node;
            }
            if (seen_cores.insert(cpu_it->core_id).second) {
                domain.primary_cpus.push_back(cpu_id);
            } else {
                domain.smt_siblings.push_back(cpu_id);
            }
        }
        domain.numa_node = domain_node;
        map.l3_domains.push_back(std::move(domain));
    }

    return map;
}

inline TopologyMap discover_topology_fallback() {
    TopologyMap map;
    map.allowed_cpus = discover_allowed_cpus();
    map.single_numa = true;
    map.symmetric_smt = true;
    L3Domain domain;
    domain.id = 0;
    domain.numa_node = -1;
    domain.logical_cpus = map.allowed_cpus;
    domain.primary_cpus = map.allowed_cpus;
    map.l3_domains.push_back(domain);
    for (int cpu : map.allowed_cpus) {
        CpuInfo info;
        info.cpu_id = cpu;
        info.core_id = cpu;
        info.online = true;
        info.l3_domain_id = 0;
        map.cpus.push_back(info);
    }
    return map;
}

inline TopologyMap discover_topology() {
#ifdef __linux__
    TopologyMap map = discover_topology_linux();
    if (map.allowed_cpus.empty()) {
        return discover_topology_fallback();
    }
    return map;
#else
    return discover_topology_fallback();
#endif
}

inline std::string topology_to_json(const TopologyMap& map) {
    std::ostringstream out;
    out << "{";
    out << "\"single_numa\":" << (map.single_numa ? "true" : "false") << ',';
    out << "\"symmetric_smt\":" << (map.symmetric_smt ? "true" : "false") << ',';
    out << "\"allowed_cpus\":[";
    for (std::size_t i = 0; i < map.allowed_cpus.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << map.allowed_cpus[i];
    }
    out << "],\"l3_domains\":[";
    for (std::size_t i = 0; i < map.l3_domains.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        const L3Domain& domain = map.l3_domains[i];
        out << '{';
        out << "\"id\":" << domain.id << ',';
        out << "\"numa_node\":" << domain.numa_node << ',';
        out << "\"logical_cpus\":[";
        for (std::size_t j = 0; j < domain.logical_cpus.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << domain.logical_cpus[j];
        }
        out << "],\"primary_cpus\":[";
        for (std::size_t j = 0; j < domain.primary_cpus.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << domain.primary_cpus[j];
        }
        out << "],\"smt_siblings\":[";
        for (std::size_t j = 0; j < domain.smt_siblings.size(); ++j) {
            if (j != 0) {
                out << ',';
            }
            out << domain.smt_siblings[j];
        }
        out << "]}";
    }
    out << "]}";
    return out.str();
}

}  // namespace

const TopologyMap& anvil_get_topology() {
    if (g_topology_initialized.load(std::memory_order_acquire)) {
        return g_topology_cache;
    }
    std::lock_guard<std::mutex> lock(g_topology_mu);
    if (!g_topology_initialized.load(std::memory_order_relaxed)) {
        g_topology_cache = discover_topology();
        g_topology_initialized.store(true, std::memory_order_release);
    }
    return g_topology_cache;
}

int anvil_refresh_topology() {
    std::lock_guard<std::mutex> lock(g_topology_mu);
    g_topology_cache = discover_topology();
    g_topology_initialized.store(true, std::memory_order_release);
    return 1;
}

int anvil_topology_export_json(char* out, int out_len) {
    if (out == nullptr || out_len <= 0) {
        return 0;
    }
    const std::string payload = topology_to_json(anvil_get_topology());
    if (static_cast<int>(payload.size()) + 1 > out_len) {
        return 0;
    }
    std::memcpy(out, payload.c_str(), payload.size());
    out[payload.size()] = '\0';
    return static_cast<int>(payload.size());
}

}  // namespace anvil::native

extern "C" {

int anvil_refresh_topology() {
    return anvil::native::anvil_refresh_topology();
}

int anvil_topology_export_json(char* out, int out_len) {
    return anvil::native::anvil_topology_export_json(out, out_len);
}

}  // extern "C"
