#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace anvil::native {

struct CpuInfo {
    int cpu_id = -1;
    int core_id = -1;
    int package_id = -1;
    int die_id = -1;
    int cluster_id = -1;
    int numa_node = -1;
    int l3_domain_id = -1;
    bool online = false;
};

struct L3Domain {
    int id = -1;
    int numa_node = -1;
    std::vector<int> logical_cpus;
    std::vector<int> primary_cpus;
    std::vector<int> smt_siblings;
};

struct TopologyMap {
    std::vector<CpuInfo> cpus;
    std::vector<L3Domain> l3_domains;
    std::vector<int> allowed_cpus;
    bool single_numa = true;
    bool symmetric_smt = true;
};

const TopologyMap& anvil_get_topology();
int anvil_refresh_topology();
int anvil_topology_export_json(char* out, int out_len);

}  // namespace anvil::native

extern "C" {

int anvil_refresh_topology();
int anvil_topology_export_json(char* out, int out_len);

}
