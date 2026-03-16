#include "hardware_telemetry.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <array>
#include <cstring>

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#include <Powrprof.h>
#include <psapi.h>
#include <comdef.h>
#include <Wbemidl.h>
#pragma comment(lib, "powrprof.lib")
#pragma comment(lib, "wbemuuid.lib")
#else
#include <unistd.h>
#include <sched.h>
#include <sys/resource.h>
#include <dirent.h>
#include <sys/types.h>
#include <cpuid.h>
#endif

// --- Helper functions ---
#ifndef _WIN32
namespace {
void write_to_sysfs(const std::string& path, const std::string& value) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        return; // Fail silently
    }
    ofs << value;
    ofs.close();
}

enum class CpuVendor { UNKNOWN, INTEL, AMD };

CpuVendor get_cpu_vendor() {
    std::array<int, 4> cpuid_info;
    char vendor_id[13];
    
    __cpuid(0, cpuid_info[0], cpuid_info[1], cpuid_info[2], cpuid_info[3]);
    memcpy(vendor_id, &cpuid_info[1], 4);
    memcpy(vendor_id + 4, &cpuid_info[3], 4);
    memcpy(vendor_id + 8, &cpuid_info[2], 4);
    vendor_id[12] = '\0';
    
    std::string vendor_str(vendor_id);
    if (vendor_str == "GenuineIntel") return CpuVendor::INTEL;
    if (vendor_str == "AuthenticAMD") return CpuVendor::AMD;
    return CpuVendor::UNKNOWN;
}

std::string find_amd_hwmon_path() {
    std::string base_path = "/sys/class/hwmon/";
    DIR* dp = opendir(base_path.c_str());
    if (dp == nullptr) return "";

    dirent* de;
    while ((de = readdir(dp)) != nullptr) {
        std::string dir_name = de->d_name;
        if (dir_name.rfind("hwmon", 0) == 0) {
            std::ifstream name_file(base_path + dir_name + "/name");
            std::string hwmon_name;
            if (name_file >> hwmon_name && (hwmon_name == "k10temp" || hwmon_name == "zenpower")) {
                std::string power_path = base_path + dir_name + "/power1_input";
                if (std::ifstream(power_path).good()) {
                    closedir(dp);
                    return power_path;
                }
            }
        }
    }
    closedir(dp);
    return "";
}
} // anonymous namespace
#endif

// --- Constructor & Destructor ---
HardwareTelemetry::HardwareTelemetry() {
    monitoring_thread_ = std::thread(&HardwareTelemetry::monitoring_thread_func, this);
}

HardwareTelemetry::~HardwareTelemetry() {
    stop_monitoring_ = true;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
}

// --- Getters for Monitored State ---
float HardwareTelemetry::get_cpu_temperature() const { return cpu_temperature_.load(); }
float HardwareTelemetry::get_cpu_power_draw() const { return cpu_power_draw_.load(); }
float HardwareTelemetry::get_memory_usage_mb() const { return memory_usage_mb_.load(); }

void HardwareTelemetry::monitoring_thread_func() {
#ifndef _WIN32
    CpuVendor vendor = get_cpu_vendor();
    long long last_energy_uj = 0;
    auto last_energy_time = std::chrono::steady_clock::now();
    std::string amd_power_path;
    if (vendor == CpuVendor::AMD) {
        amd_power_path = find_amd_hwmon_path();
    }
#endif

    while (!stop_monitoring_) {
#ifdef __linux__
        // --- Read CPU Temperature ---
        float total_temp = 0;
        int temp_count = 0;
        DIR* dp_thermal = opendir("/sys/class/thermal");
        if (dp_thermal) {
            dirent* de;
            while ((de = readdir(dp_thermal)) != nullptr) {
                if (std::string(de->d_name).rfind("thermal_zone", 0) == 0) {
                    std::ifstream temp_file(std::string("/sys/class/thermal/") + de->d_name + "/temp");
                    float temp;
                    if (temp_file >> temp) {
                        total_temp += temp / 1000.0f;
                        temp_count++;
                    }
                }
            }
            closedir(dp_thermal);
        }
        if (temp_count > 0) cpu_temperature_ = total_temp / temp_count;

        // --- Read CPU Power (Vendor Specific) ---
        if (vendor == CpuVendor::INTEL) {
            std::ifstream power_file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
            long long current_energy_uj;
            if (power_file >> current_energy_uj) {
                auto current_time = std::chrono::steady_clock::now();
                if (last_energy_uj > 0) {
                    double time_delta_s = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_energy_time).count() / 1.0e6;
                    long long energy_delta_uj = current_energy_uj - last_energy_uj;
                    if (time_delta_s > 0) {
                        cpu_power_draw_ = (energy_delta_uj / 1.0e6) / time_delta_s;
                    }
                }
                last_energy_uj = current_energy_uj;
                last_energy_time = current_time;
            }
        } else if (vendor == CpuVendor::AMD && !amd_power_path.empty()) {
            std::ifstream power_file(amd_power_path);
            long long power_uw;
            if (power_file >> power_uw) {
                cpu_power_draw_ = power_uw / 1.0e6;
            }
        }

        // --- Read Memory Usage ---
        std::ifstream statm_file("/proc/self/statm");
        long rss_pages;
        statm_file.ignore(256, ' ');
        if (statm_file >> rss_pages) {
            memory_usage_mb_ = rss_pages * sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
        }
#elif _WIN32
        // Windows implementation would go here...
#endif
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
