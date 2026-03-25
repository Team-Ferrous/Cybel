#ifndef ANVIL_CPU_MATH_RUNTIME_H_
#define ANVIL_CPU_MATH_RUNTIME_H_

#include <cstdint>

extern "C" {

enum AnvilCpuMathBlockerMask : std::uint32_t {
  ANVIL_CPU_BLOCKER_NONE = 0u,
  ANVIL_CPU_BLOCKER_NO_LOOP = 1u << 0,
  ANVIL_CPU_BLOCKER_INDIRECT = 1u << 1,
  ANVIL_CPU_BLOCKER_RECURRENCE = 1u << 2,
};

enum AnvilCpuMathBoundClass : std::int32_t {
  ANVIL_CPU_BOUND_MEMORY = 0,
  ANVIL_CPU_BOUND_BALANCED = 1,
  ANVIL_CPU_BOUND_COMPUTE = 2,
};

enum AnvilCpuMathRiskBand : std::int32_t {
  ANVIL_CPU_RISK_LOW = 0,
  ANVIL_CPU_RISK_MEDIUM = 1,
  ANVIL_CPU_RISK_HIGH = 2,
};

enum AnvilCpuMathReuseClass : std::int32_t {
  ANVIL_CPU_REUSE_REGISTER = 0,
  ANVIL_CPU_REUSE_CONTIGUOUS = 1,
  ANVIL_CPU_REUSE_STREAMING = 2,
  ANVIL_CPU_REUSE_STRIDED = 3,
  ANVIL_CPU_REUSE_INDIRECT = 4,
};

enum AnvilCpuMathScheduleKind : std::int32_t {
  ANVIL_CPU_SCHEDULE_NONE = 0,
  ANVIL_CPU_SCHEDULE_VECTORIZE = 1,
  ANVIL_CPU_SCHEDULE_CACHE_BLOCK = 2,
  ANVIL_CPU_SCHEDULE_TREE_REDUCE = 3,
  ANVIL_CPU_SCHEDULE_SCALAR_STABILIZE = 4,
};

struct AnvilCpuMathFeatures {
  std::int32_t structural_score;
  std::int32_t operator_count;
  std::int32_t symbol_count;
  std::int32_t function_call_count;
  std::int32_t max_nesting_depth;
  std::int32_t access_count;
  std::int32_t contiguous_reads;
  std::int32_t contiguous_writes;
  std::int32_t strided_accesses;
  std::int32_t indirect_accesses;
  std::int32_t streaming_accesses;
  std::int32_t temporal_accesses;
  std::int32_t accumulate_writes;
  std::int32_t has_loop;
  std::int32_t has_recurrence;
  std::int32_t has_reduction;
  std::int32_t native_execution_domain;
  std::int32_t vector_bits;
  std::int32_t lane_width_bits;
  std::int32_t preferred_alignment;
  std::int32_t cache_line_bytes;
  std::int32_t prefetch_distance;
  double gather_penalty;
};

struct AnvilCpuMathScheduleCandidate {
  std::int32_t kind;
  double score;
};

struct AnvilCpuMathReport {
  std::int32_t engine_version;
  std::int32_t vector_legal;
  std::int32_t vector_profitable;
  std::int32_t recommended_lane_count;
  std::uint32_t blocker_mask;
  double vector_score;
  std::int32_t alignment_bytes;
  std::int32_t prefetch_recommended;
  std::int32_t prefetch_distance_lines;
  std::int32_t prefetch_streaming;
  std::int32_t prefetch_conservative;
  std::int32_t prefetch_avoid_indirect;
  std::int32_t prefetch_no_runway;
  std::int32_t estimated_ops;
  std::int32_t estimated_bytes;
  double operational_intensity;
  std::int32_t bound_class;
  std::int32_t estimated_cache_lines_touched;
  std::int32_t reuse_distance_class;
  std::int32_t l1_risk;
  std::int32_t l2_risk;
  std::int32_t l3_risk;
  double memory_pressure_score;
  std::int32_t register_pressure_score;
  std::int32_t register_pressure_band;
  std::int32_t spill_risk;
  double benchmark_priority;
  std::int32_t schedule_count;
  AnvilCpuMathScheduleCandidate schedules[3];
};

void anvil_cpu_math_analyze(const AnvilCpuMathFeatures* features,
                            AnvilCpuMathReport* report);
const char* anvil_cpu_math_runtime_version();

}  // extern "C"

#endif  // ANVIL_CPU_MATH_RUNTIME_H_
