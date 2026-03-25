#include "cpu_math_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace {

constexpr int kEngineVersion = 1;

struct RankedCandidate {
  int kind = ANVIL_CPU_SCHEDULE_NONE;
  double score = 0.0;
};

int ClampNonNegative(int value) {
  return std::max(value, 0);
}

double ClampNonNegative(double value) {
  return std::max(value, 0.0);
}

int RiskBandFromScore(double score, double medium_threshold, double high_threshold) {
  if (score >= high_threshold) {
    return ANVIL_CPU_RISK_HIGH;
  }
  if (score >= medium_threshold) {
    return ANVIL_CPU_RISK_MEDIUM;
  }
  return ANVIL_CPU_RISK_LOW;
}

int ReuseClass(const AnvilCpuMathFeatures& features) {
  if (features.indirect_accesses > 0) {
    return ANVIL_CPU_REUSE_INDIRECT;
  }
  if (features.strided_accesses > 0) {
    return ANVIL_CPU_REUSE_STRIDED;
  }
  if (features.streaming_accesses > 0) {
    return ANVIL_CPU_REUSE_STREAMING;
  }
  if (features.temporal_accesses > 0) {
    return ANVIL_CPU_REUSE_CONTIGUOUS;
  }
  return ANVIL_CPU_REUSE_REGISTER;
}

void PushCandidate(std::array<RankedCandidate, 3>& ranked,
                   int& count,
                   int kind,
                   double score) {
  if (score <= 0.0 || kind == ANVIL_CPU_SCHEDULE_NONE) {
    return;
  }
  RankedCandidate candidate{kind, score};
  if (count < static_cast<int>(ranked.size())) {
    ranked[count++] = candidate;
  } else {
    auto min_it = std::min_element(
        ranked.begin(),
        ranked.end(),
        [](const RankedCandidate& lhs, const RankedCandidate& rhs) {
          return lhs.score < rhs.score;
        });
    if (min_it != ranked.end() && min_it->score < candidate.score) {
      *min_it = candidate;
    }
  }
}

}  // namespace

extern "C" void anvil_cpu_math_analyze(const AnvilCpuMathFeatures* features,
                                        AnvilCpuMathReport* report) {
  if (features == nullptr || report == nullptr) {
    return;
  }

  std::memset(report, 0, sizeof(AnvilCpuMathReport));
  report->engine_version = kEngineVersion;
  report->alignment_bytes = ClampNonNegative(features->preferred_alignment);

  const int lane_width_bytes = std::max(1, ClampNonNegative(features->lane_width_bits) / 8);
  const int lane_count = std::max(
      1,
      ClampNonNegative(features->vector_bits) /
          std::max(1, ClampNonNegative(features->lane_width_bits)));
  const int total_contiguous =
      ClampNonNegative(features->contiguous_reads) + ClampNonNegative(features->contiguous_writes);

  std::uint32_t blocker_mask = ANVIL_CPU_BLOCKER_NONE;
  if (!features->has_loop) {
    blocker_mask |= ANVIL_CPU_BLOCKER_NO_LOOP;
  }
  if (features->indirect_accesses > 0) {
    blocker_mask |= ANVIL_CPU_BLOCKER_INDIRECT;
  }
  if (features->has_recurrence && !features->has_reduction) {
    blocker_mask |= ANVIL_CPU_BLOCKER_RECURRENCE;
  }

  double vector_score = static_cast<double>(ClampNonNegative(features->structural_score));
  vector_score += static_cast<double>(total_contiguous * 2);
  vector_score += static_cast<double>(ClampNonNegative(features->has_reduction) * 2);
  vector_score -= static_cast<double>(ClampNonNegative(features->strided_accesses) * 3);
  vector_score -= static_cast<double>(ClampNonNegative(features->indirect_accesses) * 5);
  vector_score -= static_cast<double>(ClampNonNegative(features->function_call_count) * 2);
  vector_score -= ClampNonNegative(features->gather_penalty - 1.0) * 2.0;

  report->blocker_mask = blocker_mask;
  report->vector_legal = blocker_mask == ANVIL_CPU_BLOCKER_NONE;
  report->vector_profitable = report->vector_legal && vector_score >= 10.0;
  report->vector_score = vector_score;
  report->recommended_lane_count = report->vector_profitable ? lane_count : 1;

  report->prefetch_streaming = features->streaming_accesses > 0 ? 1 : 0;
  report->prefetch_conservative = features->strided_accesses > 0 ? 1 : 0;
  report->prefetch_avoid_indirect = features->indirect_accesses > 0 ? 1 : 0;
  report->prefetch_no_runway = features->has_loop ? 0 : 1;
  report->prefetch_distance_lines =
      report->prefetch_streaming ? std::max(1, features->prefetch_distance) : 0;
  report->prefetch_recommended =
      report->prefetch_streaming && !report->prefetch_avoid_indirect && !report->prefetch_no_runway;

  report->estimated_ops = ClampNonNegative(features->operator_count) +
                          (ClampNonNegative(features->function_call_count) * 2) +
                          ClampNonNegative(features->has_reduction);

  const int weighted_accesses =
      std::max(
          1,
          ClampNonNegative(features->contiguous_reads) +
              ClampNonNegative(features->contiguous_writes) +
              (ClampNonNegative(features->strided_accesses) * 2) +
              (ClampNonNegative(features->indirect_accesses) * 4) +
              ClampNonNegative(features->accumulate_writes));
  report->estimated_bytes = std::max(1, weighted_accesses * lane_width_bytes);
  report->operational_intensity =
      static_cast<double>(report->estimated_ops) /
      static_cast<double>(report->estimated_bytes);
  if (report->operational_intensity < 0.25) {
    report->bound_class = ANVIL_CPU_BOUND_MEMORY;
  } else if (report->operational_intensity < 1.0) {
    report->bound_class = ANVIL_CPU_BOUND_BALANCED;
  } else {
    report->bound_class = ANVIL_CPU_BOUND_COMPUTE;
  }

  const int cache_line_bytes = std::max(1, ClampNonNegative(features->cache_line_bytes));
  const int loop_scale = features->has_loop ? lane_count : 1;
  const int estimated_bytes_touched =
      std::max(
          lane_width_bytes,
          ((total_contiguous * lane_width_bytes) + (ClampNonNegative(features->strided_accesses) * cache_line_bytes) +
           (ClampNonNegative(features->indirect_accesses) * cache_line_bytes * 2)) *
              loop_scale);
  report->estimated_cache_lines_touched =
      std::max(1, (estimated_bytes_touched + cache_line_bytes - 1) / cache_line_bytes);
  report->reuse_distance_class = ReuseClass(*features);

  const double l1_pressure =
      static_cast<double>(report->estimated_cache_lines_touched) +
      static_cast<double>(ClampNonNegative(features->strided_accesses) * 1.5) +
      static_cast<double>(ClampNonNegative(features->indirect_accesses) * 3);
  const double l2_pressure =
      l1_pressure + static_cast<double>(ClampNonNegative(features->streaming_accesses));
  const double l3_pressure =
      l2_pressure + static_cast<double>(ClampNonNegative(features->native_execution_domain) * 2);
  report->l1_risk = RiskBandFromScore(l1_pressure, 4.0, 10.0);
  report->l2_risk = RiskBandFromScore(l2_pressure, 6.0, 14.0);
  report->l3_risk = RiskBandFromScore(l3_pressure, 8.0, 18.0);
  report->memory_pressure_score =
      (l1_pressure * 0.2) + (l2_pressure * 0.3) + (l3_pressure * 0.5);

  report->register_pressure_score =
      ClampNonNegative(features->symbol_count) +
      (ClampNonNegative(features->max_nesting_depth) * 2) +
      (ClampNonNegative(features->function_call_count) * 3) +
      ClampNonNegative(features->access_count);
  report->register_pressure_band =
      RiskBandFromScore(static_cast<double>(report->register_pressure_score), 10.0, 18.0);
  report->spill_risk =
      report->register_pressure_score >
      std::max(8, lane_count + 4);

  std::array<RankedCandidate, 3> ranked{};
  int schedule_count = 0;
  if (report->vector_legal) {
    PushCandidate(
        ranked,
        schedule_count,
        ANVIL_CPU_SCHEDULE_VECTORIZE,
        report->vector_profitable ? 0.9 : 0.55);
  }
  if (report->prefetch_streaming || report->bound_class == ANVIL_CPU_BOUND_MEMORY) {
    PushCandidate(
        ranked,
        schedule_count,
        ANVIL_CPU_SCHEDULE_CACHE_BLOCK,
        report->bound_class == ANVIL_CPU_BOUND_MEMORY ? 0.75 : 0.5);
  }
  if (features->has_reduction) {
    PushCandidate(ranked, schedule_count, ANVIL_CPU_SCHEDULE_TREE_REDUCE, 0.7);
  }
  if (!report->vector_legal && (blocker_mask & ANVIL_CPU_BLOCKER_RECURRENCE) != 0u) {
    PushCandidate(ranked, schedule_count, ANVIL_CPU_SCHEDULE_SCALAR_STABILIZE, 0.45);
  }

  std::sort(
      ranked.begin(),
      ranked.begin() + schedule_count,
      [](const RankedCandidate& lhs, const RankedCandidate& rhs) {
        return lhs.score > rhs.score;
      });
  report->schedule_count = schedule_count;
  for (int index = 0; index < schedule_count; ++index) {
    report->schedules[index].kind = ranked[index].kind;
    report->schedules[index].score = ranked[index].score;
  }

  double benchmark_priority = static_cast<double>(ClampNonNegative(features->structural_score));
  if (report->vector_profitable) {
    benchmark_priority += 6.0;
  } else if (report->vector_legal) {
    benchmark_priority += 2.0;
  }
  if (report->bound_class == ANVIL_CPU_BOUND_MEMORY) {
    benchmark_priority += 4.0;
  }
  if (report->spill_risk) {
    benchmark_priority += 3.0;
  }
  if (features->native_execution_domain) {
    benchmark_priority += 5.0;
  }
  if (report->l3_risk == ANVIL_CPU_RISK_HIGH) {
    benchmark_priority += 2.0;
  } else if (report->l3_risk == ANVIL_CPU_RISK_MEDIUM) {
    benchmark_priority += 1.0;
  }
  report->benchmark_priority = benchmark_priority;
}

extern "C" const char* anvil_cpu_math_runtime_version() {
  return "anvil.cpu_math.v1";
}
