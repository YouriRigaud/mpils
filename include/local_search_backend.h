// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef LOCAL_SEARCH_BACKEND_H
#define LOCAL_SEARCH_BACKEND_H

#include <cstdint>
#include <stdexcept>
#include <string>

enum class LocalSearchBackend {
    IteratedLocalSearch,
    ParamILS
};

inline std::string localSearchBackendToString(LocalSearchBackend backend) {
    switch (backend) {
        case LocalSearchBackend::IteratedLocalSearch:
            return "iterated_local_search";
        case LocalSearchBackend::ParamILS:
            return "paramils";
    }

    return "iterated_local_search";
}

inline LocalSearchBackend parseLocalSearchBackend(const std::string& value) {
    if (value == "iterated_local_search") {
        return LocalSearchBackend::IteratedLocalSearch;
    }
    if (value == "paramils") {
        return LocalSearchBackend::ParamILS;
    }

    throw std::runtime_error("Unknown local search backend: " + value);
}

inline std::uint32_t computeLocalSearchRunSeed(std::uint32_t base_seed, int iteration, int nb_workers, int worker_id) {
    const std::uint64_t run_seed =
        static_cast<std::uint64_t>(base_seed) +
        static_cast<std::uint64_t>(iteration - 1) * static_cast<std::uint64_t>(nb_workers) +
        static_cast<std::uint64_t>(worker_id);

    return static_cast<std::uint32_t>(run_seed);
}

#endif // LOCAL_SEARCH_BACKEND_H
