#ifndef WORKING_DIRECTORY_H
#define WORKING_DIRECTORY_H

#include <string>

#ifdef USE_MPI
#include <mpi.h>

struct ParallelILSInfo {
    MPI_Comm ils_comm = MPI_COMM_SELF;
    int ils_group_rank = 0;
    int procs_per_ils = 1;
};

inline ParallelILSInfo& parallelILSInfoStorage() {
    static ParallelILSInfo info;
    return info;
}

inline void setParallelILSInfo(MPI_Comm comm, int group_rank, int procs_per_ils) {
    parallelILSInfoStorage() = {comm, group_rank, procs_per_ils};
}

inline const ParallelILSInfo& getParallelILSInfo() {
    return parallelILSInfoStorage();
}
#endif

inline std::string normalizeTunerWorkingDirectory(std::string directory) {
    if (directory.empty()) {
        directory = "./tuner_working_dir/";
    }
    if (directory.back() != '/') {
        directory.push_back('/');
    }
    return directory;
}

inline std::string& tunerWorkingDirectoryStorage() {
    static std::string directory = "./tuner_working_dir/";
    return directory;
}

inline void setTunerWorkingDirectory(const std::string& directory) {
    tunerWorkingDirectoryStorage() = normalizeTunerWorkingDirectory(directory);
}

inline const std::string& getTunerWorkingDirectory() {
    return tunerWorkingDirectoryStorage();
}

inline std::string buildTunerPath(const std::string& relative_path) {
    return getTunerWorkingDirectory() + relative_path;
}

#endif // WORKING_DIRECTORY_H
