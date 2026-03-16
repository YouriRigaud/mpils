#ifndef WORKING_DIRECTORY_H
#define WORKING_DIRECTORY_H

#include <string>

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
