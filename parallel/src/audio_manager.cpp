#include "audio_manager.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace drumforge {

// Initialize static singleton instance
std::unique_ptr<AudioManager> AudioManager::instance = nullptr;

AudioManager::AudioManager()
    : sampleRate(44100)
    , isRecording(false) {
    std::cout << "AudioManager created" << std::endl;
}

AudioManager& AudioManager::getInstance() {
    if (!instance) {
        instance = std::unique_ptr<AudioManager>(new AudioManager());
    }
    return *instance;
}

void AudioManager::initialize(int sampleRate) {
    this->sampleRate = sampleRate;
    clearRecordBuffer();
    std::cout << "AudioManager initialized with sample rate: " << sampleRate << " Hz" << std::endl;
}

void AudioManager::startRecording() {
    clearRecordBuffer();
    isRecording = true;
    std::cout << "Recording started" << std::endl;
}

void AudioManager::stopRecording() {
    isRecording = false;
    std::cout << "Recording stopped, " << recordBuffer.size() << " samples captured" << std::endl;
}

void AudioManager::addSample(float sample) {
    if (isRecording) {
        recordBuffer.push_back(sample);
    }
}

void AudioManager::clearRecordBuffer() {
    recordBuffer.clear();
}

bool AudioManager::writeToWavFile(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // WAV header parameters
    const int numChannels = 1; // Mono
    const int bitsPerSample = 16;
    const int bytesPerSample = bitsPerSample / 8;
    const int dataSize = recordBuffer.size() * bytesPerSample;
    const int fileSize = 36 + dataSize;
    const int byteRate = sampleRate * numChannels * bytesPerSample;
    const int blockAlign = numChannels * bytesPerSample;
    
    // Write WAV header
    // RIFF header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&fileSize), 4);
    file.write("WAVE", 4);
    
    // Format chunk
    file.write("fmt ", 4);
    int fmtSize = 16;
    file.write(reinterpret_cast<const char*>(&fmtSize), 4);
    short audioFormat = 1; // PCM
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);
    file.write(reinterpret_cast<const char*>(&numChannels), 2);
    file.write(reinterpret_cast<const char*>(&sampleRate), 4);
    file.write(reinterpret_cast<const char*>(&byteRate), 4);
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);
    file.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
    
    // Data chunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&dataSize), 4);
    
    // Write audio data
    for (const float& sample : recordBuffer) {
        // Convert float [-1,1] to 16-bit PCM
        short pcmSample = static_cast<short>(sample * 32767.0f);
        file.write(reinterpret_cast<const char*>(&pcmSample), 2);
    }
    
    file.close();
    std::cout << "WAV file saved: " << filename << std::endl;
    return true;
}

} // namespace drumforge