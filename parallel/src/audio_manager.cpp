#include "audio_manager.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace drumforge {

// Initialize static singleton instance
std::unique_ptr<AudioManager> AudioManager::instance = nullptr;

AudioManager::AudioManager()
    : sampleRate(44100)
    , isRecording(false)
    , accumulatedTime(0.0)
    , sampleInterval(0.0)
    , useChannelMixing(true)
    , masterGain(1.0f) {
    // Initialize interpolation state
    interpolationState.lastValue = 0.0f;
    interpolationState.currentValue = 0.0f;
    interpolationState.lastSampleTime = 0.0;
    
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
    sampleInterval = 1.0 / sampleRate;
    clearRecordBuffer();
    
    // Clear all channels too
    channels.clear();

    // Reset time tracking
    accumulatedTime = 0.0;
    
    std::cout << "AudioManager initialized with sample rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "Sample interval: " << sampleInterval << " seconds" << std::endl;
}

// Add channel
int AudioManager::addChannel(const std::string& name, float gain) {
    AudioChannel channel;
    channel.name = name;
    channel.gain = gain;
    channel.enabled = true;
    channel.currentValue = 0.0f;
    
    channels.push_back(channel);
    std::cout << "Added audio channel '" << name << "' with gain " << gain << std::endl;
    
    return channels.size() - 1;
}

// Remove channel
void AudioManager::removeChannel(int channelIndex) {
    if (channelIndex >= 0 && channelIndex < static_cast<int>(channels.size())) {
        std::cout << "Removed audio channel '" << channels[channelIndex].name << "'" << std::endl;
        channels.erase(channels.begin() + channelIndex);
    }
}

// Set channel gain
void AudioManager::setChannelGain(int channelIndex, float gain) {
    if (channelIndex >= 0 && channelIndex < static_cast<int>(channels.size())) {
        channels[channelIndex].gain = gain;
    }
}

// Set channel enabled state
void AudioManager::setChannelEnabled(int channelIndex, bool enabled) {
    if (channelIndex >= 0 && channelIndex < static_cast<int>(channels.size())) {
        channels[channelIndex].enabled = enabled;
    }
}

// Set channel name
void AudioManager::setChannelName(int channelIndex, const std::string& name) {
    if (channelIndex >= 0 && channelIndex < static_cast<int>(channels.size())) {
        channels[channelIndex].name = name;
    }
}

// Get channel
const AudioManager::AudioChannel& AudioManager::getChannel(int channelIndex) const {
    static const AudioChannel defaultChannel = {"Invalid", 1.0f, false, 0.0f};
    
    if (channelIndex >= 0 && channelIndex < static_cast<int>(channels.size())) {
        return channels[channelIndex];
    }
    return defaultChannel;
}

// Process audio for a specific channel
void AudioManager::processAudioStepForChannel(float simulationTimeStep, float currentSampleValue, int channelIndex) {
    if (!isRecording || channelIndex < 0 || channelIndex >= static_cast<int>(channels.size())) {
        return;
    }
    
    // Just update the channel's current value, don't process audio yet
    channels[channelIndex].currentValue = currentSampleValue;
    
    // If we're mixing, we don't generate samples yet - that will be done in a separate step
    if (useChannelMixing) {
        return;
    }
    
    // If not mixing channels, immediately process this sample
    if (channels[channelIndex].enabled) {
        // Scale the sample by the channel's gain
        float scaledSample = currentSampleValue * channels[channelIndex].gain;
        
        // Pass to the sample processor
        processAudioStep(simulationTimeStep, scaledSample);
    }
}

void AudioManager::processMixedAudioStep(float simulationTimeStep) {
    if (!isRecording || channels.empty() || !useChannelMixing) {
        return;
    }
    
    // Calculate mixed sample
    float mixedSample = 0.0f;
    int activeChannels = 0;
    
    for (const auto& channel : channels) {
        if (channel.enabled) {
            mixedSample += channel.currentValue * channel.gain;
            activeChannels++;
        }
    }
    
    // Normalize by number of active channels and apply master gain
    if (activeChannels > 0) {
        // Use a sqrt scaling to preserve energy better when mixing
        mixedSample /= sqrt(static_cast<float>(activeChannels));
        mixedSample *= masterGain;
        
        // Process the mixed sample
        processAudioStep(simulationTimeStep, mixedSample);
    }
}

void AudioManager::startRecording() {
    clearRecordBuffer();
    accumulatedTime = 0.0;
    isRecording = true;
    
    // Reset interpolation state
    interpolationState.lastValue = 0.0f;
    interpolationState.currentValue = 0.0f;
    interpolationState.lastSampleTime = 0.0;
    
    std::cout << "Recording started" << std::endl;
}

void AudioManager::stopRecording() {
    isRecording = false;
    std::cout << "Recording stopped, " << recordBuffer.size() << " samples captured" << std::endl;
    std::cout << "Recording duration: " << (recordBuffer.size() / static_cast<double>(sampleRate)) << " seconds" << std::endl;
}

void AudioManager::addSample(float sample) {
    if (isRecording) {
        recordBuffer.push_back(sample);
    }
}

void AudioManager::clearRecordBuffer() {
    recordBuffer.clear();
}

void AudioManager::processAudioStep(float simulationTimeStep, float currentSampleValue) {
    if (!isRecording) return;
    
    // Update interpolation state with new value
    interpolationState.lastValue = interpolationState.currentValue;
    interpolationState.currentValue = currentSampleValue;
    
    // Update time tracking
    double previousTime = accumulatedTime;
    accumulatedTime += simulationTimeStep;
    
    // For the first sample, just store the value without interpolation
    if (recordBuffer.empty()) {
        recordBuffer.push_back(currentSampleValue);
        interpolationState.lastSampleTime = 0.0;
        return;
    }
    
    interpolationState.lastSampleTime = previousTime;
    
    // Calculate how many audio samples we need for this time step
    double nextSampleTime = (recordBuffer.size() * sampleInterval);
    
    // Generate all audio samples needed to cover this simulation step
    while (nextSampleTime <= accumulatedTime) {
        // Get interpolated sample at exactly the right time point
        float sample = getInterpolatedSample(nextSampleTime);
        
        // Add to record buffer
        recordBuffer.push_back(sample);
        
        // Move to next sample time
        nextSampleTime = (recordBuffer.size() * sampleInterval);
    }
}

float AudioManager::getInterpolatedSample(double time) {
    // If we're at the very beginning, just return current value
    if (interpolationState.lastSampleTime <= 0.0) {
        return interpolationState.currentValue;
    }
    
    // Calculate simulation interval - the time between the last two simulation steps
    double simulationInterval = accumulatedTime - interpolationState.lastSampleTime;
    
    if (simulationInterval <= 0.0) {
        return interpolationState.currentValue;
    }
    
    // Calculate interpolation factor (how far we are between previous and current time)
    double t = (time - interpolationState.lastSampleTime) / simulationInterval;
    t = std::max(0.0, std::min(1.0, t)); // Clamp to [0,1]
    
    // Linear interpolation
    return interpolationState.lastValue * (1.0f - static_cast<float>(t)) + 
           interpolationState.currentValue * static_cast<float>(t);
}

bool AudioManager::writeToWavFile(const std::string& filename) {
    if (recordBuffer.empty()) {
        std::cerr << "Warning: No samples to write to WAV file" << std::endl;
        return false;
    }
    
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
    
    // Find maximum amplitude for normalization
    float maxAmplitude = 0.0f;
    for (const float& sample : recordBuffer) {
        maxAmplitude = std::max(maxAmplitude, std::abs(sample));
    }
    
    // Normalize only if necessary
    float normalizationFactor = 1.0f;
    if (maxAmplitude > 1.0f) {
        normalizationFactor = 1.0f / maxAmplitude;
        std::cout << "Normalizing audio (max amplitude: " << maxAmplitude << ")" << std::endl;
    } else if (maxAmplitude < 0.1f) {
        // If very quiet, boost the signal
        normalizationFactor = 0.9f / maxAmplitude;
        std::cout << "Boosting quiet audio (max amplitude: " << maxAmplitude << ")" << std::endl;
    }
    
    // Write audio data
    for (const float& sample : recordBuffer) {
        // Normalize, then convert float [-1,1] to 16-bit PCM
        float normalizedSample = sample * normalizationFactor;
        // Clamp to [-1,1] range
        normalizedSample = std::max(-1.0f, std::min(normalizedSample, 1.0f));
        short pcmSample = static_cast<short>(normalizedSample * 32767.0f);
        file.write(reinterpret_cast<const char*>(&pcmSample), 2);
    }
    
    file.close();
    std::cout << "WAV file saved: " << filename << std::endl;
    std::cout << "Duration: " << (recordBuffer.size() / static_cast<double>(sampleRate)) << " seconds" << std::endl;
    return true;
}

} // namespace drumforge