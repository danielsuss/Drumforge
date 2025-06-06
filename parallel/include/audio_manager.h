#ifndef DRUMFORGE_AUDIO_MANAGER_H
#define DRUMFORGE_AUDIO_MANAGER_H

#include <string>
#include <vector>
#include <memory>

namespace drumforge {

/**
 * @brief Manages audio recording and playback
 * 
 * This class handles the recording of samples from the simulation
 * and saving them to WAV files.
 */
class AudioManager {
private:
    // Singleton instance
    static std::unique_ptr<AudioManager> instance;
    
    // Sample buffer for recording
    std::vector<float> recordBuffer;
    
    // Audio settings
    int sampleRate;
    bool isRecording;

    double accumulatedTime;    // Accumulated simulation time
    double sampleInterval;     // Time between samples (1/sampleRate)
    
    // Sample interpolation
    struct {
        float lastValue;       // Last sampled value
        float currentValue;    // Current sampled value 
        double lastSampleTime; // Time of last actual sample
    } interpolationState;
    
    struct AudioChannel {
        std::string name;         // Channel name (e.g., "Upper Membrane", "Body", etc.)
        float gain;               // Channel-specific gain
        bool enabled;             // Whether this channel is active
        float currentValue;       // Last received value for this channel
    };

    std::vector<AudioChannel> channels;      // All registered audio channels
    bool useChannelMixing;                   // Whether to mix all channels (true) or use discrete channels (false)
    float masterGain;                        // Global gain applied to all channels

    // Private constructor for singleton
    AudioManager();
    
public:
    // No copying or assignment
    AudioManager(const AudioManager&) = delete;
    AudioManager& operator=(const AudioManager&) = delete;
    
    // Get singleton instance
    static AudioManager& getInstance();
    
    // Initialize the audio system
    void initialize(int sampleRate = 44100);
    
    // Start recording
    void startRecording();
    
    // Stop recording
    void stopRecording();
    
    // Add a sample to the recording buffer
    void addSample(float sample);
    
    // Write recorded samples to a WAV file
    bool writeToWavFile(const std::string& filename);
    
    // Clear the recording buffer
    void clearRecordBuffer();
    
    // Check if recording is active
    bool getIsRecording() const { return isRecording; }
    
    // Get/set sample rate
    int getSampleRate() const { return sampleRate; }
    void setSampleRate(int rate) { sampleRate = rate; }
    
    // Get sample count
    size_t getSampleCount() const { return recordBuffer.size(); }
    
    // Get the record buffer
    const std::vector<float>& getRecordBuffer() const { return recordBuffer; }

    void processAudioStep(float simulationTimeStep, float currentSampleValue);

    float getInterpolatedSample(double time);

    int addChannel(const std::string& name, float gain = 1.0f);
    void removeChannel(int channelIndex);
    void setChannelGain(int channelIndex, float gain);
    void setChannelEnabled(int channelIndex, bool enabled);
    void setChannelName(int channelIndex, const std::string& name);
    void setChannelValue(int channelIndex, float value);
    int getChannelCount() const { return static_cast<int>(channels.size()); }
    const AudioChannel& getChannel(int channelIndex) const;

    // Channel data updating
    void processAudioStepForChannel(float simulationTimeStep, float currentSampleValue, int channelIndex);

    // Mixing settings
    void setUseChannelMixing(bool useMixing) { useChannelMixing = useMixing; }
    bool getUseChannelMixing() const { return useChannelMixing; }
    void setMasterGain(float gain) { masterGain = gain; }
    float getMasterGain() const { return masterGain; }

    void processMixedAudioStep(float simulationTimeStep);
};

} // namespace drumforge

#endif // DRUMFORGE_AUDIO_MANAGER_H