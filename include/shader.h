// include/shader.h
#ifndef DRUMFORGE_SHADER_H
#define DRUMFORGE_SHADER_H

#include <string>

class Shader {
private:
    unsigned int ID;

public:
    // Constructor reads and builds the shader
    Shader(const char* vertexPath, const char* fragmentPath);
    
    // Use/activate the shader
    void use();
    
    // Utility uniform functions
    void setVec4(const std::string &name, float x, float y, float z, float w) const;
    void setMat4(const std::string &name, const float* value) const;

    // Get the program ID
    unsigned int getID() const { return ID; }
};

#endif // DRUMFORGE_SHADER_H