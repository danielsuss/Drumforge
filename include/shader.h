#ifndef DRUMFORGE_SHADER_H
#define DRUMFORGE_SHADER_H

#include <string>
#include <glm/glm.hpp>

class Shader {
private:
    unsigned int ID;

public:
    // Constructor reads and builds the shader
    Shader(const char* vertexPath, const char* fragmentPath);
    
    // Use/activate the shader
    void use();
    
    // Utility uniform functions - only what we need for now
    void setVec4(const std::string &name, const glm::vec4 &value) const;
    void setVec4(const std::string &name, float x, float y, float z, float w) const;
    void setMat4(const std::string &name, const glm::mat4 &mat) const;

    // Get the program ID
    unsigned int getID() const { return ID; }
};

#endif // DRUMFORGE_SHADER_H