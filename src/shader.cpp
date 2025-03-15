// src/shader.cpp
#include "shader.h"
#include <GL/glew.h>
#include <fstream>
#include <sstream>
#include <iostream>

Shader::Shader(const char* vertexPath, const char* fragmentPath)
{
    // Read vertex shader
    std::ifstream vShaderFile(vertexPath);
    std::stringstream vShaderStream;
    vShaderStream << vShaderFile.rdbuf();
    vShaderFile.close();
    std::string vertexCode = vShaderStream.str();
    
    // Read fragment shader
    std::ifstream fShaderFile(fragmentPath);
    std::stringstream fShaderStream;
    fShaderStream << fShaderFile.rdbuf();
    fShaderFile.close();
    std::string fragmentCode = fShaderStream.str();
    
    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();
    
    // Compile shaders
    unsigned int vertex, fragment;
    
    // Vertex Shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    
    // Fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    
    // Shader Program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    
    // Delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Shader::use()
{
    glUseProgram(ID);
}

void Shader::setVec4(const std::string &name, float x, float y, float z, float w) const
{
    glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

void Shader::setMat4(const std::string &name, const float* value) const
{
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, value);
}