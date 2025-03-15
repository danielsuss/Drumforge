#version 330 core
out vec4 FragColor;

uniform vec4 pointColor;

void main()
{
    FragColor = pointColor;
}