#pragma once


// Whether the CPU and the operating system support all instructions of the x86-64-v3 level
// (AVX, AVX2, FMA, BMI1, BMI2, LZCNT, MOVBE, F16C) which the AVX2 build of this library may use.
// Always false on other architectures.
bool supports_x86_64_v3();
