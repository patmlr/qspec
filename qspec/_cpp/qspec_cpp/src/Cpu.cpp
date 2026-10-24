#include "Cpu.h"

#if defined(_M_X64) || defined(__x86_64__)

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif


// Registers eax, ebx, ecx, edx after executing cpuid.
static void cpuid(unsigned int leaf, unsigned int subleaf, unsigned int (&regs)[4])
{
#if defined(_MSC_VER)
	int info[4];
	__cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
	for (int i = 0; i < 4; ++i) regs[i] = static_cast<unsigned int>(info[i]);
#else
	__cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#endif
}

// The XCR0 register, which tells which register states the operating system saves on context switches.
static unsigned long long xgetbv0()
{
#if defined(_MSC_VER)
	return _xgetbv(0);
#else
	unsigned int eax, edx;
	__asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
	return (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
}

static bool bit(unsigned int reg, int i)
{
	return ((reg >> i) & 1u) != 0;
}

bool supports_x86_64_v3()
{
	unsigned int regs[4];
	cpuid(0, 0, regs);
	if (regs[0] < 7) return false;
	cpuid(0x80000000, 0, regs);
	if (regs[0] < 0x80000001) return false;

	cpuid(1, 0, regs);
	const unsigned int ecx_1 = regs[2];
	cpuid(7, 0, regs);
	const unsigned int ebx_7 = regs[1];
	cpuid(0x80000001, 0, regs);
	const unsigned int ecx_ext = regs[2];

	// xgetbv is only available if the operating system enabled it (OSXSAVE).
	if (!bit(ecx_1, 27)) return false;
	const bool os_saves_ymm = (xgetbv0() & 0x6) == 0x6;  // XMM and YMM registers.

	return os_saves_ymm
		&& bit(ecx_1, 12)  // FMA
		&& bit(ecx_1, 22)  // MOVBE
		&& bit(ecx_1, 28)  // AVX
		&& bit(ecx_1, 29)  // F16C
		&& bit(ebx_7, 3)  // BMI1
		&& bit(ebx_7, 5)  // AVX2
		&& bit(ebx_7, 8)  // BMI2
		&& bit(ecx_ext, 5);  // LZCNT
}

#else

bool supports_x86_64_v3()
{
	return false;
}

#endif
