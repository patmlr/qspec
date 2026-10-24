#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <thread>
#include <vector>


// Parallel version of std::for_each(first, last, f) for random access iterators, distributing the elements
// dynamically over all hardware threads. Replaces std::execution::par, which Apple's libc++ does not implement
// and GCC's libstdc++ only runs in parallel when linked against TBB.
// As with std::execution::par, an exception thrown by f terminates the program.
template <typename RandomIt, typename F>
void parallel_for_each(RandomIt first, RandomIt last, F f)
{
	const size_t n = static_cast<size_t>(std::distance(first, last));
	const size_t n_threads = std::min<size_t>(n, std::max(1u, std::thread::hardware_concurrency()));

	std::atomic<size_t> next{0};
	auto work = [&first, &f, &next, n]()
	{
		for (size_t i = next.fetch_add(1, std::memory_order_relaxed); i < n; i = next.fetch_add(1, std::memory_order_relaxed))
			f(first[i]);
	};

	std::vector<std::thread> threads;
	threads.reserve(n_threads);
	for (size_t i = 1; i < n_threads; ++i)
		threads.emplace_back(work);
	work();
	for (std::thread& thread : threads)
		thread.join();
}
