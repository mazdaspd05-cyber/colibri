/* Populate a file mapping without a 4 KB fault loop.
 *
 * The default COLI_MMAP touch walks the expert one page at a time. Each fault
 * waits, and the kernel's readahead window stays a few requests deep (measured
 * queue depth ~4, 4 KB reads). MADV_POPULATE_READ (Linux 5.14+) asks for the
 * whole range in one call; the kernel submits it at the device's queue depth.
 * Where that advice does not exist, fall back to WILLNEED in 128 KiB steps and
 * a touch of each step, the same window st_fadvise_willneed uses.
 *
 * coli_mmap_cold marks pages reclaimable ahead of the resident working set
 * (MADV_COLD). It does not change a single byte. Both are no-ops on a NULL
 * pointer or a zero length. */
#ifndef COLI_MMAP_IO_H
#define COLI_MMAP_IO_H

#include <stddef.h>
#if !defined(_WIN32)
#include <sys/mman.h>
#endif

static int coli_mmap_populate(void *p, size_t n) {
    if (!p || !n) return 0;
#if defined(__linux__) && defined(MADV_POPULATE_READ)
    if (madvise(p, n, MADV_POPULATE_READ) == 0) return 0;
#endif
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    const size_t chunk = 128 * 1024;
    for (size_t o = 0; o < n; o += chunk) {
        size_t m = n - o < chunk ? n - o : chunk;
        madvise((char *)p + o, m, MADV_WILLNEED);
    }
#endif
    volatile char acc = 0;
    for (size_t o = 0; o < n; o += 4096) acc += ((volatile char *)p)[o];
    if (n) acc += ((volatile char *)p)[n - 1];
    (void)acc;
    return 0;
}

static void coli_mmap_cold(void *p, size_t n) {
#if defined(__linux__)
#ifndef MADV_COLD
#define MADV_COLD 20
#endif
    if (p && n) madvise(p, n, MADV_COLD);
#else
    (void)p; (void)n;
#endif
}
#endif
