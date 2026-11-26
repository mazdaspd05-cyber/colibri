/* COLI_MMAP_IO's populate reads a whole mapping, and cold does not change bytes.
 *
 * The default mmap expert touch faults one 4 KB page at a time. coli_mmap_populate
 * asks for the range at once (MADV_POPULATE_READ, else 128 KiB WILLNEED). A
 * prefill then marks those pages cold; the bytes stay what they were. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "../mmap_io.h"

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while (0)

#if defined(__linux__)
static int resident_pages(void *p, size_t n) {
    long ps = sysconf(_SC_PAGESIZE);
    size_t pages = (n + (size_t)ps - 1) / (size_t)ps;
    unsigned char *v = malloc(pages);
    if (!v || mincore(p, n, v) != 0) { free(v); return -1; }
    int r = 0;
    for (size_t i = 0; i < pages; i++) if (v[i] & 1) r++;
    free(v);
    return r;
}
#endif

int main(void) {
#if !defined(__linux__)
    printf("test_mmap_populate: not Linux, helpers are no-ops\n");
    coli_mmap_populate(NULL, 0);
    coli_mmap_cold(NULL, 0);
    printf("test_mmap_populate: ok\n");
    return 0;
#else
    const size_t n = 8u << 20;
    int fd = open("mmap_populate.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0);
    CHECK(ftruncate(fd, (off_t)n) == 0);
    unsigned char *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK(p != MAP_FAILED);
    for (size_t i = 0; i < n; i += 4096) p[i] = (unsigned char)(i / 4096);
    CHECK(msync(p, n, MS_SYNC) == 0);
    CHECK(madvise(p, n, MADV_DONTNEED) == 0);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    usleep(20000);
    int before = resident_pages(p, n);
    CHECK(before >= 0);
    if (before > (int)(n / 4096) / 20) {
        printf("  populate: page cache would not drop, skipped\n");
        munmap(p, n); close(fd); unlink("mmap_populate.bin");
        printf("test_mmap_populate: ok\n");
        return 0;
    }
    CHECK(coli_mmap_populate(p, n) == 0);
    int after = resident_pages(p, n);
    long ps = sysconf(_SC_PAGESIZE);
    int total = (int)((n + (size_t)ps - 1) / (size_t)ps);
    printf("  populate: %d -> %d / %d pages\n", before, after, total);
    CHECK(after >= total * 9 / 10);
    unsigned char b0 = p[0];
    coli_mmap_cold(p, n);
    CHECK(p[0] == b0);
    CHECK(p[4096] == 1);
    munmap(p, n);
    close(fd);
    unlink("mmap_populate.bin");
    printf("test_mmap_populate: ok\n");
    return 0;
#endif
}
