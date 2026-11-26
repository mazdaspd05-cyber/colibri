/* WILLNEED covers the whole tensor, not one device-readahead window.
 *
 * Linux clamps a single POSIX_FADV_WILLNEED to the device readahead (128 KiB on
 * the NVMe this was measured against; llama.cpp #32515). st_willneed_plan is the
 * split st_prefetch / st_prefetch_rep use. The plan is pure. On Linux the second
 * half drops a file from the page cache and checks that chunked advice leaves
 * more pages resident than one call over the same bytes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

#include "../st.h"

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while (0)

static int test_plan(void) {
    const int64_t chunk = 128 * 1024;
    const int64_t expert = 7 * 1024 * 1024;          /* ~7 MB expert tensor */
    int64_t off[64], len[64];
    int n = st_willneed_plan(4096, expert, chunk, off, len, 64);
    CHECK(n == 56);                                  /* 7 MiB / 128 KiB */
    CHECK(off[0] == 4096 && len[0] == chunk);
    int64_t covered = 0;
    for (int i = 0; i < n; i++) {
        CHECK(off[i] == 4096 + (int64_t)i * chunk);
        CHECK(len[i] == chunk);
        covered += len[i];
    }
    CHECK(covered == expert);

    /* short tail: the last window is the remainder, and the windows abut */
    const int64_t tail = expert + 100;
    n = st_willneed_plan(0, tail, chunk, off, len, 64);
    CHECK(n == 57);
    CHECK(len[56] == 100);
    CHECK(off[56] == expert);
    int64_t sum = 0;
    for (int i = 0; i < n; i++) {
        if (i) CHECK(off[i] == off[i - 1] + len[i - 1]);
        sum += len[i];
    }
    CHECK(sum == tail);

    CHECK(st_willneed_plan(10, 100, 1000, off, len, 4) == 1);
    CHECK(off[0] == 10 && len[0] == 100);
    CHECK(st_willneed_plan(0, 0, chunk, NULL, NULL, 0) == 0);
    CHECK(st_willneed_plan(0, 10, 0, NULL, NULL, 0) == -1);
    CHECK(st_willneed_plan(-1, 10, chunk, NULL, NULL, 0) == -1);
    CHECK(st_willneed_plan(0, -1, chunk, NULL, NULL, 0) == -1);
    /* count without a buffer */
    CHECK(st_willneed_plan(0, expert, chunk, NULL, NULL, 0) == 56);
    /* short buffer still reports the full count and writes only what fits */
    n = st_willneed_plan(0, expert, chunk, off, len, 2);
    CHECK(n == 56);
    CHECK(off[0] == 0 && off[1] == chunk);
    printf("  willneed plan: 7 MiB / 128 KiB -> 56 windows, tail and rejects ok\n");
    return 0;
}

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
static int resident_pages(void *p, size_t n) {
    long ps = sysconf(_SC_PAGESIZE);
    if (ps < 1) return -1;
    size_t pages = (n + (size_t)ps - 1) / (size_t)ps;
    unsigned char *v = malloc(pages);
    if (!v) return -1;
    if (mincore(p, n, v) != 0) { free(v); return -1; }
    int r = 0;
    for (size_t i = 0; i < pages; i++) if (v[i] & 1) r++;
    free(v);
    return r;
}
static int drop_and_hint(const char *path, size_t n, int chunked, int *out_pages) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    if (ftruncate(fd, (off_t)n) != 0) { close(fd); return -1; }
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); return -1; }
    memset(p, 0xA5, n);                              /* fault every page, then drop it */
    if (msync(p, n, MS_SYNC) != 0) { munmap(p, n); close(fd); return -1; }
    if (madvise(p, n, MADV_DONTNEED) != 0) { munmap(p, n); close(fd); return -1; }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    usleep(20000);
    int before = resident_pages(p, n);
    if (before < 0) { munmap(p, n); close(fd); return -1; }
    if (chunked) st_fadvise_willneed(fd, 0, (off_t)n);
    else posix_fadvise(fd, 0, (off_t)n, POSIX_FADV_WILLNEED);
    usleep(100000);
    int after = resident_pages(p, n);
    munmap(p, n);
    close(fd);
    if (after < 0) return -1;
    /* pages we could not evict are not evidence either way */
    if (before > (int)(n / 4096) / 20) { *out_pages = -1; return 0; }
    *out_pages = after;
    return 0;
}
static int test_residency(void) {
    const size_t n = 8u << 20;                       /* larger than one readahead window */
    char pa[] = "willneed_single.bin", pb[] = "willneed_chunked.bin";
    int single = 0, chunked = 0;
    if (drop_and_hint(pa, n, 0, &single) != 0 || drop_and_hint(pb, n, 1, &chunked) != 0) {
        fprintf(stderr, "willneed residency: setup failed (%s)\n", strerror(errno));
        unlink(pa); unlink(pb);
        return 1;
    }
    unlink(pa); unlink(pb);
    if (single < 0 || chunked < 0) {
        printf("  willneed residency: page cache would not drop, skipped\n");
        return 0;
    }
    long ps = sysconf(_SC_PAGESIZE);
    int total = (int)((n + (size_t)ps - 1) / (size_t)ps);
    printf("  willneed residency: one call %d/%d pages, chunked %d/%d pages\n",
           single, total, chunked, total);
    /* one call stops inside the device cap; the chunked hint covers the file */
    if (chunked < total * 9 / 10) {
        fprintf(stderr, "chunked WILLNEED left %d/%d pages resident\n", chunked, total);
        return 1;
    }
    if (single >= chunked) {
        fprintf(stderr, "one WILLNEED (%d) was not shorter than the chunked hint (%d); "
                        "this kernel did not cap the single call\n", single, chunked);
        return 1;
    }
    return 0;
}
#endif

int main(void) {
    int rc = test_plan();
#if defined(__linux__)
    if (!rc) rc = test_residency();
#endif
    if (rc) { fprintf(stderr, "test_willneed_plan: FAIL\n"); return 1; }
    printf("test_willneed_plan: ok\n");
    return 0;
}
