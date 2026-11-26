/* The OpenMP re-exec must not discard an inherited CPU pin.
 *
 * execv keeps the calling thread's affinity mask. Widening that mask to every
 * online CPU dropped taskset and systemd CPUAffinity. COLI_CPUS installs an
 * explicit list. A process that already had OMP_PROC_BIND or OMP_PLACES still
 * widens (#471: libgomp bound this thread to place 0 before main). */
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "../affinity.h"

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } \
} while (0)

static int test_action(void) {
    CHECK(coli_affinity_reexec_action(NULL, 0) == COLI_AFF_KEEP);
    CHECK(coli_affinity_reexec_action("", 0) == COLI_AFF_KEEP);
    CHECK(coli_affinity_reexec_action(NULL, 1) == COLI_AFF_RESET_ONLINE);
    CHECK(coli_affinity_reexec_action("", 1) == COLI_AFF_RESET_ONLINE);
    CHECK(coli_affinity_reexec_action("0-3", 0) == COLI_AFF_APPLY_LIST);
    CHECK(coli_affinity_reexec_action("0-3", 1) == COLI_AFF_APPLY_LIST);
    printf("  reexec action: keep / reset-online / COLI_CPUS ok\n");
    return 0;
}

#if defined(__linux__) && defined(CPU_SETSIZE)
#include <errno.h>

static int cpu_in(const cpu_set_t *s, int cpu) { return CPU_ISSET(cpu, s); }

static int test_parse(void) {
    long ncpu = coli_affinity_ncpu();
    CHECK(ncpu >= 1);
    cpu_set_t set;
    CHECK(coli_parse_cpu_list("0", &set, ncpu) == 0);
    CHECK(cpu_in(&set, 0));
    CHECK(CPU_COUNT(&set) == 1);

    if (ncpu >= 4) {
        CHECK(coli_parse_cpu_list("0-2, 3", &set, ncpu) == 0);
        CHECK(CPU_COUNT(&set) == 4);
        for (int i = 0; i < 4; i++) CHECK(cpu_in(&set, i));
        CHECK(coli_parse_cpu_list("1-1", &set, ncpu) == 0);
        CHECK(CPU_COUNT(&set) == 1 && cpu_in(&set, 1));
    }
    CHECK(coli_parse_cpu_list("", &set, ncpu) != 0);
    CHECK(coli_parse_cpu_list("x", &set, ncpu) != 0);
    CHECK(coli_parse_cpu_list("3-1", &set, ncpu) != 0);
    CHECK(coli_parse_cpu_list("0-", &set, ncpu) != 0);
    CHECK(coli_parse_cpu_list("-1", &set, ncpu) != 0);
    char huge[32];
    snprintf(huge, sizeof huge, "%ld", ncpu);          /* one past the last online CPU */
    CHECK(coli_parse_cpu_list(huge, &set, ncpu) != 0);
    printf("  parse: taskset lists accepted, typos refused (ncpu %ld)\n", ncpu);
    return 0;
}

static int masks_equal(const cpu_set_t *a, const cpu_set_t *b) {
    return CPU_EQUAL(a, b);
}

static int test_keep_and_apply(void) {
    long ncpu = coli_affinity_ncpu();
    if (ncpu < 2) {
        printf("  affinity install: one CPU, skipped\n");
        return 0;
    }
    cpu_set_t saved;
    if (sched_getaffinity(0, sizeof saved, &saved) != 0) {
        fprintf(stderr, "sched_getaffinity: %s\n", strerror(errno));
        return 1;
    }
    cpu_set_t one; CPU_ZERO(&one); CPU_SET(0, &one);
    if (sched_setaffinity(0, sizeof one, &one) != 0) {
        fprintf(stderr, "cannot pin to CPU 0 (%s); skipping install check\n", strerror(errno));
        return 0;
    }
    /* KEEP must not widen the pin. */
    coli_affinity_before_reexec(NULL, 0);
    cpu_set_t now;
    CHECK(sched_getaffinity(0, sizeof now, &now) == 0);
    CHECK(masks_equal(&now, &one));

    /* COLI_CPUS wins, including over a pre-set OMP_PROC_BIND. */
    if (coli_affinity_apply_list("1") != 0) {
        sched_setaffinity(0, sizeof saved, &saved);
        fprintf(stderr, "COLI_CPUS=1 did not apply\n");
        return 1;
    }
    CHECK(sched_getaffinity(0, sizeof now, &now) == 0);
    CHECK(cpu_in(&now, 1) && CPU_COUNT(&now) == 1);

    /* A rejected list leaves the mask just installed. */
    CHECK(coli_affinity_apply_list("nope") != 0);
    CHECK(sched_getaffinity(0, sizeof now, &now) == 0);
    CHECK(cpu_in(&now, 1) && CPU_COUNT(&now) == 1);

    /* #471 path widens back to every online CPU. */
    coli_affinity_before_reexec(NULL, 1);
    CHECK(sched_getaffinity(0, sizeof now, &now) == 0);
    CHECK(CPU_COUNT(&now) == (int)ncpu);

    if (sched_setaffinity(0, sizeof saved, &saved) != 0)
        perror("restore affinity");
    printf("  install: inherited pin kept, COLI_CPUS applied, #471 still widens\n");
    return 0;
}
#endif

int main(void) {
    int rc = test_action();
#if defined(__linux__) && defined(CPU_SETSIZE)
    if (!rc) rc = test_parse();
    if (!rc) rc = test_keep_and_apply();
#else
    printf("  affinity install: not Linux, action only\n");
#endif
    if (rc) { fprintf(stderr, "test_reexec_affinity: FAIL\n"); return 1; }
    printf("test_reexec_affinity: ok\n");
    return 0;
}
