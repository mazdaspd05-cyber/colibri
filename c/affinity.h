/* CPU affinity across the OpenMP hot-team re-exec.
 *
 * execv preserves the calling thread's affinity mask. Two different masks
 * arrive there, and they must not be treated the same way:
 *
 *   - taskset / systemd CPUAffinity (and a full mask): the operator's pin.
 *     Widening it to every online CPU throws the pin away. Leave it.
 *   - a process that was started with OMP_PROC_BIND or OMP_PLACES already
 *     set: libgomp's constructor binds this thread to place 0 before main,
 *     and the re-exec'd image would jail its whole team on that place
 *     (#471, ~20x). Restore every online CPU so the child's constructor
 *     binds again from the full set. The user's OMP_* values are kept.
 *
 * COLI_CPUS, a taskset-style list ("0-3,8"), wins over both: it is the mask
 * installed before the re-exec, whatever was inherited. An unparseable list
 * is refused and the inherited mask is kept.
 */
#ifndef COLI_AFFINITY_H
#define COLI_AFFINITY_H

typedef enum {
    COLI_AFF_KEEP = 0,
    COLI_AFF_RESET_ONLINE = 1,
    COLI_AFF_APPLY_LIST = 2
} coli_aff_action;

/* coli_cpus is COLI_CPUS (NULL or empty means unset). user_had_omp_bind is
 * true when OMP_PROC_BIND or OMP_PLACES was already in the environment before
 * this process seeded its own defaults. */
static inline coli_aff_action coli_affinity_reexec_action(const char *coli_cpus, int user_had_omp_bind) {
    if (coli_cpus && *coli_cpus) return COLI_AFF_APPLY_LIST;
    if (user_had_omp_bind) return COLI_AFF_RESET_ONLINE;
    return COLI_AFF_KEEP;
}

#if defined(__linux__) && defined(CPU_SETSIZE)
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Parse "0-3,8,10-12" into set. Returns 0 with a non-empty set, or -1.
 * An index outside [0, ncpu) fails the whole list, so a typo cannot pin the
 * process to a truncated mask. ncpu is the online CPU count, already clamped
 * to CPU_SETSIZE. */
static int coli_parse_cpu_list(const char *s, cpu_set_t *set, long ncpu) {
    if (!s || !set || ncpu < 1) return -1;
    CPU_ZERO(set);
    int any = 0;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
        if (!*p) break;
        if (*p < '0' || *p > '9') return -1;
        char *end = NULL;
        long a = strtol(p, &end, 10);
        if (end == p) return -1;
        p = end;
        long b = a;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '-') {
            p++;
            while (*p == ' ' || *p == '\t') p++;
            if (*p < '0' || *p > '9') return -1;
            b = strtol(p, &end, 10);
            if (end == p || b < a) return -1;
            p = end;
        }
        if (a < 0 || b >= ncpu || b >= CPU_SETSIZE) return -1;
        for (long i = a; i <= b; i++) { CPU_SET((int)i, set); any = 1; }
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ',') { p++; continue; }
        if (*p) return -1;
    }
    return any ? 0 : -1;
}

static long coli_affinity_ncpu(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > CPU_SETSIZE) n = CPU_SETSIZE;
    return n;
}

/* Install cpus. On a rejected list the calling thread's mask is unchanged. */
static int coli_affinity_apply_list(const char *cpus) {
    long ncpu = coli_affinity_ncpu();
    if (ncpu < 1) return -1;
    cpu_set_t set;
    if (coli_parse_cpu_list(cpus, &set, ncpu) != 0) {
        fprintf(stderr, "[OMP] COLI_CPUS=%s is not a CPU list in 0..%ld; keeping the inherited affinity mask\n",
                cpus ? cpus : "", ncpu - 1);
        return -1;
    }
    if (sched_setaffinity(0, sizeof set, &set) != 0) {
        perror("[OMP] sched_setaffinity COLI_CPUS (keeping the inherited mask)");
        return -1;
    }
    return 0;
}

static void coli_affinity_reset_online(void) {
    cpu_set_t all; CPU_ZERO(&all);
    long ncpu = coli_affinity_ncpu();
    for (long i = 0; i < ncpu; i++) CPU_SET((int)i, &all);
    if (sched_setaffinity(0, sizeof all, &all) != 0)
        perror("[OMP] sched_setaffinity pre-reexec (continuing)");
}

static void coli_affinity_before_reexec(const char *coli_cpus, int user_had_omp_bind) {
    switch (coli_affinity_reexec_action(coli_cpus, user_had_omp_bind)) {
    case COLI_AFF_APPLY_LIST: coli_affinity_apply_list(coli_cpus); break;
    case COLI_AFF_RESET_ONLINE: coli_affinity_reset_online(); break;
    case COLI_AFF_KEEP: break;
    }
}
#endif /* linux */
#endif /* COLI_AFFINITY_H */
