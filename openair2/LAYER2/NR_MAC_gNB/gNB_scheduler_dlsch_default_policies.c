/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief       Default pluggable policy functions for the DL scheduler pipeline.
 *
 * These are the built-in implementations behind the function pointers
 * (dl_ri_pmi_select, dl_tda_select, dl_beam_select, dl_mcs_select, dl_rb_alloc, dl_lcid_alloc)
 * wired up at MAC init time.  They can be replaced at runtime by external
 * scheduler plug-ins without touching the core scheduling loop in
 * gNB_scheduler_dlsch.c.
 */

#include "common/utils/nr/nr_common.h"
#include "gNB_scheduler_dlsch_default_policies.h"
#include "genann.h"
#include <pthread.h>
#include <unistd.h>
/*MAC*/
#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_MAC_gNB/nr_mac_gNB.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_oai_api.h"

/*TAG*/
#include "NR_TAG-Id.h"

/*Softmodem params*/
#include "executables/softmodem-common.h"
#include "../../../nfapi/oai_integration/vendor_ext.h"

#define REPLAY_BUFFER_CAPACITY 10000  /* how much history the buffer holds */
#define TRAIN_BATCH_SIZE 32          /* how many random samples per training round */
#define PENDING_TABLE_SIZE 64

/* One replay entry = one (state, target) pair captured from one slot.
 * "target" is a placeholder for now (fixed 0.5) until a real reward
 * signal is defined in a later step. */
typedef struct {
  double state[2];   /* avg_throughput (normalized), cqi (normalized) */
  double target[1];  /* placeholder training target */
  long long write_time_ns; /* CLOCK_MONOTONIC timestamp at the moment this sample was written */
} replay_sample_t;

/* One pending entry = one action taken, waiting for its ACK/NACK outcome.
 * Correlated by RNTI only (see limitation note: with several HARQ
 * processes in flight per UE, this may occasionally match the wrong
 * specific transmission - acceptable simplification for now). */
typedef struct {
  int valid;
  uint16_t rnti;
  double state[2];
} pending_transition_t;

static replay_sample_t replay_buffer[REPLAY_BUFFER_CAPACITY];
static int replay_write_idx = 0;   /* next slot to write into (circular) */
static int replay_count = 0;       /* how many valid entries so far, caps at CAPACITY */
static pthread_mutex_t replay_mutex = PTHREAD_MUTEX_INITIALIZER;

static genann *shared_nn = NULL;
static pthread_t training_thread_handle;
static int training_thread_started = 0;

/* ==================== background training thread ==================== */
/* Runs independently of the real-time scheduler loop. Wakes up periodically,
 * and if enough samples are available, draws a random batch and trains on it.
 * This function never touches OAI's scheduling data structures directly -
 * it only reads from the replay buffer, which is filled by the real-time
 * thread. This keeps the two thread's responsibilities cleanly separated. */
static void *genann_training_thread(void *arg) {
  (void)arg;

  while (1) {
    usleep(10000); /* 10 ms between rounds - not time critical, can be tuned */

    replay_sample_t batch[TRAIN_BATCH_SIZE];
    int available;

    /* Critical section: copy out a random batch under the lock, then
     * release immediately. We do NOT train while holding the lock, so
     * the real-time thread is never blocked for more than a few memcpy. */
    pthread_mutex_lock(&replay_mutex);
    available = replay_count;
    if (available >= TRAIN_BATCH_SIZE) {
      for (int i = 0; i < TRAIN_BATCH_SIZE; i++) {
        int idx = rand() % available; /* random sample: breaks time correlation */
        batch[i] = replay_buffer[idx];
      }
    }
    pthread_mutex_unlock(&replay_mutex);

    if (available < TRAIN_BATCH_SIZE) {
      continue; /* buffer not filled enough yet, wait for more data */
    }

    struct timespec sample_now_ts;
    clock_gettime(CLOCK_MONOTONIC, &sample_now_ts);
    long long now_ns = sample_now_ts.tv_sec * 1000000000LL + sample_now_ts.tv_nsec;

    long long age_min_ns = -1, age_max_ns = -1, age_sum_ns = 0;
    for (int i = 0; i < TRAIN_BATCH_SIZE; i++) {
      long long age_ns = now_ns - batch[i].write_time_ns;
      if (age_min_ns < 0 || age_ns < age_min_ns) age_min_ns = age_ns;
      if (age_ns > age_max_ns) age_max_ns = age_ns;
      age_sum_ns += age_ns;
    }
    long long age_avg_ns = age_sum_ns / TRAIN_BATCH_SIZE;

    /* Actual training happens outside the lock, on our local copy of the
     * batch. This is where the "no lock on weights" trade-off applies:
     * the real-time thread may call genann_run() concurrently while we
     * write into shared_nn's weights here. */
    struct timespec bt0, bt1;
    clock_gettime(CLOCK_MONOTONIC, &bt0);
    for (int i = 0; i < TRAIN_BATCH_SIZE; i++) {
      genann_train(shared_nn, batch[i].state, batch[i].target, 0.05);
    }
    clock_gettime(CLOCK_MONOTONIC, &bt1);

    long long elapsed_ns = (bt1.tv_sec - bt0.tv_sec) * 1000000000LL + (bt1.tv_nsec - bt0.tv_nsec);
    LOG_I(NR_MAC,
          "[genann_test] BACKGROUND TRAIN: batch=%d, took=%lld ns, buffer_fill=%d/%d, "
          "sample_age_avg=%.2f ms, sample_age_min=%.2f ms, sample_age_max=%.2f ms\n",
          TRAIN_BATCH_SIZE, elapsed_ns, available, REPLAY_BUFFER_CAPACITY,
          age_avg_ns / 1e6, age_min_ns / 1e6, age_max_ns / 1e6);
  }

  return NULL; /* never reached in practice, thread runs forever */
}

// Default RI/PMI selector: reads rank and PMI from CSI feedback for new-tx,
// or from HARQ process state for retx.
void nr_dl_ri_pmi_select_default(const gNB_MAC_INST *mac, nr_dl_candidate_t *candidates, int n_candidates)
{
  FOR_EACH_CANDIDATE(cand, candidates, n_candidates)
  {
    NR_UE_sched_ctrl_t *sched_ctrl = &cand->UE->UE_sched_ctrl;
    NR_UE_DL_BWP_t *dl_bwp = &cand->UE->current_DL_BWP;
    if (cand->is_retx) {
      cand->sched_pdsch.nrOfLayers = sched_ctrl->harq_processes[cand->retx_harq_pid].sched_pdsch.nrOfLayers;
      cand->sched_pdsch.pm_index =
          get_pm_index(mac, cand->UE, dl_bwp->dci_format, cand->sched_pdsch.nrOfLayers, mac->radio_config.pdsch_AntennaPorts.XP);
    } else {
      cand->sched_pdsch.nrOfLayers = cand->csi_ri + 1;
      cand->sched_pdsch.pm_index = cand->csi_pm_index;
    }
  }
}

// Default TDA selector: picks the slot-wide TDA index from get_dl_tda(),
// then resolves tda_info per candidate using each UE's own BWP / search
// space / coreset. Marks invalids with skipped=true.
int nr_dl_tda_select_default(const gNB_MAC_INST *mac, nr_dl_candidate_t *candidates, int n_candidates, frame_t frame, slot_t slot)
{
  int tda = get_dl_tda(mac, slot);
  AssertFatal(tda >= 0, "Unable to find PDSCH time domain allocation in list\n");
  const NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;

  int n_valid = 0;
  FOR_EACH_CANDIDATE(cand, candidates, n_candidates)
  {
    if (cand->skipped)
      continue;
    NR_UE_info_t *UE = cand->UE;
    NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
    NR_UE_DL_BWP_t *dl_bwp = &UE->current_DL_BWP;
    int coresetid = sched_ctrl->coreset->controlResourceSetId;
    NR_tda_info_t tda_info = get_dl_tda_info(dl_bwp,
                                             sched_ctrl->search_space->searchSpaceType->present,
                                             tda,
                                             scc->dmrs_TypeA_Position,
                                             1,
                                             TYPE_C_RNTI_,
                                             coresetid,
                                             false);
    if (!tda_info.valid_tda) {
      cand->skipped = true;
      continue;
    }

    /* For retransmissions with a changed TDA, refit rbSize to preserve TBS */
    if (cand->is_retx) {
      const NR_sched_pdsch_t *orig = &sched_ctrl->harq_processes[cand->retx_harq_pid].sched_pdsch;
      bool tda_changed =
          tda_info.startSymbolIndex != orig->tda_info.startSymbolIndex || tda_info.nrOfSymbols != orig->tda_info.nrOfSymbols;
      if (tda_changed) {
        uint16_t new_rbSize = check_dl_retx_feasibility(cand, tda, &tda_info, scc, dl_bwp->BWPSize);
        if (!new_rbSize) {
          cand->skipped = true;
          continue;
        }
        cand->retx_rbSize = new_rbSize;
      }
    }

    cand->sched_pdsch.time_domain_allocation = tda;
    cand->sched_pdsch.tda_info = tda_info;
    cand->alloc_slbitmap = SL_to_bitmap(tda_info.startSymbolIndex, tda_info.nrOfSymbols);
    n_valid++;
  }
  return n_valid;
}

static int compare_dl_pf_ptrs(const void *a, const void *b)
{
  const nr_dl_candidate_t *ca = *(const nr_dl_candidate_t *const *)a;
  const nr_dl_candidate_t *cb = *(const nr_dl_candidate_t *const *)b;
  /* retx first (INFINITY weight), then highest PF weight */
  float wa = ca->is_retx ? INFINITY : dl_pf_weight(ca->current_mcs, ca->mcs_table, ca->sched_pdsch.nrOfLayers, ca->avg_throughput);
  float wb = cb->is_retx ? INFINITY : dl_pf_weight(cb->current_mcs, cb->mcs_table, cb->sched_pdsch.nrOfLayers, cb->avg_throughput);
  return (wa < wb) - (wa > wb);
}

int nr_dl_beam_select_default(NR_beam_info_t *beam_info,
                              const int16_t *beam_index_list,
                              nr_dl_candidate_t *candidates,
                              int n_candidates,
                              frame_t frame,
                              slot_t slot,
                              int slots_per_frame)
{
  /* Build pointer array sorted by PF priority so retx and high-priority UEs claim beams first. */
  nr_dl_candidate_t *order[MAX_MOBILES_PER_GNB];
  int n_active = 0;
  FOR_EACH_CANDIDATE(cand, candidates, n_candidates)
  if (!cand->skipped)
    order[n_active++] = cand;
  qsort(order, n_active, sizeof(*order), compare_dl_pf_ptrs);

  int n_valid = 0;
  for (int i = 0; i < n_active; i++) {
    nr_dl_candidate_t *cand = order[i];
    NR_beam_alloc_t beam = beam_allocation_procedure(beam_info, frame, slot, cand->alloc_beam_dir, slots_per_frame);
    if (beam.idx < 0) {
      cand->skipped = true;
      continue;
    }

    cand->alloc_beam_idx = beam.idx;
    cand->alloc_new_beam = beam.new_beam;
    n_valid++;
  }
  return n_valid;
}

void nr_dl_mcs_select_default(const gNB_MAC_INST *mac, nr_dl_candidate_t *candidates, int n_candidates)
{
  const NR_bler_options_t *bo = &mac->dl_bler;
  FOR_EACH_CANDIDATE(cand, candidates, n_candidates)
  {
    int mcs;
    if (cand->is_retx) {
      mcs = cand->current_mcs; /* retx MCS is fixed by the HARQ round */
    } else if (bo->harq_round_max == 1) {
      mcs = max(bo->min_mcs, min(bo->max_mcs, cand->max_mcs));
    } else if (!cand->bler_updated) {
      mcs = cand->current_mcs;
    } else {
      mcs = nr_adapt_mcs_from_bler(cand->current_mcs,
                                   bo->min_mcs,
                                   cand->max_mcs,
                                   cand->bler,
                                   bo->lower,
                                   bo->upper,
                                   cand->last_num_sched);
    }
    cand->sched_pdsch.mcs = mcs;
    /* Persist for all candidates — BLER-based MCS ramps even for UEs the
     * policy rejects this slot (failed CCE, no free RBs, etc.). */
    if (!cand->is_retx)
      cand->UE->UE_sched_ctrl.dl_bler_stats.mcs = mcs;
  }
}

static int compare_dl_pf_rb_ptrs(const void *a, const void *b)
{
  const nr_dl_candidate_t *ca = *(const nr_dl_candidate_t *const *)a;
  const nr_dl_candidate_t *cb = *(const nr_dl_candidate_t *const *)b;
  /* retx first, then highest PF weight (uses sched_pdsch.mcs, which is set by mcs_select) */
  float wa =
      ca->is_retx ? INFINITY : dl_pf_weight(ca->sched_pdsch.mcs, ca->mcs_table, ca->sched_pdsch.nrOfLayers, ca->avg_throughput);
  float wb =
      cb->is_retx ? INFINITY : dl_pf_weight(cb->sched_pdsch.mcs, cb->mcs_table, cb->sched_pdsch.nrOfLayers, cb->avg_throughput);
  return (wa < wb) - (wa > wb);
}

int nr_dl_proportional_fair(const nr_dl_sched_params_t *params, nr_dl_candidate_t *candidates, int n_candidates)
{
  const int min_rbSize = 5;
  int n_scheduled = 0;

  /* Build pointer array sorted by PF priority (retx first, then highest weight) */
  nr_dl_candidate_t *order[MAX_MOBILES_PER_GNB];
  int n_active = 0;
  FOR_EACH_CANDIDATE(cand, candidates, n_candidates)
  if (!cand->skipped) {
    order[n_active++] = cand;

      /* Lazy one-time setup: shared network + background thread, started once */
  if (!shared_nn) {
    shared_nn = genann_init(2, 1, 8, 1);
    srand((unsigned)time(NULL)); /* seed rand() once, for batch sampling */
    LOG_I(NR_MAC, "[genann_test] shared network initialized (%d weights)\n", shared_nn->total_weights);
  }
  if (!training_thread_started) {
    pthread_create(&training_thread_handle, NULL, genann_training_thread, NULL);
    training_thread_started = 1;
    LOG_I(NR_MAC, "[genann_test] background training thread started\n");
  }

  double input[2];
  input[0] = cand->avg_throughput / 1e6;
  input[1] = cand->cqi / 15.0;

  /* Inference stays in the real-time path: this is what must never exceed
   * the slot deadline, as confirmed with the tutor. */
  static long long inference_total_ns = 0;
  static int inference_call_count = 0;

  struct timespec it0, it1;
  clock_gettime(CLOCK_MONOTONIC, &it0);
  double const *out = genann_run(shared_nn, input);
  clock_gettime(CLOCK_MONOTONIC, &it1);
  (void)out; /* not used yet for any scheduling decision */

  long long inf_elapsed_ns = (it1.tv_sec - it0.tv_sec) * 1000000000LL + (it1.tv_nsec - it0.tv_nsec);
  inference_total_ns += inf_elapsed_ns;
  inference_call_count++;

  if (inference_call_count % 128 == 0) {
    LOG_I(NR_MAC,
          "[genann_test] INFERENCE call #%d, avg_inference=%lld ns\n",
          inference_call_count, inference_total_ns / inference_call_count);
  }

  struct timespec now_ts;
  clock_gettime(CLOCK_MONOTONIC, &now_ts);

  replay_sample_t sample;
  sample.state[0] = input[0];
  sample.state[1] = input[1];
  sample.target[0] = 1.0 - cand->bler; /* real reward: 1.0 = perfect, 0.0 = total failure */
  sample.write_time_ns = now_ts.tv_sec * 1000000000LL + now_ts.tv_nsec;

  pthread_mutex_lock(&replay_mutex);
  replay_buffer[replay_write_idx] = sample;
  replay_write_idx = (replay_write_idx + 1) % REPLAY_BUFFER_CAPACITY;
  if (replay_count < REPLAY_BUFFER_CAPACITY) replay_count++;
  pthread_mutex_unlock(&replay_mutex);

  if (inference_call_count % 128 == 0) {
    LOG_I(NR_MAC, "[genann_test] REWARD (bler-based) RNTI %04x, bler=%.5f, reward=%.5f\n",
          cand->rnti, cand->bler, sample.target[0]);
  }
  }
  qsort(order, n_active, sizeof(*order), compare_dl_pf_rb_ptrs);

  /* Phase 1: HARQ retransmissions (highest priority, exact RBs) */
  for (int j = 0; j < n_active; j++) {
    nr_dl_candidate_t *cand = order[j];
    if (!cand->is_retx)
      continue;

    int needed_rbs = cand->retx_rbSize;
    uint16_t *vrb_map = params->vrb_map[cand->alloc_beam_idx];
    int rbStart, rbSize;
    if (!get_rb_alloc(needed_rbs,
                      cand->bwp_size,
                      cand->bwp_start,
                      cand->bwp_size,
                      vrb_map,
                      cand->alloc_slbitmap,
                      &rbStart,
                      &rbSize))
      continue;

    COMMIT_ALLOC(params, cand, rbStart, needed_rbs, cand->sched_pdsch.mcs, n_scheduled);
  }

  /* Phase 2: No-data UEs (TA command or beam switch MAC CE, no RLC data) */
  for (int j = 0; j < n_active; j++) {
    nr_dl_candidate_t *cand = order[j];
    if (cand->is_retx || cand->pending_bytes > 0)
      continue;

    uint16_t *vrb_map = params->vrb_map[cand->alloc_beam_idx];
    int rbStart, rbSize;
    if (!get_rb_alloc(min_rbSize,
                      cand->bwp_size,
                      cand->bwp_start,
                      cand->bwp_size,
                      vrb_map,
                      cand->alloc_slbitmap,
                      &rbStart,
                      &rbSize))
      continue;

    COMMIT_ALLOC(params, cand, rbStart, min_rbSize, cand->sched_pdsch.mcs, n_scheduled);
  }

  /* Phase 3: New data UEs — PF priority order, largest free block */
  for (int j = 0; j < n_active; j++) {
    nr_dl_candidate_t *cand = order[j];
    if (cand->is_retx || cand->pending_bytes == 0)
      continue;

    int rbStart;
    uint16_t *vrb_map = params->vrb_map[cand->alloc_beam_idx];
    int max_rbSize = find_largest_free_block(vrb_map, cand->alloc_slbitmap, cand->bwp_start, cand->bwp_size, &rbStart);
    if (max_rbSize < min_rbSize)
      continue;

    int mcs = cand->sched_pdsch.mcs;
    uint8_t Qm = nr_get_Qm_dl(mcs, cand->mcs_table);
    uint16_t R = nr_get_code_rate_dl(mcs, cand->mcs_table);
    NR_pdsch_dmrs_t dmrs = get_dl_dmrs_params(params->mac->common_channels->ServingCellConfigCommon,
                                              &cand->UE->current_DL_BWP,
                                              &cand->sched_pdsch.tda_info,
                                              cand->sched_pdsch.nrOfLayers);
    const int oh = 3 * 4 + (cand->UE->UE_sched_ctrl.ta_apply ? 2 : 0);
    uint32_t tbs;
    uint16_t rbSize;
    nr_find_nb_rb(Qm,
                  R,
                  1,
                  cand->sched_pdsch.nrOfLayers,
                  cand->sched_pdsch.tda_info.nrOfSymbols,
                  dmrs.N_PRB_DMRS * dmrs.N_DMRS_SLOT,
                  cand->pending_bytes + oh,
                  min_rbSize,
                  max_rbSize,
                  &tbs,
                  &rbSize);

    COMMIT_ALLOC(params, cand, rbStart, rbSize, mcs, n_scheduled);
  }

  return n_scheduled;
}

void nr_dl_lcid_alloc_default(const gNB_MAC_INST *mac,
                              const nr_dl_candidate_t *candidate,
                              int tbs_available,
                              int lcid_alloc[NR_MAX_NUM_LCID])
{
  (void)mac;
  (void)tbs_available;
  memset(lcid_alloc, 0, NR_MAX_NUM_LCID * sizeof(int));
  for (int lcid = 0; lcid < NR_MAX_NUM_LCID; lcid++)
    lcid_alloc[lcid] = candidate->pending_bytes_per_lcid[lcid];
}
