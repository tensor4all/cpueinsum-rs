//! Work-weighted lane placement of the per-job route under `Auto`
//! (cpueinsum-rs#80).
//!
//! # Provenance
//!
//! Ported (close translation, adapted) from symtn-rs,
//! <https://github.com/Ryo-wtnb11/symtn-rs>, commit `09a3ebb`
//! (`09a3ebbd`, "Let workers first-touch every fresh GEMM output (#2582)"),
//! MIT OR Apache-2.0, Copyright (c) 2026 Ryo-wtnb11; the upstream MIT text is
//! `third_party/symtn-rs/LICENSE-MIT` in this crate:
//!
//! - `symtn-dense/src/lanes/placement.rs`: `item_time`, `place`,
//!   `equal_shares`, `Moldable::place`, `Reservation`, `layout`, `compact`;
//! - `symtn-dense/src/lanes.rs`: the `Placement` tables.
//!
//! What changed in the port: one stage per job (symtn prices up to three);
//! the grant is tcontract's [`Plan::execution_width`](tcontract::Plan::execution_width)
//! memoised per child plan and width (symtn composes it from the plan's batch
//! decisions in `block_kernel/gemm.rs::grant`); the baseline is the
//! equal-share placement at the lane cap scored like every other count (symtn
//! fills it by serial time); no process-global cache, no grant-key
//! representatives and no lower-bound pruning (the caller caches the result per
//! grouped plan instead); and every candidate must pass [`within_bound`].
//!
//! # Model
//!
//! Job `j` has a serial time `t_j`. On a lane of `w` slots its child takes
//! `g = grant(j, w)` participants and its modeled item time is `t_j` when
//! `g = 1`, otherwise `t_j / g` plus the token's dispatch cost at `g`
//! ([`item_time`]). A lane's modeled time is the sum of its jobs' item times
//! at its share; a placement's score is its largest lane time.
//!
//! # Bound
//!
//! Every placement returned satisfies [`within_bound`]: a lane holding two or
//! more jobs finishes, in the model, within one of its own jobs' item time of
//! the least-loaded lane,
//!
//! ```text
//! tau_k <= min_l tau_l + max_{j on k} max_l item(j, share_l)
//! ```
//!
//! A lane holding one job is that job's item time at its share, which no
//! placement can split. The equal-share placement at the lane cap always
//! satisfies it (list scheduling: lane `k`'s last job went to the lane of least
//! `load + item`, so `tau_k <= tau_l + item(j, share_l)` for every lane `l`),
//! so a candidate that does not is discarded and the search never comes back
//! empty.

use core::cmp::Reverse;
use core::time::Duration;
use std::collections::BinaryHeap;

/// A job's modeled time on a lane where its child takes `granted`
/// participants: `serial` at one, else `serial / granted` plus `dispatch`, the
/// token's dispatch cost at `granted`. symtn's `item_time`.
pub(super) fn item_time(serial: Duration, granted: usize, dispatch: Duration) -> Duration {
    if granted <= 1 {
        return serial;
    }
    (serial / u32::try_from(granted).unwrap_or(u32::MAX)).saturating_add(dispatch)
}

/// What the placement asks of the jobs: each job's serial time and its child's
/// grant on a lane of `width` slots (`1..=slots`, never growing as the lane
/// narrows), and the dispatch cost at a granted width.
pub(super) trait Model {
    /// Job `j`'s serial time.
    fn serial(&self, j: usize) -> Duration;
    /// The participants job `j`'s child takes on a lane of `width` slots.
    fn grant(&self, j: usize, width: usize) -> usize;
    /// The token's dispatch cost at `width > 1` participants.
    fn dispatch(&self, width: usize) -> Duration;

    /// Job `j`'s modeled time on a lane of `width` slots.
    fn item(&self, j: usize, width: usize) -> Duration {
        let granted = self.grant(j, width);
        item_time(self.serial(j), granted, self.dispatch(granted))
    }
}

/// A placed batch: lanes of nonzero shares summing to at most the slot count,
/// every job in exactly one lane, each lane's jobs ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Placement {
    shares: Vec<usize>,
    lane_starts: Vec<usize>,
    lane_jobs: Vec<usize>,
    lane_of: Vec<usize>,
    /// Each job's index in `lane_jobs`.
    position: Vec<usize>,
}

impl Placement {
    /// Job `j`'s index in [`Placement::lane_jobs`].
    pub(super) fn position(&self, j: usize) -> usize {
        self.position[j]
    }

    /// The number of lanes (zero for no job).
    pub(super) fn lanes(&self) -> usize {
        self.shares.len()
    }

    /// Every lane's share, in lane order.
    pub(super) fn shares(&self) -> &[usize] {
        &self.shares
    }

    /// Lane `lane`'s jobs, ascending.
    pub(super) fn jobs(&self, lane: usize) -> &[usize] {
        &self.lane_jobs[self.lane_starts[lane]..self.lane_starts[lane + 1]]
    }

    /// Each lane's modeled time under `model`, into `times`.
    pub(super) fn lane_times(&self, model: &impl Model, times: &mut Vec<Duration>) {
        times.clear();
        times.extend((0..self.lanes()).map(|lane| {
            self.jobs(lane).iter().fold(Duration::ZERO, |sum, &j| {
                sum.saturating_add(model.item(j, self.shares[lane]))
            })
        }));
    }

    /// One lane of `share` slots holding every job.
    fn sequential(&mut self, share: usize, jobs: usize) {
        self.shares.clear();
        self.shares.push(share.max(1));
        self.lane_starts.clear();
        self.lane_starts.extend([0, jobs]);
        self.lane_jobs.clear();
        self.lane_jobs.extend(0..jobs);
        self.lane_of.clear();
        self.lane_of.resize(jobs, 0);
        self.position.clear();
        self.position.extend(0..jobs);
    }
}

/// Lanes by `(load, jobs, lane)`, as a min-heap's storage.
type Lanes = Vec<Reverse<(Duration, usize, usize)>>;

/// The working tables of [`place`], kept by the caller so a warm search
/// reuses every capacity.
#[derive(Debug, Default)]
pub(super) struct PlaceScratch {
    order: Vec<usize>,
    wide_heap: Lanes,
    narrow_heap: Lanes,
    count: Vec<usize>,
    times: Vec<Duration>,
    candidate: Placement,
    moldable: Moldable,
}

/// Places the jobs of `model` (`jobs` of them) on at most `cap` job-holding
/// lanes of at most `slots` slots in all, as the module documentation
/// describes, into `placement`. `slots` and `cap` are at least one.
pub(super) fn place(
    slots: usize,
    cap: usize,
    jobs: usize,
    model: &impl Model,
    placement: &mut Placement,
    scratch: &mut PlaceScratch,
) {
    let slots = slots.max(1);
    let lanes = cap.min(slots).min(jobs).max(1);
    if jobs == 0 {
        placement.sequential(slots, 0);
        placement.shares.clear();
        placement.lane_starts.clear();
        placement.lane_starts.push(0);
        return;
    }
    if lanes == 1 {
        placement.sequential(slots, jobs);
        return;
    }
    let PlaceScratch {
        order,
        wide_heap,
        narrow_heap,
        count,
        times,
        candidate,
        moldable,
    } = scratch;
    // Longest serial time first, ties to the lowest index.
    order.clear();
    order.extend(0..jobs);
    order.sort_unstable_by_key(|&j| (Reverse(model.serial(j)), j));

    // The baseline: equal shares at the lane cap. It is admissible by the
    // module's list-scheduling argument, so the search never comes back empty.
    placement.lane_of.clear();
    placement.lane_of.resize(jobs, 0);
    let mut best = equal_shares(
        slots,
        lanes,
        order,
        model,
        (wide_heap, narrow_heap),
        Some(placement),
    );
    compact(placement, count);
    debug_assert!(within_bound(placement, model, times));
    // Ties keep the larger lane count.
    candidate.lane_of.clear();
    candidate.lane_of.resize(jobs, 0);
    for lanes in (1..lanes).rev() {
        let score = equal_shares(slots, lanes, order, model, (wide_heap, narrow_heap), None);
        if score < best {
            equal_shares(
                slots,
                lanes,
                order,
                model,
                (wide_heap, narrow_heap),
                Some(candidate),
            );
            compact(candidate, count);
            if within_bound(candidate, model, times) {
                best = score;
                core::mem::swap(placement, candidate);
                candidate.lane_of.clear();
                candidate.lane_of.resize(jobs, 0);
            }
        }
    }
    // Ties keep the equal shares.
    if let Some(score) = moldable.place((slots, lanes), order, model) {
        if score < best {
            core::mem::swap(&mut candidate.shares, &mut moldable.shares);
            core::mem::swap(&mut candidate.lane_of, &mut moldable.lane_of);
            compact(candidate, count);
            if within_bound(candidate, model, times) {
                core::mem::swap(placement, candidate);
            }
        }
    }
    debug_assert!(
        placement.shares.iter().all(|&share| share > 0)
            && placement.shares.iter().sum::<usize>() <= slots
    );
}

/// Whether `placement` satisfies the module's bound under `model`: every lane
/// holding two or more jobs has a modeled time at most the least lane time
/// plus the largest item time, at any lane's share, of a job on it. `times`
/// is scratch.
pub(super) fn within_bound(
    placement: &Placement,
    model: &impl Model,
    times: &mut Vec<Duration>,
) -> bool {
    placement.lane_times(model, times);
    let Some(&least) = times.iter().min() else {
        return true;
    };
    (0..placement.lanes()).all(|lane| {
        let jobs = placement.jobs(lane);
        if jobs.len() < 2 {
            return true;
        }
        let slack = jobs
            .iter()
            .flat_map(|&j| placement.shares.iter().map(move |&w| model.item(j, w)))
            .max()
            .unwrap_or(Duration::ZERO);
        times[lane] <= least.saturating_add(slack)
    })
}

/// The modeled makespan of the equal-share placement over `count` lanes of
/// `slots`: lane `k` reserves `slots / count + [k < slots % count]`, the jobs
/// in `order` seed one lane each (the largest on the widest), and every later
/// job goes to the lane minimising `(load + item, jobs, lane)`. With `record`,
/// the placement is written into its shares and each job's lane.
///
/// Lanes of one width share each job's item time, so each width class keeps
/// one heap ordered by `(load, jobs, lane)` and a job compares the two tops.
fn equal_shares(
    slots: usize,
    count: usize,
    order: &[usize],
    model: &impl Model,
    (wide_heap, narrow_heap): (&mut Lanes, &mut Lanes),
    mut record: Option<&mut Placement>,
) -> Duration {
    let (narrow, wide_lanes) = (slots / count, slots % count);
    let width = |k: usize| narrow + usize::from(k < wide_lanes);
    let mut wide = BinaryHeap::from(core::mem::take(wide_heap));
    let mut thin = BinaryHeap::from(core::mem::take(narrow_heap));
    wide.clear();
    thin.clear();
    if let Some(placement) = record.as_deref_mut() {
        placement.shares.clear();
        placement.shares.extend((0..count).map(width));
    }
    for (k, &j) in order.iter().take(count).enumerate() {
        let entry = Reverse((model.item(j, width(k)), 1, k));
        if k < wide_lanes {
            wide.push(entry);
        } else {
            thin.push(entry);
        }
        if let Some(placement) = record.as_deref_mut() {
            placement.lane_of[j] = k;
        }
    }
    for &j in order.iter().skip(count) {
        let wide_key = wide.peek().map(|&Reverse((load, jobs, k))| {
            (load.saturating_add(model.item(j, narrow + 1)), jobs, k)
        });
        let thin_key = thin
            .peek()
            .map(|&Reverse((load, jobs, k))| (load.saturating_add(model.item(j, narrow)), jobs, k));
        let (heap, key) = match (wide_key, thin_key) {
            (Some(a), Some(b)) if a <= b => (&mut wide, a),
            (Some(a), None) => (&mut wide, a),
            (_, Some(b)) => (&mut thin, b),
            (None, None) => unreachable!("count >= 1 seeds a lane"),
        };
        heap.pop();
        let (load, jobs, k) = key;
        heap.push(Reverse((load, jobs + 1, k)));
        if let Some(placement) = record.as_deref_mut() {
            placement.lane_of[j] = k;
        }
    }
    let score = wide
        .iter()
        .chain(thin.iter())
        .map(|&Reverse((load, _, _))| load)
        .max()
        .unwrap_or(Duration::ZERO);
    *wide_heap = wide.into_vec();
    *narrow_heap = thin.into_vec();
    score
}

/// The large-job reservations: their target tables and each layout's lanes
/// (each job's lane and each lane's share), kept for reuse.
#[derive(Debug, Default)]
struct Moldable {
    /// Row `i` holds the item time of the `i`-th largest job at every width
    /// `1..=S`, for the jobs above the mean load only.
    table: Vec<Duration>,
    targets: Vec<Duration>,
    /// `tail[r]`: the serial time of the jobs from the `r`-th largest on, in
    /// ns.
    tail: Vec<u128>,
    /// Each reserved lane's load and job count.
    wide: Vec<(Duration, usize)>,
    /// The width-one lanes, by `(load, jobs, lane)`.
    thin: Lanes,
    shares: Vec<usize>,
    lane_of: Vec<usize>,
    other_shares: Vec<usize>,
    other_lane_of: Vec<usize>,
}

/// `n` nanoseconds as a duration, saturating.
fn from_nanos(n: u128) -> Duration {
    Duration::new(
        u64::try_from(n / 1_000_000_000).unwrap_or(u64::MAX),
        (n % 1_000_000_000) as u32,
    )
}

/// What both reservation layouts read.
struct Reservation<'a> {
    slots: usize,
    cap: usize,
    order: &'a [usize],
    serial: &'a dyn Fn(usize) -> Duration,
    table: &'a [Duration],
    targets: &'a [Duration],
    tail: &'a [u128],
    large: usize,
}

impl Reservation<'_> {
    fn row(&self, i: usize) -> &[Duration] {
        &self.table[i * self.slots..(i + 1) * self.slots]
    }

    /// The least width at which the `i`-th largest job meets `target`.
    fn need(&self, i: usize, target: Duration) -> usize {
        1 + self
            .row(i)
            .iter()
            .position(|&t| t <= target)
            .unwrap_or(self.slots - 1)
    }

    /// The large jobs above `target`, a prefix of the order.
    fn reserved(&self, target: Duration) -> usize {
        self.order[..self.large]
            .iter()
            .take_while(|&&j| (self.serial)(j) > target)
            .count()
    }

    /// Whether `target` passes: the reserved jobs' least widths fit in S, and,
    /// for the exclusive layout, the other jobs' serial time fits on the
    /// one-slot lanes left, `tail <= (S - used) * target`. A passing target
    /// keeps passing as it rises (symtn's argument), so the least one is found
    /// by bisection.
    fn passes(&self, target: Duration, exclusive: bool) -> bool {
        let reserved = self.reserved(target);
        let used = (0..reserved)
            .map(|i| self.need(i, target))
            .fold(0usize, usize::saturating_add);
        if used > self.slots {
            return false;
        }
        if !exclusive {
            return true;
        }
        let free = (self.slots - used) as u128;
        (reserved == self.order.len() || free >= 1)
            && self.tail[reserved] <= free.saturating_mul(target.as_nanos())
    }
}

impl Moldable {
    /// Places the jobs, `order` largest first, on lanes of uneven widths over
    /// `slots` with two reservations (symtn's `Moldable::place`), and returns
    /// the better one's modeled makespan (ties keep the shared layout), its
    /// lanes in `self.shares` and `self.lane_of`, including the empty one-slot
    /// lanes, which the caller's compaction drops; or `None` when both put
    /// jobs on more than `cap` lanes.
    fn place(
        &mut self,
        (slots, cap): (usize, usize),
        order: &[usize],
        model: &impl Model,
    ) -> Option<Duration> {
        let Self {
            table,
            targets,
            tail,
            wide,
            thin,
            shares,
            lane_of,
            other_shares,
            other_lane_of,
        } = self;
        tail.clear();
        tail.resize(order.len() + 1, 0);
        for (i, &j) in order.iter().enumerate().rev() {
            tail[i] = tail[i + 1].saturating_add(model.serial(j).as_nanos());
        }
        let mean = tail[0].div_ceil(slots as u128);
        let heavy = order
            .iter()
            .take_while(|&&j| model.serial(j).as_nanos() > mean)
            .count();
        table.clear();
        for &j in &order[..heavy] {
            for width in 1..=slots {
                table.push(model.item(j, width));
            }
        }
        let rows = &*table;
        let row = |i: usize| &rows[i * slots..(i + 1) * slots];
        let bound = (0..heavy)
            .filter_map(|i| row(i).iter().min())
            .fold(from_nanos(mean), |bound, &least| bound.max(least));
        let large = order[..heavy]
            .iter()
            .take_while(|&&j| model.serial(j) > bound)
            .count();
        targets.clear();
        targets.push(bound);
        for i in 0..large {
            targets.extend(row(i).iter().filter(|&&t| t >= bound));
        }
        targets.sort_unstable();
        targets.dedup();
        let serial = |j: usize| model.serial(j);
        let reservation = Reservation {
            slots,
            cap,
            order,
            serial: &serial,
            table: rows,
            targets,
            tail,
            large,
        };
        let shared = layout(&reservation, false, (wide, thin), (shares, lane_of), model);
        let exclusive = layout(
            &reservation,
            true,
            (wide, thin),
            (other_shares, other_lane_of),
            model,
        );
        match (shared, exclusive) {
            (shared, Some(x)) if shared.is_none_or(|shared| x < shared) => {
                core::mem::swap(shares, other_shares);
                core::mem::swap(lane_of, other_lane_of);
                Some(x)
            }
            (shared, _) => shared,
        }
    }
}

/// One reservation layout (symtn's `layout`, steps 2 to 4 of
/// `Moldable::place`) into `shares` and `lane_of`; its score, or `None` when
/// more than the cap of its lanes hold jobs.
///
/// 2. The least passing target `T` among the candidates is found by bisection.
/// 3. Each job above `T` gets a lane of its least width meeting `T`, the
///    remaining slots become one-slot lanes, and every other job, largest
///    first, goes to the lane minimising `(load + item, jobs, lane)`: any lane
///    in the shared layout, a one-slot lane in the exclusive one.
/// 4. Empty one-slot lanes are dropped, and their slots widen the reserved
///    lanes, largest job first, while that job's grant still grows.
fn layout(
    r: &Reservation<'_>,
    exclusive: bool,
    (wide, thin): (&mut Vec<(Duration, usize)>, &mut Lanes),
    (shares, lane_of): (&mut Vec<usize>, &mut Vec<usize>),
    model: &impl Model,
) -> Option<Duration> {
    let (slots, order) = (r.slots, r.order);
    let target = r.targets[r.targets.partition_point(|&t| !r.passes(t, exclusive))];
    let reserved = r.reserved(target);
    let used: usize = (0..reserved).map(|i| r.need(i, target)).sum();

    let count = reserved + (slots - used);
    shares.clear();
    lane_of.clear();
    lane_of.resize(order.len(), 0);
    wide.clear();
    for (i, &j) in order.iter().take(reserved).enumerate() {
        let width = r.need(i, target);
        shares.push(width);
        lane_of[j] = i;
        wide.push((r.row(i)[width - 1], 1));
    }
    let mut heap = BinaryHeap::from(core::mem::take(thin));
    heap.clear();
    for k in reserved..count {
        shares.push(1);
        heap.push(Reverse((Duration::ZERO, 0, k)));
    }
    for &j in &order[reserved..] {
        let mut best = heap
            .peek()
            .map(|&Reverse((load, jobs, k))| (load.saturating_add(model.item(j, 1)), jobs, k));
        if !exclusive {
            for (k, &(load, jobs)) in wide.iter().enumerate() {
                let key = (load.saturating_add(model.item(j, shares[k])), jobs, k);
                best = Some(best.map_or(key, |best| best.min(key)));
            }
        }
        let Some((load, jobs, k)) = best else {
            unreachable!("a target passes with a lane for every job")
        };
        if k < reserved {
            wide[k] = (load, jobs + 1);
        } else {
            heap.pop();
            heap.push(Reverse((load, jobs + 1, k)));
        }
        lane_of[j] = k;
    }
    let mut idle = heap
        .iter()
        .filter(|Reverse((_, jobs, _))| *jobs == 0)
        .count();
    let score = heap
        .iter()
        .map(|&Reverse((load, _, _))| load)
        .max()
        .unwrap_or(Duration::ZERO);
    *thin = heap.into_vec();
    if count - idle > r.cap {
        return None;
    }
    for (k, &j) in order.iter().take(reserved).enumerate() {
        while idle > 0 && model.grant(j, shares[k] + 1) > model.grant(j, shares[k]) {
            shares[k] += 1;
            idle -= 1;
        }
        wide[k].0 = Duration::ZERO;
    }
    // Each reserved lane's load at its final width, in one job pass.
    for (j, &k) in lane_of.iter().enumerate() {
        if k < reserved {
            wide[k].0 = wide[k].0.saturating_add(model.item(j, shares[k]));
        }
    }
    Some(wide.iter().fold(score, |score, &(load, _)| score.max(load)))
}

/// Drops the empty lanes of `placement.shares`, renumbering the others in
/// order, and builds each lane's ascending job list from `placement.lane_of`,
/// with `count` as scratch. symtn's `compact`.
fn compact(placement: &mut Placement, count: &mut Vec<usize>) {
    let jobs = placement.lane_of.len();
    count.clear();
    count.resize(placement.shares.len(), 0);
    for &l in &placement.lane_of {
        count[l] += 1;
    }
    // `count` becomes each kept lane's new index.
    let mut kept = 0;
    placement.lane_starts.clear();
    for (l, size) in count.iter_mut().enumerate() {
        if *size > 0 {
            placement.shares[kept] = placement.shares[l];
            placement.lane_starts.push(*size);
            *size = kept;
            kept += 1;
        }
    }
    placement.shares.truncate(kept);
    // Lane sizes to starts, then a source-order fill.
    let mut start = 0;
    for size in placement.lane_starts.iter_mut() {
        (*size, start) = (start, start + *size);
    }
    placement.lane_starts.push(start);
    placement.lane_jobs.clear();
    placement.lane_jobs.resize(jobs, 0);
    placement.position.clear();
    placement.position.resize(jobs, 0);
    for j in 0..jobs {
        let l = count[placement.lane_of[j]];
        placement.lane_of[j] = l;
        placement.lane_jobs[placement.lane_starts[l]] = j;
        placement.position[j] = placement.lane_starts[l];
        placement.lane_starts[l] += 1;
    }
    // The fill advanced each start to the next lane's.
    for l in (1..kept).rev() {
        placement.lane_starts[l] = placement.lane_starts[l - 1];
    }
    placement.lane_starts[0] = 0;
}

#[cfg(test)]
mod tests;
