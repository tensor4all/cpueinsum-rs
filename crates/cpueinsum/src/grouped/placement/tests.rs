use super::*;

/// Jobs of serial times `us` microseconds; a job's child widens up to one
/// thread per 50 us of its work, and a width `w > 1` costs `2 + w / 2` us of
/// dispatch. Small numbers, so every expectation below can be checked by hand.
struct Synthetic {
    us: Vec<u64>,
}

impl Model for Synthetic {
    fn serial(&self, j: usize) -> Duration {
        Duration::from_micros(self.us[j])
    }

    fn grant(&self, j: usize, width: usize) -> usize {
        width.min((self.us[j] / 50).max(1) as usize)
    }

    fn dispatch(&self, width: usize) -> Duration {
        Duration::from_nanos(2_000 + 500 * width as u64)
    }
}

fn placed(slots: usize, us: &[u64]) -> (Placement, Synthetic) {
    let model = Synthetic { us: us.to_vec() };
    let mut placement = Placement::default();
    place(
        slots,
        slots.min(us.len()),
        us.len(),
        &model,
        &mut placement,
        &mut PlaceScratch::default(),
    );
    (placement, model)
}

fn lanes(placement: &Placement) -> Vec<(usize, Vec<usize>)> {
    (0..placement.lanes())
        .map(|lane| (placement.shares()[lane], placement.jobs(lane).to_vec()))
        .collect()
}

/// Each lane's modeled time in whole nanoseconds.
fn times(placement: &Placement, model: &Synthetic) -> Vec<u128> {
    let mut times = Vec::new();
    placement.lane_times(model, &mut times);
    times.iter().map(Duration::as_nanos).collect()
}

/// The bound of the module documentation, evaluated directly.
fn assert_bound(placement: &Placement, model: &Synthetic) {
    let mut times = Vec::new();
    placement.lane_times(model, &mut times);
    let least = *times.iter().min().unwrap();
    for lane in 0..placement.lanes() {
        let jobs = placement.jobs(lane);
        if jobs.len() < 2 {
            continue;
        }
        let slack = jobs
            .iter()
            .flat_map(|&j| placement.shares().iter().map(move |&w| model.item(j, w)))
            .max()
            .unwrap();
        assert!(
            times[lane] <= least + slack,
            "lane {lane}: {:?} > {least:?} + {slack:?}",
            times[lane]
        );
    }
}

#[test]
fn item_time_is_serial_at_one_and_divided_plus_dispatch_above() {
    let us = Duration::from_micros;
    assert_eq!(item_time(us(90), 1, us(7)), us(90));
    assert_eq!(item_time(us(90), 3, us(7)), us(37));
}

#[test]
fn a_dominant_job_gets_a_wide_lane() {
    // gm-dom in miniature: one 1000 us job and fifteen 4 us ones on 16 slots.
    let mut us = vec![1000];
    us.extend([4; 15]);
    let (placement, model) = placed(16, &us);
    // 1000 / 15 + (2 + 7.5) = 76.2 us beside 15 x 4 = 60 us; a 14-wide lane
    // would take 1000 / 14 + 9 = 80.4 us.
    assert_eq!(
        lanes(&placement),
        [(15, vec![0]), (1, (1..16).collect::<Vec<_>>())]
    );
    assert_eq!(times(&placement, &model), [76_166, 60_000]);
    assert_bound(&placement, &model);
}

#[test]
fn a_uniform_list_keeps_one_thread_per_lane() {
    // Eight 10 us jobs on 4 slots: nobody widens, so this is the historical
    // `Outer { lanes: 4 }` with two jobs per lane.
    let (placement, model) = placed(4, &[10; 8]);
    assert_eq!(
        lanes(&placement),
        [
            (1, vec![0, 4]),
            (1, vec![1, 5]),
            (1, vec![2, 6]),
            (1, vec![3, 7])
        ]
    );
    assert_eq!(times(&placement, &model), [20_000; 4]);
    assert_bound(&placement, &model);
}

#[test]
fn a_skewed_short_list_gets_uneven_widths() {
    // c u1f in miniature: shares 0.60, 0.18, 0.18, 0.02, 0.02 on 16 slots.
    let (placement, model) = placed(16, &[600, 180, 180, 20, 20]);
    // 600 / 9 + 6.5, 180 / 3 + 3.5 twice, and the two small jobs together.
    assert_eq!(
        lanes(&placement),
        [(9, vec![0]), (3, vec![1]), (3, vec![2]), (1, vec![3, 4])]
    );
    assert_eq!(times(&placement, &model), [73_166, 63_500, 63_500, 40_000]);
    assert_bound(&placement, &model);
}

#[test]
fn one_job_takes_every_slot() {
    let (placement, _) = placed(16, &[600]);
    assert_eq!(lanes(&placement), [(16, vec![0])]);
}

#[test]
fn placement_is_deterministic_and_reuses_its_tables() {
    let mut us = vec![1000, 300];
    us.extend([4; 30]);
    let model = Synthetic { us };
    let (first, _) = placed(12, &model.us);
    let mut scratch = PlaceScratch::default();
    let mut placement = Placement::default();
    for _ in 0..3 {
        place(12, 12, model.us.len(), &model, &mut placement, &mut scratch);
        assert_eq!(placement, first);
    }
}
