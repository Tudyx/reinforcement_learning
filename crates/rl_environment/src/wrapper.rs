use crate::Gym3Environment;

/// Est-ce qu'un extension trait serait plus adapaté?
/// -> Pour l'instant impl env sur un env est mieux que le trait wrapper.

/// Fourni une impélémentation par default des méthodes.
pub trait Wrapper {
    type Environment: Gym3Environment;

    fn environment_mut(&mut self) -> &mut Self::Environment;
    fn environment(&self) -> &Self::Environment;

    fn act(&mut self, action: <Self::Environment as Gym3Environment>::Action) {
        self.environment_mut().act(action);
    }

    fn observe(
        &self,
    ) -> (
        f64,
        <Self::Environment as Gym3Environment>::Observation,
        bool,
    ) {
        self.environment().observe()
    }
}

struct TimingEnv {
    steps: u64,
    episode_len: u64,
}

impl TimingEnv {
    fn new(episode_len: u64) -> Self {
        Self {
            steps: 0,
            episode_len,
        }
    }
}

impl Gym3Environment for TimingEnv {
    type Action = ();

    type Observation = ();

    fn act(&mut self, _: Self::Action) {
        self.steps += 1;
        if self.steps >= self.episode_len {
            self.steps = 0;
        }
    }

    fn observe(&self) -> (f64, Self::Observation, bool) {
        let first = self.steps == 0;

        (0.0, (), first)
    }
}

struct RecordActs<E>
where
    E: Gym3Environment,
    E::Action: Clone,
{
    acts: Vec<E::Action>,
}

struct RecordActs2<E>
where
    E: Gym3Environment,
    E::Action: Clone,
{
    acts: Vec<E::Action>,
    env: E,
}

impl<E> Gym3Environment for RecordActs2<E>
where
    E: Gym3Environment,
    E::Action: Clone,
{
    type Action = E::Action;
    type Observation = E::Observation;

    fn act(&mut self, action: Self::Action) {
        self.acts.push(action.clone());
        self.env.act(action)
    }

    fn observe(&self) -> (f64, Self::Observation, bool) {
        self.env.observe()
    }
}

impl<E> RecordActs2<E>
where
    E: Gym3Environment,
    E::Action: Clone,
{
    pub fn new(env: E) -> Self {
        Self { acts: vec![], env }
    }
}

impl<E> Wrapper for RecordActs<E>
where
    E: Gym3Environment,
    E::Action: Clone,
{
    type Environment = E;

    fn act(&mut self, action: E::Action) {
        self.acts.push(action.clone());
        self.environment_mut().act(action);
    }

    fn environment_mut(&mut self) -> &mut Self::Environment {
        todo!()
    }

    fn environment(&self) -> &Self::Environment {
        todo!()
    }
}

impl<E> RecordActs<E>
where
    E: Gym3Environment,
    E::Action: Clone,
{
    pub fn new() -> Self {
        Self { acts: vec![] }
    }
}

use std::time::{Duration, Instant};

#[test]
fn fast() {
    const NUM_STEPS: u64 = 10000;
    const EPISODE_LEN: u64 = 100;

    let now = Instant::now();

    let mut env = TimingEnv::new(EPISODE_LEN);

    let mut episode_count = 0;
    let expected_episode_count = NUM_STEPS / EPISODE_LEN;

    for step in 0..NUM_STEPS {
        env.act(());
        let (_, _, first) = env.observe();
        if first {
            episode_count += 1;
        }
        if step == NUM_STEPS - 1 {
            assert_eq!(episode_count, expected_episode_count);
        }
    }

    let elapsed = now.elapsed().as_millis();

    println!("{elapsed}");
    // We assert we spend less than one miliseconds by step
    assert!((elapsed as f64 / NUM_STEPS as f64) < 1.0);
}

#[test]
fn wrapper() {
    const NUM_STEPS: u64 = 1000;
    const EPISODE_LEN: u64 = 10;

    let now = Instant::now();

    let mut env = TimingEnv::new(EPISODE_LEN);

    let mut env: RecordActs<TimingEnv> = RecordActs::new();

    let mut episode_count = 0;
    let expected_episode_count = NUM_STEPS / EPISODE_LEN;

    for step in 0..NUM_STEPS {
        env.act(());
        let (_, _, first) = env.observe();
        if first {
            episode_count += 1;
        }
        if step == NUM_STEPS - 1 {
            assert_eq!(episode_count, expected_episode_count);
        }
    }

    let elapsed = now.elapsed().as_millis();

    println!("{elapsed}");
    // We assert we spend less than one miliseconds by step
    assert!((elapsed as f64 / NUM_STEPS as f64) < 1.0);
}

#[test]
fn wrapper2() {
    const NUM_STEPS: u64 = 1000;
    const EPISODE_LEN: u64 = 10;

    let now = Instant::now();

    let env = TimingEnv::new(EPISODE_LEN);

    let mut env = RecordActs2::new(env);

    let mut episode_count = 0;
    let expected_episode_count = NUM_STEPS / EPISODE_LEN;

    for step in 0..NUM_STEPS {
        env.act(());
        let (_, _, first) = env.observe();
        if first {
            episode_count += 1;
        }
        if step == NUM_STEPS - 1 {
            assert_eq!(episode_count, expected_episode_count);
        }
    }

    let elapsed = now.elapsed().as_millis();

    println!("{elapsed}");
    assert_eq!(env.acts.len(), NUM_STEPS as usize);
    // We assert we spend less than one miliseconds by step
    assert!((elapsed as f64 / NUM_STEPS as f64) < 1.0);
}
