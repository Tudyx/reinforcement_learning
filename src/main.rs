// In python you can portailly update that's why environment initialization is divided into the constructor and the reset function.
// They choose to use an array of 2 elements to represent the cordonate in an array of 2 dimension. That's probably because they want to use tensor later.

// TODO: manage la seed.

// open question
// - Est-ce que je peux utiliser de trucs plus idiomatque pour les enum? Je pourrait faire des "From"
// - Est ce que l'action space je peux le definir à partir d'un associated constant?
// - Est ce que je fois parallelizer l'environment pour voir si cela raoute des contarintes?
mod algorithm;
mod easy_21;
mod environment;
mod viewer;

use std::collections::HashMap;

use ndarray::{array, Array, Array1};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand_isaac::isaac64::Isaac64Rng;

use environment::{Environment, RenderMode, Wrapper};

use egui::{Color32, Sense, Vec2};

// fn random_cordinate() -> Array1<i8> {}

struct GridWorldEnv {
    /// The size of the square grid.
    size: isize,
    // The location need to be signed be able to be sum with a direction.
    // ex de location [34, 12]
    agent_location: Array1<isize>,
    target_location: Array1<isize>,
    action_to_direction: HashMap<u8, Array1<isize>>,
    np_random: Isaac64Rng,
    terminated: bool,
}

impl eframe::App for GridWorldEnv {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Blue is looking for red");
            // let mut shape = Shape::circle_filled(Pos2::new(10.0, 10.0), 10.0, Color32::RED);

            egui::Grid::new("some_unique_id")
                .striped(true)
                .num_columns(self.size as usize)
                .show(ui, |ui| {
                    for x in 0..self.size {
                        for y in 0..self.size {
                            if self.agent_location == array![x, y] {
                                let radius = 10.0;
                                let size = Vec2::splat(2.0 * radius + 5.0);
                                let (rect, _response) = ui.allocate_at_least(size, Sense::hover());
                                ui.painter()
                                    .circle_filled(rect.center(), radius, Color32::BLUE);
                            } else if self.target_location == array![x, y] {
                                let radius = 10.0;
                                let size = Vec2::splat(2.0 * radius + 5.0);
                                let (rect, _response) = ui.allocate_at_least(size, Sense::hover());
                                ui.painter()
                                    .circle_filled(rect.center(), radius, Color32::RED);
                            } else {
                                ui.label("-");
                            }
                        }
                        ui.end_row();
                    }
                });
        });
    }
}

impl GridWorldEnv {
    /// Size is the size of the grid
    pub fn new(grid_size: isize, seed: Option<u64>) -> Self {
        let action_to_direction = HashMap::from([
            (0, array![1, 0]),
            (1, array![0, 1]),
            (2, array![-1, 0]),
            (3, array![0, -1]),
        ]);

        let mut env = Self {
            size: grid_size,
            action_to_direction,
            // This one will be initized in the reset function.
            agent_location: array![],
            target_location: array![],
            np_random: Isaac64Rng::seed_from_u64(0),
            terminated: true,
        };

        if let Some(seed) = seed {
            env.np_random = Isaac64Rng::seed_from_u64(seed);
        }
        env.reset(seed);
        env
    }
    fn reset(&mut self, seed: Option<u64>) {
        if let Some(seed) = seed {
            self.np_random = Isaac64Rng::seed_from_u64(seed);
        }
        self.agent_location =
            Array::random_using((2,), Uniform::new(0, self.size), &mut self.np_random);
        self.target_location =
            Array::random_using((2,), Uniform::new(0, self.size), &mut self.np_random);

        // If the location is the same we want to change it.
        while self.agent_location == self.target_location {
            self.target_location =
                Array::random_using((2,), Uniform::new(0, self.size), &mut self.np_random);
        }
    }
}

pub enum Action {
    Right,
    Up,
    Left,
    Down,
}

impl From<Action> for u8 {
    fn from(action: Action) -> Self {
        match action {
            Action::Right => 0,
            Action::Up => 1,
            Action::Left => 2,
            Action::Down => 3,
        }
    }
}

impl Environment for GridWorldEnv {
    type Action = u8;

    type Observation = HashMap<String, Array1<isize>>;

    fn act(&mut self, action: Self::Action) {
        // Map the action (element of {0,1,2,3}) to the direction we walk in
        let direction = self.action_to_direction[&action].clone();

        self.agent_location = self.agent_location.clone() + direction;

        // We clamp  the coordinate to make sure we don't leave the grid
        self.agent_location
            .mapv_inplace(|el| el.clamp(0, self.size - 1));

        self.terminated = self.agent_location == self.target_location;

        if self.terminated {
            self.reset(None);
        }
    }

    fn metadata() -> environment::Metadata {
        environment::Metadata {
            render_modes: vec![RenderMode::Human, RenderMode::RgbArray],
            render_fps: 4,
        }
    }

    // Si cela est fini, est ce que l'env doit être reset et on doit renvoyer la première observation?

    fn observe(&self) -> (f64, Self::Observation, bool) {
        let reward = if self.terminated { 1.0 } else { 0.0 };
        let observation = HashMap::from([
            ("agent".to_string(), self.agent_location.clone()),
            ("target".to_string(), self.target_location.clone()),
        ]);
        (reward, observation, self.terminated)
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

impl Environment for TimingEnv {
    type Action = ();

    type Observation = ();

    fn act(&mut self, _: Self::Action) {
        self.steps += 1;
        if self.steps >= self.episode_len {
            self.steps = 0;
        }
    }

    fn observe(&self) -> (f64, Self::Observation, bool) {
        let first = if self.steps == 0 { true } else { false };

        (0.0, (), first)
    }

    fn metadata() -> environment::Metadata {
        todo!()
    }
}

struct RecordActs<E>
where
    E: Environment,
    E::Action: Clone,
{
    acts: Vec<E::Action>,
}

struct RecordActs2<E>
where
    E: Environment,
    E::Action: Clone,
{
    acts: Vec<E::Action>,
    env: E,
}

impl<E> Environment for RecordActs2<E>
where
    E: Environment,
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

    fn metadata() -> environment::Metadata {
        E::metadata()
    }
}

impl<E> RecordActs2<E>
where
    E: Environment,
    E::Action: Clone,
{
    pub fn new(env: E) -> Self {
        Self { acts: vec![], env }
    }
}

impl<E> Wrapper for RecordActs<E>
where
    E: Environment,
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
    E: Environment,
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
fn train() {
    const NUM_STEPS: u64 = 1000;

    let mut grid_env = GridWorldEnv::new(4, None);

    for _ in 0..NUM_STEPS {
        grid_env.observe();
        grid_env.act(Action::Up.into());
    }
}

fn main() {
    let mut grid_env = GridWorldEnv::new(4, None);
    dbg!(grid_env.observe());
    grid_env.act(Action::Up.into());
    dbg!(grid_env.observe());

    train();
}
