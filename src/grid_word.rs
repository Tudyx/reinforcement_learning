use std::collections::HashMap;

use ndarray::{array, Array, Array1};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand_isaac::isaac64::Isaac64Rng;

use crate::environment::RenderMode;
use crate::Environment;

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

    fn metadata() -> crate::environment::Metadata {
        crate::environment::Metadata {
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

fn train() {
    const NUM_STEPS: u64 = 1000;

    let mut grid_env = GridWorldEnv::new(4, None);

    for _ in 0..NUM_STEPS {
        grid_env.observe();
        grid_env.act(Action::Up.into());
    }
}
